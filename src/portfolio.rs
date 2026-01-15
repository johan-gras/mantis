//! Portfolio management and position tracking.

use crate::error::{BacktestError, Result};
use crate::types::{
    AssetClass, AssetConfig, Bar, EquityPoint, Order, OrderType, Position, Side, Trade,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Additional cost configuration for futures markets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuturesCost {
    pub clearing_fee_per_contract: f64,
    pub exchange_fee_per_contract: f64,
    /// Annualized rate charged on margin capital (e.g., 0.05 = 5%).
    pub margin_interest_rate: f64,
}

impl Default for FuturesCost {
    fn default() -> Self {
        Self {
            clearing_fee_per_contract: 0.0,
            exchange_fee_per_contract: 0.0,
            margin_interest_rate: 0.0,
        }
    }
}

/// Additional cost configuration for crypto venues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoCost {
    pub maker_fee_pct: f64,
    pub taker_fee_pct: f64,
    pub withdrawal_fee: f64,
}

impl Default for CryptoCost {
    fn default() -> Self {
        Self {
            maker_fee_pct: 0.0,
            taker_fee_pct: 0.0,
            withdrawal_fee: 0.0,
        }
    }
}

/// Additional cost configuration for FX trading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForexCost {
    pub spread_pips: f64,
    /// Daily swap rate for long positions (decimal, e.g., -0.0001).
    pub swap_long: f64,
    /// Daily swap rate for short positions (decimal).
    pub swap_short: f64,
}

impl Default for ForexCost {
    fn default() -> Self {
        Self {
            spread_pips: 0.0,
            swap_long: 0.0,
            swap_short: 0.0,
        }
    }
}

/// Configuration for trade execution costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Commission per trade (flat fee).
    pub commission_flat: f64,
    /// Commission as percentage of trade value.
    pub commission_pct: f64,
    /// Slippage as percentage of price.
    pub slippage_pct: f64,
    /// Minimum commission per trade.
    pub min_commission: f64,
    /// Futures-specific fees and carrying costs.
    pub futures: FuturesCost,
    /// Crypto maker/taker fee model.
    pub crypto: CryptoCost,
    /// Forex spread/swap settings.
    pub forex: ForexCost,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            commission_flat: 0.0,
            commission_pct: 0.001, // 0.1% (10 bps)
            slippage_pct: 0.0005,  // 0.05% (5 bps)
            min_commission: 0.0,
            futures: FuturesCost::default(),
            crypto: CryptoCost::default(),
            forex: ForexCost::default(),
        }
    }
}

impl CostModel {
    /// Create a zero-cost model (no commissions or slippage).
    pub fn zero() -> Self {
        Self {
            commission_flat: 0.0,
            commission_pct: 0.0,
            slippage_pct: 0.0,
            min_commission: 0.0,
            futures: FuturesCost::default(),
            crypto: CryptoCost::default(),
            forex: ForexCost::default(),
        }
    }

    /// Calculate commission for a trade.
    pub fn calculate_commission(&self, trade_value: f64) -> f64 {
        let commission = self.commission_flat + trade_value * self.commission_pct;
        commission.max(self.min_commission)
    }

    /// Calculate slippage for a trade.
    pub fn calculate_slippage(&self, price: f64, side: Side) -> f64 {
        let slippage_amount = price * self.slippage_pct;
        match side {
            Side::Buy => slippage_amount,   // Pay more when buying
            Side::Sell => -slippage_amount, // Receive less when selling
        }
    }

    /// Get execution price after slippage.
    pub fn execution_price(&self, price: f64, side: Side) -> f64 {
        price + self.calculate_slippage(price, side)
    }

    /// Apply additional spread adjustments for certain asset classes.
    pub fn apply_asset_spread(&self, price: f64, side: Side, asset: &AssetConfig) -> f64 {
        match &asset.asset_class {
            AssetClass::Forex { pip_size, .. } if self.forex.spread_pips > 0.0 => {
                let half_spread = self.forex.spread_pips * pip_size * 0.5;
                match side {
                    Side::Buy => price + half_spread,
                    Side::Sell => price - half_spread,
                }
            }
            _ => price,
        }
    }

    /// Calculate maker/taker and per-contract fees based on asset class.
    pub fn additional_fees(
        &self,
        asset: &AssetConfig,
        quantity: f64,
        notional: f64,
        order_type: &OrderType,
    ) -> f64 {
        match &asset.asset_class {
            AssetClass::Future { .. } => {
                let per_contract =
                    self.futures.clearing_fee_per_contract + self.futures.exchange_fee_per_contract;
                per_contract * quantity
            }
            AssetClass::Crypto { .. } => {
                let is_taker = matches!(order_type, OrderType::Market | OrderType::Stop(_));
                let fee_pct = if is_taker {
                    self.crypto.taker_fee_pct
                } else {
                    self.crypto.maker_fee_pct
                };
                notional * fee_pct
            }
            _ => 0.0,
        }
    }

    /// Calculate FX swap cost for a closed trade.
    pub fn forex_swap_cost(
        &self,
        asset: &AssetConfig,
        side: Side,
        notional: f64,
        holding_days: f64,
    ) -> f64 {
        match &asset.asset_class {
            AssetClass::Forex { .. } if holding_days > 0.0 => {
                let rate = match side {
                    Side::Buy => self.forex.swap_long,
                    Side::Sell => self.forex.swap_short,
                };
                notional * rate * holding_days
            }
            _ => 0.0,
        }
    }
}

/// Portfolio state and position management.
#[derive(Debug)]
pub struct Portfolio {
    /// Available cash.
    pub cash: f64,
    /// Initial capital.
    pub initial_capital: f64,
    /// Current positions by symbol.
    positions: HashMap<String, Position>,
    /// All trades (including closed).
    trades: Vec<Trade>,
    /// Equity curve.
    equity_curve: Vec<EquityPoint>,
    /// Cost model for trade execution.
    cost_model: CostModel,
    /// Peak equity for drawdown calculation.
    peak_equity: f64,
    /// Allow short selling.
    pub allow_short: bool,
    /// Use fractional shares.
    pub fractional_shares: bool,
    /// Instrument-specific metadata.
    asset_configs: HashMap<String, AssetConfig>,
    /// Margin set aside for futures positions.
    margin_reserve: HashMap<String, f64>,
}

impl Portfolio {
    /// Create a new portfolio with initial capital.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            cash: initial_capital,
            initial_capital,
            positions: HashMap::new(),
            trades: Vec::new(),
            equity_curve: Vec::new(),
            cost_model: CostModel::default(),
            peak_equity: initial_capital,
            allow_short: true,
            fractional_shares: true,
            asset_configs: HashMap::new(),
            margin_reserve: HashMap::new(),
        }
    }

    /// Create a portfolio with custom cost model.
    pub fn with_cost_model(initial_capital: f64, cost_model: CostModel) -> Self {
        Self {
            cash: initial_capital,
            initial_capital,
            positions: HashMap::new(),
            trades: Vec::new(),
            equity_curve: Vec::new(),
            cost_model,
            peak_equity: initial_capital,
            allow_short: true,
            fractional_shares: true,
            asset_configs: HashMap::new(),
            margin_reserve: HashMap::new(),
        }
    }

    /// Set the cost model.
    pub fn set_cost_model(&mut self, cost_model: CostModel) {
        self.cost_model = cost_model;
    }

    /// Register or override the asset configuration for a symbol.
    pub fn set_asset_config(&mut self, config: AssetConfig) {
        self.asset_configs.insert(config.symbol.clone(), config);
    }

    /// Bulk update asset configurations from a data manager.
    pub fn set_asset_configs(&mut self, configs: &HashMap<String, AssetConfig>) {
        for config in configs.values() {
            self.set_asset_config(config.clone());
        }
    }

    fn asset_config_for(&self, symbol: &str) -> AssetConfig {
        self.asset_configs
            .get(symbol)
            .cloned()
            .unwrap_or_else(|| AssetConfig::equity(symbol))
    }

    fn position_value(&self, symbol: &str, pos: &Position, current_price: f64) -> f64 {
        let asset = self.asset_config_for(symbol);
        match asset.asset_class {
            AssetClass::Future { .. } => {
                let multiplier = asset.notional_multiplier();
                let direction = if pos.is_long() { 1.0 } else { -1.0 };
                let pnl =
                    direction * (current_price - pos.avg_entry_price) * pos.quantity * multiplier;
                let margin = self.margin_reserve.get(symbol).copied().unwrap_or(0.0);
                margin + pnl
            }
            _ => {
                let multiplier = asset.notional_multiplier();
                match pos.side {
                    Side::Buy => pos.quantity * current_price * multiplier,
                    Side::Sell => {
                        let synthetic = 2.0 * pos.avg_entry_price - current_price;
                        pos.quantity * synthetic * multiplier
                    }
                }
            }
        }
    }

    /// Get current equity (cash + positions value).
    pub fn equity(&self, prices: &HashMap<String, f64>) -> f64 {
        let positions_value: f64 = self
            .positions
            .iter()
            .map(|(symbol, pos)| {
                let price = prices.get(symbol).copied().unwrap_or(pos.avg_entry_price);
                self.position_value(symbol, pos, price)
            })
            .sum();
        self.cash + positions_value
    }

    /// Get position for a symbol.
    pub fn position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get position quantity for a symbol.
    pub fn position_qty(&self, symbol: &str) -> f64 {
        self.positions
            .get(symbol)
            .map(|p| match p.side {
                Side::Buy => p.quantity,
                Side::Sell => -p.quantity,
            })
            .unwrap_or(0.0)
    }

    /// Check if we have a position in a symbol.
    pub fn has_position(&self, symbol: &str) -> bool {
        self.positions.contains_key(symbol)
    }

    /// Get all positions.
    pub fn positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    /// Get all trades.
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Get closed trades only.
    pub fn closed_trades(&self) -> Vec<&Trade> {
        self.trades.iter().filter(|t| t.is_closed()).collect()
    }

    /// Get the equity curve.
    pub fn equity_curve(&self) -> &[EquityPoint] {
        &self.equity_curve
    }

    /// Execute an order.
    pub fn execute_order(&mut self, order: &Order, bar: &Bar) -> Result<Option<Trade>> {
        if !order.validate() {
            return Err(BacktestError::InvalidOrder(format!(
                "Invalid order: {:?}",
                order
            )));
        }

        let asset = self.asset_config_for(&order.symbol);
        let quantity = asset.normalize_quantity(order.quantity);
        if quantity <= 0.0 {
            return Ok(None);
        }

        let Some(base_price) = self.order_fill_price(order, bar) else {
            return Ok(None);
        };

        let price_with_spread = self
            .cost_model
            .apply_asset_spread(base_price, order.side, &asset);
        let exec_price = asset.normalize_price(
            self.cost_model
                .execution_price(price_with_spread, order.side),
        );
        let notional = exec_price * quantity * asset.notional_multiplier();
        let mut commission = self.cost_model.calculate_commission(notional);
        commission +=
            self.cost_model
                .additional_fees(&asset, quantity, notional, &order.order_type);
        let slippage = self.cost_model.slippage_pct * price_with_spread;

        if matches!(asset.asset_class, AssetClass::Future { .. }) {
            return self.execute_future_order(
                order, bar, quantity, exec_price, commission, slippage, &asset,
            );
        }

        self.execute_spot_order(
            order, bar, quantity, exec_price, notional, commission, slippage, &asset,
        )
    }

    fn execute_spot_order(
        &mut self,
        order: &Order,
        bar: &Bar,
        quantity: f64,
        exec_price: f64,
        notional: f64,
        commission: f64,
        slippage: f64,
        asset: &AssetConfig,
    ) -> Result<Option<Trade>> {
        match order.side {
            Side::Buy => self.spot_buy(
                order, bar, quantity, exec_price, notional, commission, slippage, asset,
            ),
            Side::Sell => self.spot_sell(
                order, bar, quantity, exec_price, notional, commission, slippage, asset,
            ),
        }
    }

    fn spot_buy(
        &mut self,
        order: &Order,
        bar: &Bar,
        quantity: f64,
        exec_price: f64,
        notional: f64,
        commission: f64,
        slippage: f64,
        asset: &AssetConfig,
    ) -> Result<Option<Trade>> {
        let total_cost = notional + commission;
        if total_cost > self.cash {
            return Err(BacktestError::InsufficientFunds {
                required: total_cost,
                available: self.cash,
            });
        }
        self.cash -= total_cost;

        let existing_position = self.positions.get(&order.symbol).cloned();
        let closing_qty = existing_position
            .as_ref()
            .filter(|pos| pos.is_short() && quantity >= pos.quantity)
            .map(|pos| pos.quantity)
            .unwrap_or(0.0);
        let opening_qty = (quantity - closing_qty).max(0.0);
        let per_unit_commission = if quantity > 0.0 {
            commission / quantity
        } else {
            0.0
        };
        let closing_commission = per_unit_commission * closing_qty;
        let opening_commission = (commission - closing_commission).max(0.0);

        if let Some(mut pos) = existing_position {
            if pos.is_short() {
                if quantity >= pos.quantity {
                    let closed_trade =
                        if let Some(trade) = self.find_open_trade_mut(&order.symbol, Side::Sell) {
                            trade.close(exec_price, bar.timestamp, closing_commission);
                            Some(trade.clone())
                        } else {
                            None
                        };
                    if let Some(trade) = closed_trade {
                        self.handle_trade_close_costs(&trade, asset, bar.timestamp);
                    }
                    self.positions.remove(&order.symbol);
                    self.handle_position_exit(&order.symbol, asset);
                    if opening_qty > 0.0 {
                        let pos = Position::new(&order.symbol, Side::Buy, opening_qty, exec_price);
                        self.positions.insert(order.symbol.clone(), pos);
                    } else {
                        return Ok(None);
                    }
                } else {
                    pos.quantity -= quantity;
                    self.positions.insert(order.symbol.clone(), pos);
                    return Ok(None);
                }
            } else {
                let new_qty = pos.quantity + quantity;
                let new_avg =
                    (pos.avg_entry_price * pos.quantity + exec_price * quantity) / new_qty;
                pos.quantity = new_qty;
                pos.avg_entry_price = new_avg;
                self.positions.insert(order.symbol.clone(), pos);
            }
        } else {
            let pos = Position::new(&order.symbol, Side::Buy, quantity, exec_price);
            self.positions.insert(order.symbol.clone(), pos);
        }

        if opening_qty > 0.0 {
            let trade = Trade::open(
                &order.symbol,
                Side::Buy,
                opening_qty,
                exec_price,
                bar.timestamp,
                opening_commission,
                slippage,
            );
            self.trades.push(trade.clone());
            debug!(
                "Executed BUY {} {} @ {:.4}",
                opening_qty, order.symbol, exec_price
            );
            Ok(Some(trade))
        } else {
            Ok(None)
        }
    }

    fn spot_sell(
        &mut self,
        order: &Order,
        bar: &Bar,
        quantity: f64,
        exec_price: f64,
        notional: f64,
        commission: f64,
        slippage: f64,
        asset: &AssetConfig,
    ) -> Result<Option<Trade>> {
        let existing_position = self.positions.get(&order.symbol).cloned();
        let closing_qty = existing_position
            .as_ref()
            .filter(|pos| pos.is_long() && quantity >= pos.quantity)
            .map(|pos| pos.quantity)
            .unwrap_or(0.0);
        let opening_qty = (quantity - closing_qty).max(0.0);
        if opening_qty > 0.0 && !self.allow_short {
            return Err(BacktestError::InvalidOrder(
                "Short selling not allowed".to_string(),
            ));
        }

        self.cash += notional - commission;

        let per_unit_commission = if quantity > 0.0 {
            commission / quantity
        } else {
            0.0
        };
        let closing_commission = per_unit_commission * closing_qty;
        let opening_commission = (commission - closing_commission).max(0.0);

        if let Some(mut pos) = existing_position {
            if pos.is_long() {
                if quantity >= pos.quantity {
                    let closed_trade =
                        if let Some(trade) = self.find_open_trade_mut(&order.symbol, Side::Buy) {
                            trade.close(exec_price, bar.timestamp, closing_commission);
                            Some(trade.clone())
                        } else {
                            None
                        };
                    if let Some(trade) = closed_trade {
                        self.handle_trade_close_costs(&trade, asset, bar.timestamp);
                    }
                    self.positions.remove(&order.symbol);
                    self.handle_position_exit(&order.symbol, asset);
                    if opening_qty > 0.0 {
                        let pos = Position::new(&order.symbol, Side::Sell, opening_qty, exec_price);
                        self.positions.insert(order.symbol.clone(), pos);
                    } else {
                        return Ok(None);
                    }
                } else {
                    pos.quantity -= quantity;
                    self.positions.insert(order.symbol.clone(), pos);
                    return Ok(None);
                }
            } else {
                let new_qty = pos.quantity + quantity;
                let new_avg =
                    (pos.avg_entry_price * pos.quantity + exec_price * quantity) / new_qty;
                pos.quantity = new_qty;
                pos.avg_entry_price = new_avg;
                self.positions.insert(order.symbol.clone(), pos);
            }
        } else if opening_qty > 0.0 {
            let pos = Position::new(&order.symbol, Side::Sell, opening_qty, exec_price);
            self.positions.insert(order.symbol.clone(), pos);
        }

        if opening_qty > 0.0 {
            let trade = Trade::open(
                &order.symbol,
                Side::Sell,
                opening_qty,
                exec_price,
                bar.timestamp,
                opening_commission,
                slippage,
            );
            self.trades.push(trade.clone());
            debug!(
                "Executed SELL {} {} @ {:.4}",
                opening_qty, order.symbol, exec_price
            );
            Ok(Some(trade))
        } else {
            Ok(None)
        }
    }

    fn execute_future_order(
        &mut self,
        order: &Order,
        bar: &Bar,
        quantity: f64,
        exec_price: f64,
        commission: f64,
        slippage: f64,
        asset: &AssetConfig,
    ) -> Result<Option<Trade>> {
        if commission > self.cash {
            return Err(BacktestError::InsufficientFunds {
                required: commission,
                available: self.cash,
            });
        }
        self.cash -= commission;

        let multiplier = asset.notional_multiplier();
        let mut qty_remaining = quantity;
        let per_unit_commission = if quantity > 0.0 {
            commission / quantity
        } else {
            0.0
        };

        if let Some(existing) = self.positions.get(&order.symbol).cloned() {
            match (order.side, existing.side) {
                (Side::Buy, Side::Sell) => {
                    let close_qty = qty_remaining.min(existing.quantity);
                    if close_qty > 0.0 {
                        let pnl = (existing.avg_entry_price - exec_price) * close_qty * multiplier;
                        self.cash += pnl;
                        qty_remaining -= close_qty;
                        if close_qty >= existing.quantity - f64::EPSILON {
                            let closed_trade = if let Some(trade) =
                                self.find_open_trade_mut(&order.symbol, Side::Sell)
                            {
                                trade.close(
                                    exec_price,
                                    bar.timestamp,
                                    per_unit_commission * close_qty,
                                );
                                Some(trade.clone())
                            } else {
                                None
                            };
                            if let Some(trade) = closed_trade {
                                self.handle_trade_close_costs(&trade, asset, bar.timestamp);
                            }
                            self.positions.remove(&order.symbol);
                            self.update_future_margin(&order.symbol, asset, None)?;
                            self.handle_position_exit(&order.symbol, asset);
                        } else {
                            let mut pos = existing.clone();
                            pos.quantity -= close_qty;
                            self.positions.insert(order.symbol.clone(), pos.clone());
                            self.update_future_margin(&order.symbol, asset, Some(&pos))?;
                        }
                    }
                }
                (Side::Sell, Side::Buy) => {
                    let close_qty = qty_remaining.min(existing.quantity);
                    if close_qty > 0.0 {
                        let pnl = (exec_price - existing.avg_entry_price) * close_qty * multiplier;
                        self.cash += pnl;
                        qty_remaining -= close_qty;
                        if close_qty >= existing.quantity - f64::EPSILON {
                            let closed_trade = if let Some(trade) =
                                self.find_open_trade_mut(&order.symbol, Side::Buy)
                            {
                                trade.close(
                                    exec_price,
                                    bar.timestamp,
                                    per_unit_commission * close_qty,
                                );
                                Some(trade.clone())
                            } else {
                                None
                            };
                            if let Some(trade) = closed_trade {
                                self.handle_trade_close_costs(&trade, asset, bar.timestamp);
                            }
                            self.positions.remove(&order.symbol);
                            self.update_future_margin(&order.symbol, asset, None)?;
                            self.handle_position_exit(&order.symbol, asset);
                        } else {
                            let mut pos = existing.clone();
                            pos.quantity -= close_qty;
                            self.positions.insert(order.symbol.clone(), pos.clone());
                            self.update_future_margin(&order.symbol, asset, Some(&pos))?;
                        }
                    }
                }
                _ => {}
            }
        }

        if qty_remaining <= f64::EPSILON {
            return Ok(None);
        }

        let mut created_trade = None;
        if let Some(mut pos) = self.positions.get(&order.symbol).cloned() {
            if pos.side == order.side {
                let new_qty = pos.quantity + qty_remaining;
                let new_avg =
                    (pos.avg_entry_price * pos.quantity + exec_price * qty_remaining) / new_qty;
                pos.quantity = new_qty;
                pos.avg_entry_price = new_avg;
                self.positions.insert(order.symbol.clone(), pos.clone());
                self.update_future_margin(&order.symbol, asset, Some(&pos))?;
            } else {
                let pos = Position::new(&order.symbol, order.side, qty_remaining, exec_price);
                self.positions.insert(order.symbol.clone(), pos.clone());
                self.update_future_margin(&order.symbol, asset, Some(&pos))?;
            }
        } else {
            let pos = Position::new(&order.symbol, order.side, qty_remaining, exec_price);
            self.positions.insert(order.symbol.clone(), pos.clone());
            self.update_future_margin(&order.symbol, asset, Some(&pos))?;
        }

        if qty_remaining > 0.0 {
            let trade = Trade::open(
                &order.symbol,
                order.side,
                qty_remaining,
                exec_price,
                bar.timestamp,
                per_unit_commission * qty_remaining,
                slippage,
            );
            self.trades.push(trade.clone());
            debug!(
                "Executed FUTURE {:?} {} {} @ {:.4}",
                order.side, qty_remaining, order.symbol, exec_price
            );
            created_trade = Some(trade);
        }

        Ok(created_trade)
    }

    fn update_future_margin(
        &mut self,
        symbol: &str,
        asset: &AssetConfig,
        position: Option<&Position>,
    ) -> Result<()> {
        if let Some(requirement) = asset.margin_requirement() {
            let previous = self.margin_reserve.get(symbol).copied().unwrap_or(0.0);
            let new_margin = position
                .filter(|pos| pos.quantity > 0.0)
                .map(|pos| {
                    pos.avg_entry_price * pos.quantity * asset.notional_multiplier() * requirement
                })
                .unwrap_or(0.0);

            if new_margin > previous {
                let delta = new_margin - previous;
                if delta > self.cash {
                    return Err(BacktestError::InsufficientFunds {
                        required: delta,
                        available: self.cash,
                    });
                }
                self.cash -= delta;
            } else if new_margin < previous {
                self.cash += previous - new_margin;
            }

            if new_margin > 0.0 {
                self.margin_reserve.insert(symbol.to_string(), new_margin);
            } else {
                self.margin_reserve.remove(symbol);
            }
        }
        Ok(())
    }

    fn handle_trade_close_costs(
        &mut self,
        trade: &Trade,
        asset: &AssetConfig,
        _close_time: DateTime<Utc>,
    ) {
        if let Some(period) = trade.holding_period() {
            let days = period.num_seconds().max(0) as f64 / 86_400.0;
            if days > 0.0 {
                let notional = trade.entry_price * trade.quantity * asset.notional_multiplier();
                let swap = self
                    .cost_model
                    .forex_swap_cost(asset, trade.side, notional, days);
                if swap != 0.0 {
                    self.cash -= swap;
                }

                if let AssetClass::Future { .. } = asset.asset_class {
                    if let Some(margin) = asset.margin_requirement() {
                        let rate = self.cost_model.futures.margin_interest_rate;
                        if rate > 0.0 {
                            let interest = notional * margin * rate * (days / 365.0);
                            self.cash -= interest;
                        }
                    }
                }
            }
        }
    }

    fn handle_position_exit(&mut self, symbol: &str, asset: &AssetConfig) {
        if let AssetClass::Crypto { .. } = asset.asset_class {
            if self.cost_model.crypto.withdrawal_fee > 0.0 {
                self.cash -= self.cost_model.crypto.withdrawal_fee;
            }
        }
        if matches!(asset.asset_class, AssetClass::Future { .. }) {
            self.margin_reserve.remove(symbol);
        }
    }

    fn find_open_trade_mut(&mut self, symbol: &str, side: Side) -> Option<&mut Trade> {
        self.trades
            .iter_mut()
            .rev()
            .find(|t| t.symbol == symbol && !t.is_closed() && t.side == side)
    }

    fn order_fill_price(&self, order: &Order, bar: &Bar) -> Option<f64> {
        match order.order_type {
            OrderType::Market => Some(bar.open),
            OrderType::Limit(limit) => match order.side {
                Side::Buy => {
                    if bar.low <= limit.0 {
                        Some(limit.0.min(bar.open))
                    } else {
                        None
                    }
                }
                Side::Sell => {
                    if bar.high >= limit.0 {
                        Some(limit.0.max(bar.open))
                    } else {
                        None
                    }
                }
            },
            OrderType::Stop(stop) => match order.side {
                Side::Buy => {
                    if bar.high >= stop.0 {
                        Some(stop.0.max(bar.open))
                    } else {
                        None
                    }
                }
                Side::Sell => {
                    if bar.low <= stop.0 {
                        Some(stop.0.min(bar.open))
                    } else {
                        None
                    }
                }
            },
            OrderType::StopLimit { stop, limit } => {
                let triggered = match order.side {
                    Side::Buy => bar.high >= stop.0,
                    Side::Sell => bar.low <= stop.0,
                };
                if !triggered {
                    return None;
                }
                match order.side {
                    Side::Buy => {
                        if bar.low <= limit.0 {
                            Some(limit.0.min(bar.open))
                        } else {
                            None
                        }
                    }
                    Side::Sell => {
                        if bar.high >= limit.0 {
                            Some(limit.0.max(bar.open))
                        } else {
                            None
                        }
                    }
                }
            }
        }
    }
    /// Record an equity point.
    pub fn record_equity(&mut self, timestamp: DateTime<Utc>, prices: &HashMap<String, f64>) {
        let positions_value: f64 = self
            .positions
            .iter()
            .map(|(symbol, pos)| {
                let price = prices.get(symbol).copied().unwrap_or(pos.avg_entry_price);
                self.position_value(symbol, pos, price)
            })
            .sum();

        let equity = self.cash + positions_value;
        self.peak_equity = self.peak_equity.max(equity);

        let drawdown = self.peak_equity - equity;
        let drawdown_pct = if self.peak_equity > 0.0 {
            drawdown / self.peak_equity * 100.0
        } else {
            0.0
        };

        let point = EquityPoint {
            timestamp,
            equity,
            cash: self.cash,
            positions_value,
            drawdown,
            drawdown_pct,
        };

        self.equity_curve.push(point);
    }

    /// Calculate position size based on risk.
    pub fn calculate_position_size(&self, price: f64, risk_pct: f64, stop_loss_pct: f64) -> f64 {
        let risk_amount = self.cash * risk_pct;
        let risk_per_share = price * stop_loss_pct;
        let shares = risk_amount / risk_per_share;

        if self.fractional_shares {
            shares
        } else {
            shares.floor()
        }
    }

    /// Get a summary of portfolio state.
    pub fn summary(&self, prices: &HashMap<String, f64>) -> PortfolioSummary {
        let equity = self.equity(prices);
        let total_return = (equity - self.initial_capital) / self.initial_capital * 100.0;

        let closed = self.closed_trades();
        let winning: Vec<_> = closed
            .iter()
            .filter(|t| t.net_pnl().unwrap_or(0.0) > 0.0)
            .collect();
        let losing: Vec<_> = closed
            .iter()
            .filter(|t| t.net_pnl().unwrap_or(0.0) < 0.0)
            .collect();

        let win_rate = if !closed.is_empty() {
            winning.len() as f64 / closed.len() as f64 * 100.0
        } else {
            0.0
        };

        let avg_win = if !winning.is_empty() {
            winning.iter().filter_map(|t| t.net_pnl()).sum::<f64>() / winning.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing.is_empty() {
            losing.iter().filter_map(|t| t.net_pnl()).sum::<f64>() / losing.len() as f64
        } else {
            0.0
        };

        PortfolioSummary {
            initial_capital: self.initial_capital,
            final_equity: equity,
            cash: self.cash,
            total_return,
            total_trades: closed.len(),
            winning_trades: winning.len(),
            losing_trades: losing.len(),
            win_rate,
            avg_win,
            avg_loss,
        }
    }
}

/// Portfolio summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub initial_capital: f64,
    pub final_equity: f64,
    pub cash: f64,
    pub total_return: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssetClass, AssetConfig, Bar};
    use chrono::{TimeZone, Utc};

    fn sample_bar() -> Bar {
        Bar::new(
            Utc.with_ymd_and_hms(2024, 1, 15, 9, 30, 0).unwrap(),
            100.0,
            105.0,
            98.0,
            102.0,
            1000.0,
        )
    }

    #[test]
    fn test_portfolio_creation() {
        let portfolio = Portfolio::new(100000.0);
        assert_eq!(portfolio.cash, 100000.0);
        assert_eq!(portfolio.initial_capital, 100000.0);
        assert!(portfolio.positions.is_empty());
    }

    #[test]
    fn test_cost_model() {
        let cost_model = CostModel::default();

        let commission = cost_model.calculate_commission(10000.0);
        assert!((commission - 10.0).abs() < 0.01); // 0.1% of 10000

        let exec_price = cost_model.execution_price(100.0, Side::Buy);
        assert!(exec_price > 100.0); // Should add slippage for buy
    }

    #[test]
    fn test_market_order_execution() {
        let mut portfolio = Portfolio::with_cost_model(100000.0, CostModel::zero());
        let bar = sample_bar();
        let order = Order::market("AAPL", Side::Buy, 100.0, bar.timestamp);

        let trade = portfolio.execute_order(&order, &bar).unwrap();
        assert!(trade.is_some());

        let trade = trade.unwrap();
        assert_eq!(trade.symbol, "AAPL");
        assert_eq!(trade.quantity, 100.0);
        assert_eq!(trade.entry_price, 100.0); // Open price

        assert!(portfolio.has_position("AAPL"));
        assert_eq!(portfolio.position_qty("AAPL"), 100.0);
        assert_eq!(portfolio.cash, 90000.0); // 100000 - 100*100
    }

    #[test]
    fn test_position_close() {
        let mut portfolio = Portfolio::with_cost_model(100000.0, CostModel::zero());
        let bar = sample_bar();

        // Open long
        let buy_order = Order::market("AAPL", Side::Buy, 100.0, bar.timestamp);
        portfolio.execute_order(&buy_order, &bar).unwrap();

        // Close long
        let sell_bar = Bar::new(
            Utc.with_ymd_and_hms(2024, 1, 16, 9, 30, 0).unwrap(),
            110.0,
            115.0,
            108.0,
            112.0,
            1000.0,
        );
        let sell_order = Order::market("AAPL", Side::Sell, 100.0, sell_bar.timestamp);
        portfolio.execute_order(&sell_order, &sell_bar).unwrap();

        assert!(!portfolio.has_position("AAPL"));
        // Started with 100k, bought at 100, sold at 110, profit = 1000
        assert_eq!(portfolio.cash, 101000.0);
    }

    #[test]
    fn test_insufficient_funds() {
        let mut portfolio = Portfolio::with_cost_model(1000.0, CostModel::zero());
        let bar = sample_bar();
        let order = Order::market("AAPL", Side::Buy, 100.0, bar.timestamp); // Need 10000

        let result = portfolio.execute_order(&order, &bar);
        assert!(matches!(
            result,
            Err(BacktestError::InsufficientFunds { .. })
        ));
    }

    #[test]
    fn test_equity_recording() {
        let mut portfolio = Portfolio::new(100000.0);
        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 150.0);

        let timestamp = Utc.with_ymd_and_hms(2024, 1, 15, 9, 30, 0).unwrap();
        portfolio.record_equity(timestamp, &prices);

        assert_eq!(portfolio.equity_curve.len(), 1);
        assert_eq!(portfolio.equity_curve[0].equity, 100000.0);
    }

    #[test]
    fn test_position_sizing() {
        let portfolio = Portfolio::new(100000.0);

        // Risk 1% of portfolio with 5% stop loss
        let size = portfolio.calculate_position_size(100.0, 0.01, 0.05);
        // Risk amount = 1000, risk per share = 5, size = 200
        assert!((size - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_futures_margin_handling() {
        let mut portfolio = Portfolio::with_cost_model(100000.0, CostModel::zero());
        let config = AssetConfig::new(
            "ES",
            AssetClass::Future {
                multiplier: 50.0,
                tick_size: 0.25,
                margin_requirement: 0.1,
            },
        );
        portfolio.set_asset_config(config);

        let bar = sample_bar();
        let buy_order = Order::market("ES", Side::Buy, 1.0, bar.timestamp);
        portfolio.execute_order(&buy_order, &bar).unwrap();

        assert!(portfolio.has_position("ES"));
        let reserve = portfolio.margin_reserve.get("ES").copied().unwrap();
        assert!((reserve - 500.0).abs() < 1e-6);
        assert!((portfolio.cash - 99500.0).abs() < 1e-6);

        let exit_bar = Bar::new(
            bar.timestamp + chrono::Duration::days(1),
            110.0,
            112.0,
            109.0,
            110.0,
            1000.0,
        );
        let sell_order = Order::market("ES", Side::Sell, 1.0, exit_bar.timestamp);
        portfolio.execute_order(&sell_order, &exit_bar).unwrap();

        assert!(!portfolio.has_position("ES"));
        assert!(portfolio.margin_reserve.get("ES").is_none());
        assert!((portfolio.cash - 100500.0).abs() < 1e-6);
    }

    #[test]
    fn test_crypto_withdrawal_fee() {
        let mut cost_model = CostModel::zero();
        cost_model.crypto.withdrawal_fee = 5.0;
        let mut portfolio = Portfolio::with_cost_model(1000.0, cost_model);
        let config = AssetConfig::new(
            "BTC",
            AssetClass::Crypto {
                base_precision: 8,
                quote_precision: 2,
            },
        );
        portfolio.set_asset_config(config);

        let bar = sample_bar();
        let buy_order = Order::market("BTC", Side::Buy, 1.0, bar.timestamp);
        portfolio.execute_order(&buy_order, &bar).unwrap();

        let exit_bar = Bar::new(
            bar.timestamp + chrono::Duration::days(1),
            120.0,
            125.0,
            119.0,
            120.0,
            1000.0,
        );
        let sell_order = Order::market("BTC", Side::Sell, 1.0, exit_bar.timestamp);
        portfolio.execute_order(&sell_order, &exit_bar).unwrap();

        assert!(!portfolio.has_position("BTC"));
        assert!((portfolio.cash - 1015.0).abs() < 1e-6);
    }
}

//! Portfolio management and position tracking.

use crate::error::{BacktestError, Result};
use crate::types::{Bar, EquityPoint, Order, OrderType, Position, Side, Trade};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

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
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            commission_flat: 0.0,
            commission_pct: 0.001, // 0.1% (10 bps)
            slippage_pct: 0.0005,  // 0.05% (5 bps)
            min_commission: 0.0,
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
        }
    }

    /// Set the cost model.
    pub fn set_cost_model(&mut self, cost_model: CostModel) {
        self.cost_model = cost_model;
    }

    /// Get current equity (cash + positions value).
    pub fn equity(&self, prices: &HashMap<String, f64>) -> f64 {
        let positions_value: f64 = self
            .positions
            .iter()
            .map(|(symbol, pos)| {
                let price = prices.get(symbol).copied().unwrap_or(pos.avg_entry_price);
                pos.market_value(price)
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

        // Determine execution price based on order type
        let base_price = match order.order_type {
            OrderType::Market => bar.open, // Execute at open of next bar
            OrderType::Limit(ref limit) => {
                // Check if limit can be filled
                match order.side {
                    Side::Buy => {
                        if bar.low <= limit.0 {
                            limit.0.min(bar.open) // Fill at limit or open if gapped down
                        } else {
                            return Ok(None); // Not filled
                        }
                    }
                    Side::Sell => {
                        if bar.high >= limit.0 {
                            limit.0.max(bar.open) // Fill at limit or open if gapped up
                        } else {
                            return Ok(None); // Not filled
                        }
                    }
                }
            }
            OrderType::Stop(ref stop) => {
                // Check if stop is triggered
                match order.side {
                    Side::Buy => {
                        if bar.high >= stop.0 {
                            stop.0.max(bar.open) // Triggered, fill at stop or open
                        } else {
                            return Ok(None); // Not triggered
                        }
                    }
                    Side::Sell => {
                        if bar.low <= stop.0 {
                            stop.0.min(bar.open) // Triggered, fill at stop or open
                        } else {
                            return Ok(None); // Not triggered
                        }
                    }
                }
            }
            OrderType::StopLimit {
                ref stop,
                ref limit,
            } => {
                // First check if stop is triggered, then check limit
                let triggered = match order.side {
                    Side::Buy => bar.high >= stop.0,
                    Side::Sell => bar.low <= stop.0,
                };
                if !triggered {
                    return Ok(None);
                }
                // Now act as limit order
                match order.side {
                    Side::Buy => {
                        if bar.low <= limit.0 {
                            limit.0.min(bar.open)
                        } else {
                            return Ok(None);
                        }
                    }
                    Side::Sell => {
                        if bar.high >= limit.0 {
                            limit.0.max(bar.open)
                        } else {
                            return Ok(None);
                        }
                    }
                }
            }
        };

        // Apply slippage
        let exec_price = self.cost_model.execution_price(base_price, order.side);
        let trade_value = exec_price * order.quantity;
        let commission = self.cost_model.calculate_commission(trade_value);
        let slippage = self.cost_model.slippage_pct * base_price;

        // Check if we have an existing position
        let existing_position = self.positions.get(&order.symbol).cloned();

        match order.side {
            Side::Buy => {
                // Check if we have enough cash
                let total_cost = trade_value + commission;
                if total_cost > self.cash {
                    return Err(BacktestError::InsufficientFunds {
                        required: total_cost,
                        available: self.cash,
                    });
                }

                self.cash -= total_cost;

                if let Some(mut pos) = existing_position {
                    if pos.is_short() {
                        // Closing or reducing short position
                        if order.quantity >= pos.quantity {
                            // Close short completely, possibly go long
                            let close_qty = pos.quantity;
                            let remaining = order.quantity - close_qty;

                            // Close the short trade
                            if let Some(trade) = self.trades.iter_mut().rev().find(|t| {
                                t.symbol == order.symbol && !t.is_closed() && t.side == Side::Sell
                            }) {
                                trade.close(exec_price, bar.timestamp, commission);
                            }

                            self.positions.remove(&order.symbol);

                            if remaining > 0.0 {
                                // Open new long position
                                let new_pos =
                                    Position::new(&order.symbol, Side::Buy, remaining, exec_price);
                                self.positions.insert(order.symbol.clone(), new_pos);

                                let trade = Trade::open(
                                    &order.symbol,
                                    Side::Buy,
                                    remaining,
                                    exec_price,
                                    bar.timestamp,
                                    commission,
                                    slippage,
                                );
                                self.trades.push(trade.clone());
                                return Ok(Some(trade));
                            }
                        } else {
                            // Reduce short position
                            pos.quantity -= order.quantity;
                            self.positions.insert(order.symbol.clone(), pos);
                        }
                    } else {
                        // Adding to long position
                        let new_qty = pos.quantity + order.quantity;
                        let new_avg = (pos.avg_entry_price * pos.quantity
                            + exec_price * order.quantity)
                            / new_qty;
                        pos.quantity = new_qty;
                        pos.avg_entry_price = new_avg;
                        self.positions.insert(order.symbol.clone(), pos);
                    }
                } else {
                    // New long position
                    let pos = Position::new(&order.symbol, Side::Buy, order.quantity, exec_price);
                    self.positions.insert(order.symbol.clone(), pos);
                }

                let trade = Trade::open(
                    &order.symbol,
                    Side::Buy,
                    order.quantity,
                    exec_price,
                    bar.timestamp,
                    commission,
                    slippage,
                );
                self.trades.push(trade.clone());
                debug!(
                    "Executed BUY {} {} @ {:.2}",
                    order.quantity, order.symbol, exec_price
                );
                Ok(Some(trade))
            }
            Side::Sell => {
                if !self.allow_short && !self.has_position(&order.symbol) {
                    return Err(BacktestError::InvalidOrder(
                        "Short selling not allowed".to_string(),
                    ));
                }

                if let Some(mut pos) = existing_position {
                    if pos.is_long() {
                        // Closing or reducing long position
                        if order.quantity >= pos.quantity {
                            // Close long completely, possibly go short
                            let close_qty = pos.quantity;
                            let remaining = order.quantity - close_qty;

                            // Close the long trade
                            if let Some(trade) = self.trades.iter_mut().rev().find(|t| {
                                t.symbol == order.symbol && !t.is_closed() && t.side == Side::Buy
                            }) {
                                trade.close(exec_price, bar.timestamp, commission);
                            }

                            self.cash += exec_price * close_qty - commission;
                            self.positions.remove(&order.symbol);

                            if remaining > 0.0 && self.allow_short {
                                // Open new short position
                                let new_pos =
                                    Position::new(&order.symbol, Side::Sell, remaining, exec_price);
                                self.positions.insert(order.symbol.clone(), new_pos);
                                self.cash += exec_price * remaining; // Receive cash for short

                                let trade = Trade::open(
                                    &order.symbol,
                                    Side::Sell,
                                    remaining,
                                    exec_price,
                                    bar.timestamp,
                                    commission,
                                    slippage,
                                );
                                self.trades.push(trade.clone());
                                return Ok(Some(trade));
                            }
                        } else {
                            // Reduce long position
                            pos.quantity -= order.quantity;
                            self.positions.insert(order.symbol.clone(), pos);
                            self.cash += exec_price * order.quantity - commission;
                        }

                        let trade = Trade::open(
                            &order.symbol,
                            Side::Sell,
                            order.quantity,
                            exec_price,
                            bar.timestamp,
                            commission,
                            slippage,
                        );
                        debug!(
                            "Executed SELL {} {} @ {:.2}",
                            order.quantity, order.symbol, exec_price
                        );
                        return Ok(Some(trade));
                    } else {
                        // Adding to short position
                        let new_qty = pos.quantity + order.quantity;
                        let new_avg = (pos.avg_entry_price * pos.quantity
                            + exec_price * order.quantity)
                            / new_qty;
                        pos.quantity = new_qty;
                        pos.avg_entry_price = new_avg;
                        self.positions.insert(order.symbol.clone(), pos);
                        self.cash += exec_price * order.quantity - commission;
                    }
                } else if self.allow_short {
                    // New short position
                    let pos = Position::new(&order.symbol, Side::Sell, order.quantity, exec_price);
                    self.positions.insert(order.symbol.clone(), pos);
                    self.cash += exec_price * order.quantity - commission;
                }

                let trade = Trade::open(
                    &order.symbol,
                    Side::Sell,
                    order.quantity,
                    exec_price,
                    bar.timestamp,
                    commission,
                    slippage,
                );
                self.trades.push(trade.clone());
                debug!(
                    "Executed SELL {} {} @ {:.2}",
                    order.quantity, order.symbol, exec_price
                );
                Ok(Some(trade))
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
                match pos.side {
                    Side::Buy => pos.quantity * price,
                    Side::Sell => pos.quantity * (2.0 * pos.avg_entry_price - price), // Short P&L
                }
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
    use chrono::{TimeZone, Utc};

    fn sample_bar() -> Bar {
        use crate::types::Bar;
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
}

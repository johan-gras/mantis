//! Portfolio management and position tracking.

use crate::error::{BacktestError, Result};
use crate::types::{
    AssetClass, AssetConfig, Bar, EquityPoint, ExecutionPrice, LotSelectionMethod, Order,
    OrderType, Position, Side, TaxLot, Trade, VolumeProfile,
};
use chrono::{DateTime, Utc};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
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

/// Configuration describing Reg T and portfolio margin rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginConfig {
    /// Enable equity/spot margin calculations.
    pub enabled: bool,
    /// Reg T initial requirement for long positions (e.g., 0.50).
    pub reg_t_long_initial: f64,
    /// Reg T initial requirement for short positions (e.g., 1.50).
    pub reg_t_short_initial: f64,
    /// Maintenance margin percentage for long positions.
    pub maintenance_long_pct: f64,
    /// Maintenance margin percentage for short positions.
    pub maintenance_short_pct: f64,
    /// Maximum allowable leverage (gross exposure / equity).
    pub max_leverage: f64,
    /// Enable portfolio margin checks.
    pub use_portfolio_margin: bool,
    /// Portfolio margin requirement as percent of gross exposure.
    pub portfolio_margin_pct: f64,
    /// Annualized interest rate charged on borrowed capital.
    pub interest_rate: f64,
}

impl Default for MarginConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reg_t_long_initial: 0.5,
            reg_t_short_initial: 1.5,
            maintenance_long_pct: 0.25,
            maintenance_short_pct: 0.30,
            max_leverage: 2.0,
            use_portfolio_margin: false,
            portfolio_margin_pct: 0.15,
            interest_rate: 0.03,
        }
    }
}

/// Snapshot of portfolio margin exposure.
#[derive(Debug, Clone, Default)]
pub struct MarginState {
    pub equity: f64,
    pub gross_exposure: f64,
    pub reg_t_requirement: f64,
    pub maintenance_requirement: f64,
    pub portfolio_requirement: f64,
    pub leverage: f64,
}

/// Available market impact models for execution pricing.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MarketImpactModel {
    /// Disable market impact adjustments.
    #[default]
    None,
    /// Linear model where impact grows directly with relative order size.
    Linear { coefficient: f64 },
    /// Square-root model, commonly used for large orders.
    SquareRoot { coefficient: f64 },
    /// Almgren-Chriss implementation using volatility and temporary/permanent impact.
    AlmgrenChriss { sigma: f64, eta: f64, gamma: f64 },
}

impl MarketImpactModel {
    fn impact(&self, order_size: f64, avg_volume: f64, price: f64) -> f64 {
        if order_size <= 0.0 || avg_volume <= f64::EPSILON {
            return 0.0;
        }

        let relative_size = (order_size / avg_volume).abs();
        match self {
            MarketImpactModel::None => 0.0,
            MarketImpactModel::Linear { coefficient } => price * coefficient * relative_size,
            MarketImpactModel::SquareRoot { coefficient } => {
                price * coefficient * relative_size.sqrt()
            }
            MarketImpactModel::AlmgrenChriss { sigma, eta, gamma } => {
                let sqrt_component = relative_size.sqrt();
                price * (eta * relative_size + gamma * sigma * sqrt_component)
            }
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
    /// Market impact model configuration.
    #[serde(default)]
    pub market_impact: MarketImpactModel,
    /// Maximum participation rate as fraction of bar volume (e.g., 0.10 = 10%).
    /// None means no limit. Prevents unrealistic large order fills in illiquid markets.
    #[serde(default)]
    pub max_volume_participation: Option<f64>,
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
            market_impact: MarketImpactModel::default(),
            max_volume_participation: None, // No limit by default
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
            market_impact: MarketImpactModel::None,
            max_volume_participation: None,
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

    /// Calculate additional market impact adjustment in absolute price terms.
    pub fn calculate_market_impact(&self, order_size: f64, avg_volume: f64, price: f64) -> f64 {
        self.market_impact.impact(order_size, avg_volume, price)
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

    /// Calculate the maximum allowed quantity based on volume participation limits.
    /// Returns the original quantity if no limit is set.
    /// Returns 0.0 if bar volume is zero (to prevent unrealistic fills).
    pub fn apply_volume_participation_limit(
        &self,
        requested_quantity: f64,
        bar_volume: f64,
    ) -> f64 {
        match self.max_volume_participation {
            Some(max_pct) if max_pct > 0.0 => {
                let max_allowed = bar_volume * max_pct;
                requested_quantity.min(max_allowed)
            }
            _ => requested_quantity,
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
    /// Open tax lots per symbol (source of truth for entries).
    tax_lots: HashMap<String, Vec<TaxLot>>,
    /// All trades (including closed).
    trades: Vec<Trade>,
    /// Equity curve.
    equity_curve: Vec<EquityPoint>,
    /// Cost model for trade execution.
    cost_model: CostModel,
    /// Margin configuration for spot assets.
    margin_config: MarginConfig,
    /// Peak equity for drawdown calculation.
    peak_equity: f64,
    /// Allow short selling.
    pub allow_short: bool,
    /// Use fractional shares.
    pub fractional_shares: bool,
    /// Instrument-specific metadata.
    asset_configs: HashMap<String, AssetConfig>,
    /// Rolling volume statistics for each symbol.
    volume_profiles: HashMap<String, VolumeProfile>,
    /// Last observed prices for each symbol (used for margin checks).
    last_prices: HashMap<String, f64>,
    /// Margin set aside for futures positions.
    margin_reserve: HashMap<String, f64>,
    /// Last computed margin snapshot.
    last_margin_state: Option<MarginState>,
    /// Execution price model for market orders.
    execution_price: ExecutionPrice,
    /// Last execution price (used for fill reporting).
    last_fill_price: Option<f64>,
    /// Default tax-lot selection method when closing positions.
    lot_selection: LotSelectionMethod,
    /// Last timestamp used for equity recording (for accruals).
    last_equity_timestamp: Option<DateTime<Utc>>,
    /// Base seed for RNG (None = use timestamp-based deterministic seed).
    rng_base_seed: Option<u64>,
}

/// Result information for an executed (potentially partial) order.
#[derive(Debug, Clone)]
pub struct FillResult {
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub fill_price: f64,
    pub partial: bool,
    pub trade: Option<Trade>,
}

impl Portfolio {
    /// Create a new portfolio with initial capital.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            cash: initial_capital,
            initial_capital,
            positions: HashMap::new(),
            tax_lots: HashMap::new(),
            trades: Vec::new(),
            equity_curve: Vec::new(),
            cost_model: CostModel::default(),
            margin_config: MarginConfig::default(),
            peak_equity: initial_capital,
            allow_short: true,
            fractional_shares: true,
            asset_configs: HashMap::new(),
            volume_profiles: HashMap::new(),
            last_prices: HashMap::new(),
            margin_reserve: HashMap::new(),
            last_margin_state: None,
            execution_price: ExecutionPrice::Open,
            last_fill_price: None,
            lot_selection: LotSelectionMethod::default(),
            last_equity_timestamp: None,
            rng_base_seed: None,
        }
    }

    /// Create a portfolio with custom cost model.
    pub fn with_cost_model(initial_capital: f64, cost_model: CostModel) -> Self {
        Self {
            cash: initial_capital,
            initial_capital,
            positions: HashMap::new(),
            tax_lots: HashMap::new(),
            trades: Vec::new(),
            equity_curve: Vec::new(),
            cost_model,
            margin_config: MarginConfig::default(),
            peak_equity: initial_capital,
            allow_short: true,
            fractional_shares: true,
            asset_configs: HashMap::new(),
            volume_profiles: HashMap::new(),
            last_prices: HashMap::new(),
            margin_reserve: HashMap::new(),
            last_margin_state: None,
            execution_price: ExecutionPrice::Open,
            last_fill_price: None,
            lot_selection: LotSelectionMethod::default(),
            last_equity_timestamp: None,
            rng_base_seed: None,
        }
    }

    /// Set the execution price model.
    pub fn set_execution_price(&mut self, price: ExecutionPrice) {
        self.execution_price = price;
    }

    /// Override the default lot-selection method used for tax lots.
    pub fn set_lot_selection_method(&mut self, method: LotSelectionMethod) {
        self.lot_selection = method;
    }

    /// Read the current default lot-selection method.
    pub fn lot_selection_method(&self) -> &LotSelectionMethod {
        &self.lot_selection
    }

    /// Set the base RNG seed for reproducible random execution.
    /// If None, uses timestamp-based deterministic seeding.
    pub fn set_rng_seed(&mut self, seed: Option<u64>) {
        self.rng_base_seed = seed;
    }

    /// Get the current RNG base seed.
    pub fn rng_seed(&self) -> Option<u64> {
        self.rng_base_seed
    }

    /// Set the cost model.
    pub fn set_cost_model(&mut self, cost_model: CostModel) {
        self.cost_model = cost_model;
    }

    /// Override margin configuration.
    pub fn set_margin_config(&mut self, config: MarginConfig) {
        self.margin_config = config;
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

    /// Set or override a volume profile for a symbol.
    pub fn set_volume_profile(&mut self, symbol: impl Into<String>, profile: VolumeProfile) {
        self.volume_profiles.insert(symbol.into(), profile);
    }

    /// Bulk update volume profiles.
    pub fn set_volume_profiles(&mut self, profiles: &HashMap<String, VolumeProfile>) {
        for (symbol, profile) in profiles {
            self.volume_profiles.insert(symbol.clone(), *profile);
        }
    }

    /// Get the volume profile for a symbol if configured.
    pub fn volume_profile(&self, symbol: &str) -> Option<&VolumeProfile> {
        self.volume_profiles.get(symbol)
    }

    fn update_price_cache(&mut self, prices: &HashMap<String, f64>) {
        for (symbol, price) in prices {
            self.last_prices.insert(symbol.clone(), *price);
        }
    }

    fn cached_price_or_entry(&self, symbol: &str, entry: f64) -> f64 {
        self.last_prices.get(symbol).copied().unwrap_or(entry)
    }

    fn order_lot_method(&self, order: &Order) -> LotSelectionMethod {
        order
            .lot_selection
            .clone()
            .unwrap_or_else(|| self.lot_selection.clone())
    }

    fn fallback_lot_selection(&self) -> LotSelectionMethod {
        match &self.lot_selection {
            LotSelectionMethod::SpecificLot(_) => LotSelectionMethod::FIFO,
            other => other.clone(),
        }
    }

    fn add_tax_lot(
        &mut self,
        symbol: &str,
        side: Side,
        quantity: f64,
        price: f64,
        timestamp: DateTime<Utc>,
    ) {
        if quantity <= f64::EPSILON {
            return;
        }
        let lot = TaxLot::new(side, quantity, price, timestamp);
        self.tax_lots
            .entry(symbol.to_string())
            .or_default()
            .push(lot);
    }

    fn cleanup_tax_lots(&mut self, symbol: &str) {
        if let Some(lots) = self.tax_lots.get_mut(symbol) {
            lots.retain(|lot| !lot.is_empty());
            if lots.is_empty() {
                self.tax_lots.remove(symbol);
            }
        }
    }

    fn close_tax_lots(
        &mut self,
        symbol: &str,
        side: Side,
        quantity: f64,
        method: LotSelectionMethod,
    ) -> Result<()> {
        if quantity <= f64::EPSILON {
            return Ok(());
        }

        if matches!(method, LotSelectionMethod::SpecificLot(_)) {
            let remaining = {
                let lots = self.tax_lots.get_mut(symbol).ok_or_else(|| {
                    BacktestError::InvalidOrder(format!(
                        "No tax lots available for symbol {}",
                        symbol
                    ))
                })?;

                if let LotSelectionMethod::SpecificLot(target) = method {
                    if let Some((_, lot)) = lots
                        .iter_mut()
                        .enumerate()
                        .find(|(_, lot)| lot.id == target && lot.side == side)
                    {
                        quantity - lot.consume(quantity)
                    } else {
                        return Err(BacktestError::InvalidOrder(format!(
                            "Lot {} not found for symbol {}",
                            target, symbol
                        )));
                    }
                } else {
                    quantity
                }
            };
            self.cleanup_tax_lots(symbol);
            if remaining > f64::EPSILON {
                let fallback = self.fallback_lot_selection();
                return self.close_tax_lots(symbol, side, remaining, fallback);
            }
            return Ok(());
        }

        let mut remaining = quantity;
        {
            let lots = self.tax_lots.get_mut(symbol).ok_or_else(|| {
                BacktestError::InvalidOrder(format!("No tax lots available for symbol {}", symbol))
            })?;
            let mut indices: Vec<usize> = lots
                .iter()
                .enumerate()
                .filter(|(_, lot)| lot.side == side && lot.quantity > f64::EPSILON)
                .map(|(idx, _)| idx)
                .collect();

            if indices.is_empty() {
                return Err(BacktestError::InvalidOrder(format!(
                    "No {} tax lots available for symbol {}",
                    match side {
                        Side::Buy => "long",
                        Side::Sell => "short",
                    },
                    symbol
                )));
            }

            match method {
                LotSelectionMethod::FIFO => {
                    indices.sort_by(|a, b| lots[*a].acquired_date.cmp(&lots[*b].acquired_date));
                }
                LotSelectionMethod::LIFO => {
                    indices.sort_by(|a, b| lots[*b].acquired_date.cmp(&lots[*a].acquired_date));
                }
                LotSelectionMethod::HighestCost => {
                    indices.sort_by(|a, b| {
                        lots[*b]
                            .cost_basis
                            .partial_cmp(&lots[*a].cost_basis)
                            .unwrap_or(Ordering::Equal)
                    });
                }
                LotSelectionMethod::LowestCost => {
                    indices.sort_by(|a, b| {
                        lots[*a]
                            .cost_basis
                            .partial_cmp(&lots[*b].cost_basis)
                            .unwrap_or(Ordering::Equal)
                    });
                }
                LotSelectionMethod::SpecificLot(_) => unreachable!(),
            }

            for idx in indices {
                if remaining <= f64::EPSILON {
                    break;
                }
                let lot = &mut lots[idx];
                remaining -= lot.consume(remaining);
            }
        }

        self.cleanup_tax_lots(symbol);

        if remaining > f64::EPSILON {
            return Err(BacktestError::InvalidOrder(format!(
                "Attempted to close {:.4} more {} quantity than available for {}",
                remaining,
                match side {
                    Side::Buy => "long",
                    Side::Sell => "short",
                },
                symbol
            )));
        }

        Ok(())
    }

    fn market_execution_price(&self, bar: &Bar) -> f64 {
        match self.execution_price {
            ExecutionPrice::Open => bar.open,
            ExecutionPrice::Close => bar.close,
            ExecutionPrice::Vwap => (bar.high + bar.low + bar.close) / 3.0,
            ExecutionPrice::Twap => (bar.open + bar.close) / 2.0,
            ExecutionPrice::RandomInRange => {
                let (min_price, max_price) = if bar.high >= bar.low {
                    (bar.low, bar.high)
                } else {
                    (bar.high, bar.low)
                };
                if (max_price - min_price).abs() < f64::EPSILON {
                    min_price
                } else {
                    let seed = self.compute_rng_seed(bar.timestamp, None, self.trades.len() as u64);
                    let mut rng = StdRng::seed_from_u64(seed);
                    rng.gen_range(min_price..=max_price)
                }
            }
            ExecutionPrice::Midpoint => (bar.high + bar.low) / 2.0,
        }
    }

    fn compute_rng_seed(
        &self,
        bar_time: DateTime<Utc>,
        order_time: Option<DateTime<Utc>>,
        extra: u64,
    ) -> u64 {
        let bar_part = bar_time.timestamp_nanos_opt().unwrap_or(0) as u64;
        let order_part = order_time
            .and_then(|ts| ts.timestamp_nanos_opt())
            .unwrap_or(0) as u64;
        let base = bar_part ^ order_part ^ extra;

        // If a base seed is set, XOR it with the deterministic components
        if let Some(seed) = self.rng_base_seed {
            seed ^ base
        } else {
            base
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

    fn accrue_margin_interest(&mut self, timestamp: DateTime<Utc>) {
        if !self.margin_config.enabled || self.margin_config.interest_rate <= 0.0 {
            self.last_equity_timestamp = Some(timestamp);
            return;
        }

        if let Some(prev) = self.last_equity_timestamp {
            let seconds = (timestamp - prev).num_seconds().max(0) as f64;
            if seconds > 0.0 && self.cash < 0.0 {
                let days = seconds / 86_400.0;
                let interest = -self.cash * self.margin_config.interest_rate * (days / 365.0);
                if interest > 0.0 {
                    self.cash -= interest;
                }
            }
        }
        self.last_equity_timestamp = Some(timestamp);
    }

    fn compute_margin_state(&self, override_price: Option<(&str, f64)>) -> MarginState {
        let mut prices: HashMap<String, f64> = HashMap::new();
        let mut gross_exposure = 0.0;
        let mut reg_t_requirement = 0.0;
        let mut maintenance_requirement = 0.0;

        for (symbol, pos) in &self.positions {
            let mut price = self.cached_price_or_entry(symbol, pos.avg_entry_price);
            if let Some((sym, override_val)) = override_price {
                if sym == symbol {
                    price = override_val;
                }
            }
            prices.insert(symbol.clone(), price);

            let asset = self.asset_config_for(symbol);
            let notional = price * pos.quantity * asset.notional_multiplier();
            let exposure = notional.abs();
            gross_exposure += exposure;

            match asset.asset_class {
                AssetClass::Future {
                    margin_requirement, ..
                } => {
                    let requirement = exposure * margin_requirement.max(0.0);
                    reg_t_requirement += requirement;
                    maintenance_requirement += requirement;
                }
                _ => {
                    if pos.is_long() {
                        reg_t_requirement +=
                            exposure * self.margin_config.reg_t_long_initial.max(0.0);
                        maintenance_requirement +=
                            exposure * self.margin_config.maintenance_long_pct.max(0.0);
                    } else {
                        reg_t_requirement +=
                            exposure * self.margin_config.reg_t_short_initial.max(0.0);
                        maintenance_requirement +=
                            exposure * self.margin_config.maintenance_short_pct.max(0.0);
                    }
                }
            }
        }

        let portfolio_requirement = if self.margin_config.use_portfolio_margin {
            gross_exposure * self.margin_config.portfolio_margin_pct.max(0.0)
        } else {
            0.0
        };

        let equity = self.equity(&prices);
        let leverage = if equity.abs() > f64::EPSILON {
            gross_exposure / equity.abs()
        } else {
            f64::INFINITY
        };

        MarginState {
            equity,
            gross_exposure,
            reg_t_requirement,
            maintenance_requirement,
            portfolio_requirement,
            leverage,
        }
    }

    fn enforce_margin_limits(&mut self, override_price: Option<(&str, f64)>) -> Result<()> {
        if !self.margin_config.enabled {
            return Ok(());
        }

        let state = self.compute_margin_state(override_price);
        let mut requirement = state.reg_t_requirement;
        if self.margin_config.use_portfolio_margin {
            requirement = requirement.max(state.portfolio_requirement);
        }

        if self.margin_config.max_leverage > 0.0
            && state.leverage.is_finite()
            && state.leverage > self.margin_config.max_leverage + f64::EPSILON
        {
            self.last_margin_state = Some(state.clone());
            return Err(BacktestError::ConstraintViolation(format!(
                "Leverage {:.2}x exceeds max {:.2}x",
                state.leverage, self.margin_config.max_leverage
            )));
        }

        if state.equity < state.maintenance_requirement - f64::EPSILON {
            self.last_margin_state = Some(state.clone());
            return Err(BacktestError::MarginCall {
                equity: state.equity,
                requirement: state.maintenance_requirement,
                reason: "Maintenance requirement".to_string(),
            });
        }

        if state.equity < requirement - f64::EPSILON {
            self.last_margin_state = Some(state.clone());
            return Err(BacktestError::MarginCall {
                equity: state.equity,
                requirement,
                reason: "Initial requirement".to_string(),
            });
        }

        self.last_margin_state = Some(state);
        Ok(())
    }

    fn finalize_execution(&mut self, symbol: &str, price: f64) -> Result<()> {
        self.last_prices.insert(symbol.to_string(), price);
        self.enforce_margin_limits(Some((symbol, price)))
    }

    fn complete_spot_execution(
        &mut self,
        symbol: &str,
        exec_price: f64,
        trade: Option<Trade>,
    ) -> Result<Option<Trade>> {
        self.finalize_execution(symbol, exec_price)?;
        Ok(trade)
    }

    /// Get position for a symbol.
    pub fn position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Inspect open tax lots for a symbol.
    pub fn tax_lots(&self, symbol: &str) -> Option<&[TaxLot]> {
        self.tax_lots.get(symbol).map(|lots| lots.as_slice())
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

    /// Latest computed margin snapshot, if available.
    pub fn margin_state(&self) -> Option<&MarginState> {
        self.last_margin_state.as_ref()
    }

    /// Execute an order.
    pub fn execute_order(&mut self, order: &Order, bar: &Bar) -> Result<Option<Trade>> {
        if !order.validate() {
            return Err(BacktestError::InvalidOrder(format!(
                "Invalid order: {:?}",
                order
            )));
        }

        self.last_fill_price = None;

        let asset = self.asset_config_for(&order.symbol);
        let normalized_quantity = asset.normalize_quantity(order.quantity);
        if normalized_quantity <= 0.0 {
            return Ok(None);
        }

        // Apply volume participation limit if configured
        let quantity = self
            .cost_model
            .apply_volume_participation_limit(normalized_quantity, bar.volume);
        if quantity <= 0.0 {
            return Ok(None);
        }

        let Some(base_price) = self.order_fill_price(order, bar) else {
            self.last_fill_price = None;
            return Ok(None);
        };

        let price_with_spread = self
            .cost_model
            .apply_asset_spread(base_price, order.side, &asset);
        let mut exec_price = self
            .cost_model
            .execution_price(price_with_spread, order.side);

        if let Some(profile) = self.volume_profiles.get(&order.symbol) {
            let reference_volume = profile.reference_volume();
            if reference_volume > 0.0 {
                let impact = self.cost_model.calculate_market_impact(
                    quantity,
                    reference_volume,
                    price_with_spread,
                );
                let signed_impact = match order.side {
                    Side::Buy => impact,
                    Side::Sell => -impact,
                };
                exec_price += signed_impact;
            }
        }

        let exec_price = asset.normalize_price(exec_price);
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

    /// Execute an order applying a probabilistic partial-fill model.
    pub fn execute_with_fill_probability(
        &mut self,
        order: &Order,
        bar: &Bar,
        fill_probability: f64,
    ) -> Result<Option<FillResult>> {
        let probability = fill_probability.clamp(0.0, 1.0);
        if probability <= 0.0 {
            return Ok(None);
        }

        let fraction = if probability >= 1.0 {
            1.0
        } else {
            let seed = self.compute_rng_seed(
                bar.timestamp,
                Some(order.timestamp),
                self.trades.len() as u64,
            );
            let mut rng = StdRng::seed_from_u64(seed);
            let outcome: f64 = rng.gen();
            if outcome <= probability {
                1.0
            } else {
                let lower = (probability * 0.5).max(f64::EPSILON);
                let upper = probability.max(lower);
                if upper <= lower {
                    lower
                } else {
                    rng.gen_range(lower..=upper)
                }
            }
        };

        if fraction <= f64::EPSILON {
            return Ok(None);
        }

        let fill_quantity = (order.quantity * fraction).min(order.quantity);
        if fill_quantity <= f64::EPSILON {
            return Ok(None);
        }

        let mut adjusted_order = order.clone();
        adjusted_order.quantity = fill_quantity;
        let trade = self.execute_order(&adjusted_order, bar)?;
        let Some(fill_price) = self.last_fill_price else {
            return Ok(None);
        };

        let remaining_quantity = (order.quantity - fill_quantity).max(0.0);
        Ok(Some(FillResult {
            filled_quantity: fill_quantity,
            remaining_quantity,
            fill_price,
            partial: remaining_quantity > f64::EPSILON,
            trade,
        }))
    }

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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
        if !self.margin_config.enabled && total_cost > self.cash {
            return Err(BacktestError::InsufficientFunds {
                required: total_cost,
                available: self.cash,
            });
        }
        self.cash -= total_cost;
        self.last_fill_price = Some(exec_price);

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
        let lot_method = self.order_lot_method(order);

        if let Some(mut pos) = existing_position {
            if pos.is_short() {
                if quantity >= pos.quantity {
                    self.close_tax_lots(
                        &order.symbol,
                        Side::Sell,
                        pos.quantity,
                        lot_method.clone(),
                    )?;
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
                        return self.complete_spot_execution(&order.symbol, exec_price, None);
                    }
                } else {
                    self.close_tax_lots(&order.symbol, Side::Sell, quantity, lot_method)?;
                    pos.quantity -= quantity;
                    self.positions.insert(order.symbol.clone(), pos);
                    return self.complete_spot_execution(&order.symbol, exec_price, None);
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
            self.add_tax_lot(
                &order.symbol,
                Side::Buy,
                opening_qty,
                exec_price,
                bar.timestamp,
            );
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
            self.complete_spot_execution(&order.symbol, exec_price, Some(trade))
        } else {
            self.complete_spot_execution(&order.symbol, exec_price, None)
        }
    }

    #[allow(clippy::too_many_arguments)]
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
        self.last_fill_price = Some(exec_price);

        let per_unit_commission = if quantity > 0.0 {
            commission / quantity
        } else {
            0.0
        };
        let closing_commission = per_unit_commission * closing_qty;
        let opening_commission = (commission - closing_commission).max(0.0);
        let lot_method = self.order_lot_method(order);

        if let Some(mut pos) = existing_position {
            if pos.is_long() {
                if quantity >= pos.quantity {
                    self.close_tax_lots(
                        &order.symbol,
                        Side::Buy,
                        pos.quantity,
                        lot_method.clone(),
                    )?;
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
                        return self.complete_spot_execution(&order.symbol, exec_price, None);
                    }
                } else {
                    self.close_tax_lots(&order.symbol, Side::Buy, quantity, lot_method)?;
                    pos.quantity -= quantity;
                    self.positions.insert(order.symbol.clone(), pos);
                    return self.complete_spot_execution(&order.symbol, exec_price, None);
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
            self.add_tax_lot(
                &order.symbol,
                Side::Sell,
                opening_qty,
                exec_price,
                bar.timestamp,
            );
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
            self.complete_spot_execution(&order.symbol, exec_price, Some(trade))
        } else {
            self.complete_spot_execution(&order.symbol, exec_price, None)
        }
    }

    #[allow(clippy::too_many_arguments)]
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
        self.last_fill_price = Some(exec_price);

        let multiplier = asset.notional_multiplier();
        let mut qty_remaining = quantity;
        let per_unit_commission = if quantity > 0.0 {
            commission / quantity
        } else {
            0.0
        };
        let lot_method = self.order_lot_method(order);

        if let Some(existing) = self.positions.get(&order.symbol).cloned() {
            match (order.side, existing.side) {
                (Side::Buy, Side::Sell) => {
                    let close_qty = qty_remaining.min(existing.quantity);
                    if close_qty > 0.0 {
                        self.close_tax_lots(
                            &order.symbol,
                            Side::Sell,
                            close_qty,
                            lot_method.clone(),
                        )?;
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
                        self.close_tax_lots(
                            &order.symbol,
                            Side::Buy,
                            close_qty,
                            lot_method.clone(),
                        )?;
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
            self.add_tax_lot(
                &order.symbol,
                order.side,
                qty_remaining,
                exec_price,
                bar.timestamp,
            );
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

        self.finalize_execution(&order.symbol, exec_price)?;
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
            OrderType::Market => Some(self.market_execution_price(bar)),
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
    /// Record an equity point and update margin state.
    pub fn record_equity(
        &mut self,
        timestamp: DateTime<Utc>,
        prices: &HashMap<String, f64>,
    ) -> Result<()> {
        self.accrue_margin_interest(timestamp);
        self.update_price_cache(prices);
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
        self.enforce_margin_limits(None)?;
        Ok(())
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
    use crate::types::{AssetClass, AssetConfig, Bar, VolumeProfile};
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

    fn sample_volume_profile() -> VolumeProfile {
        VolumeProfile::new(1_000.0, 100.0)
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
    fn test_execution_price_close_model() {
        let mut portfolio = Portfolio::with_cost_model(100000.0, CostModel::zero());
        portfolio.set_execution_price(ExecutionPrice::Close);
        let bar = sample_bar();
        let order = Order::market("AAPL", Side::Buy, 10.0, bar.timestamp);

        let trade = portfolio.execute_order(&order, &bar).unwrap().unwrap();
        assert!((trade.entry_price - bar.close).abs() < 1e-6);
    }

    #[test]
    fn test_fill_probability_zero_no_fill() {
        let mut portfolio = Portfolio::with_cost_model(100000.0, CostModel::zero());
        let bar = sample_bar();
        let order = Order::market("AAPL", Side::Buy, 10.0, bar.timestamp);

        let result = portfolio
            .execute_with_fill_probability(&order, &bar, 0.0)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_fill_probability_partial_fill_occurs() {
        let base_ts = Utc.with_ymd_and_hms(2024, 1, 15, 9, 30, 0).unwrap();
        let mut partial_seen = false;

        for offset in 0..50 {
            let mut portfolio = Portfolio::with_cost_model(100000.0, CostModel::zero());
            let order_ts = base_ts + chrono::Duration::seconds(offset);
            let bar_ts = order_ts + chrono::Duration::minutes(1);
            let bar = Bar::new(bar_ts, 100.0, 105.0, 98.0, 101.0, 1000.0);
            let order = Order::market("AAPL", Side::Buy, 10.0, order_ts);

            if let Some(fill) = portfolio
                .execute_with_fill_probability(&order, &bar, 0.5)
                .unwrap()
            {
                if fill.partial {
                    assert!(fill.filled_quantity < 10.0);
                    assert!(fill.remaining_quantity > 0.0);
                    partial_seen = true;
                    break;
                }
            }
        }

        assert!(
            partial_seen,
            "expected at least one partial fill within the sample window"
        );
    }

    #[test]
    fn test_market_impact_buy_adjusts_execution_price() {
        let mut cost_model = CostModel::zero();
        cost_model.market_impact = MarketImpactModel::Linear { coefficient: 0.01 };
        let mut portfolio = Portfolio::with_cost_model(100000.0, cost_model);
        portfolio.set_volume_profile("AAPL", sample_volume_profile());

        let bar = sample_bar();
        // 50 / 100 = 0.5, impact = 100 * 0.01 * 0.5 = 0.5
        let order = Order::market("AAPL", Side::Buy, 50.0, bar.timestamp);
        let trade = portfolio.execute_order(&order, &bar).unwrap().unwrap();
        assert!((trade.entry_price - 100.5).abs() < 1e-6);
    }

    #[test]
    fn test_market_impact_sell_direction() {
        let mut cost_model = CostModel::zero();
        cost_model.market_impact = MarketImpactModel::Linear { coefficient: 0.01 };
        let mut portfolio = Portfolio::with_cost_model(100000.0, cost_model);
        portfolio.set_volume_profile("AAPL", sample_volume_profile());

        let bar = sample_bar();
        let order = Order::market("AAPL", Side::Sell, 50.0, bar.timestamp);
        let trade = portfolio.execute_order(&order, &bar).unwrap().unwrap();
        assert!((trade.entry_price - 99.5).abs() < 1e-6);
    }

    #[test]
    fn test_tax_lot_fifo_consumption() {
        let mut portfolio = Portfolio::with_cost_model(100000.0, CostModel::zero());
        let bar1 = sample_bar();
        let bar2 = Bar::new(
            bar1.timestamp + chrono::Duration::days(1),
            110.0,
            112.0,
            108.0,
            111.0,
            900.0,
        );

        portfolio
            .execute_order(
                &Order::market("AAPL", Side::Buy, 50.0, bar1.timestamp),
                &bar1,
            )
            .unwrap();
        portfolio
            .execute_order(
                &Order::market("AAPL", Side::Buy, 30.0, bar2.timestamp),
                &bar2,
            )
            .unwrap();

        let sell_bar = Bar::new(
            bar2.timestamp + chrono::Duration::days(1),
            120.0,
            121.0,
            118.0,
            119.0,
            1000.0,
        );
        portfolio
            .execute_order(
                &Order::market("AAPL", Side::Sell, 40.0, sell_bar.timestamp),
                &sell_bar,
            )
            .unwrap();

        let lots = portfolio.tax_lots("AAPL").unwrap();
        let first_lot = lots
            .iter()
            .find(|lot| lot.acquired_date == bar1.timestamp)
            .unwrap();
        let second_lot = lots
            .iter()
            .find(|lot| lot.acquired_date == bar2.timestamp)
            .unwrap();

        assert!((first_lot.quantity - 10.0).abs() < 1e-6);
        assert!((second_lot.quantity - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_tax_lot_specific_selection() {
        let mut portfolio = Portfolio::with_cost_model(100000.0, CostModel::zero());
        let bar1 = sample_bar();
        let bar2 = Bar::new(
            bar1.timestamp + chrono::Duration::days(1),
            110.0,
            112.0,
            108.0,
            111.0,
            900.0,
        );

        portfolio
            .execute_order(
                &Order::market("AAPL", Side::Buy, 40.0, bar1.timestamp),
                &bar1,
            )
            .unwrap();
        portfolio
            .execute_order(
                &Order::market("AAPL", Side::Buy, 20.0, bar2.timestamp),
                &bar2,
            )
            .unwrap();

        let lot_id = portfolio.tax_lots("AAPL").unwrap()[1].id;
        let sell_bar = Bar::new(
            bar2.timestamp + chrono::Duration::days(1),
            115.0,
            116.0,
            112.0,
            114.0,
            1000.0,
        );
        let specific_order = Order::market("AAPL", Side::Sell, 15.0, sell_bar.timestamp)
            .with_lot_selection(LotSelectionMethod::SpecificLot(lot_id));
        portfolio.execute_order(&specific_order, &sell_bar).unwrap();

        let lots = portfolio.tax_lots("AAPL").unwrap();
        // Second lot should be reduced first due to explicit selection
        let selected_lot = lots.iter().find(|lot| lot.id == lot_id).unwrap();
        assert!((selected_lot.quantity - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_tax_lot_lifo_policy() {
        let mut portfolio = Portfolio::with_cost_model(100000.0, CostModel::zero());
        portfolio.set_lot_selection_method(LotSelectionMethod::LIFO);
        let bar1 = sample_bar();
        let bar2 = Bar::new(
            bar1.timestamp + chrono::Duration::days(1),
            108.0,
            110.0,
            106.0,
            107.0,
            900.0,
        );

        portfolio
            .execute_order(
                &Order::market("AAPL", Side::Buy, 25.0, bar1.timestamp),
                &bar1,
            )
            .unwrap();
        portfolio
            .execute_order(
                &Order::market("AAPL", Side::Buy, 25.0, bar2.timestamp),
                &bar2,
            )
            .unwrap();

        let sell_bar = Bar::new(
            bar2.timestamp + chrono::Duration::days(1),
            109.0,
            110.0,
            105.0,
            106.0,
            1000.0,
        );
        portfolio
            .execute_order(
                &Order::market("AAPL", Side::Sell, 15.0, sell_bar.timestamp),
                &sell_bar,
            )
            .unwrap();

        let lots = portfolio.tax_lots("AAPL").unwrap();
        let newer_lot = lots
            .iter()
            .find(|lot| lot.acquired_date == bar2.timestamp)
            .unwrap();
        let older_lot = lots
            .iter()
            .find(|lot| lot.acquired_date == bar1.timestamp)
            .unwrap();

        assert!((newer_lot.quantity - 10.0).abs() < 1e-6);
        assert!((older_lot.quantity - 25.0).abs() < 1e-6);
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
        portfolio.set_margin_config(MarginConfig {
            enabled: false,
            ..MarginConfig::default()
        });
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
        portfolio.record_equity(timestamp, &prices).unwrap();

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
    fn test_margin_supports_leverage() {
        let mut portfolio = Portfolio::with_cost_model(100_000.0, CostModel::zero());
        let bar = sample_bar();
        // Buy 1,500 shares at ~$100 => $150k notional with only $100k cash.
        let order = Order::market("AAPL", Side::Buy, 1500.0, bar.timestamp);
        portfolio.execute_order(&order, &bar).unwrap();

        assert!(portfolio.cash < 0.0, "Margin trade should create borrowing");
        let state = portfolio.margin_state().expect("margin state missing");
        assert!(state.gross_exposure > portfolio.initial_capital);
        assert!(state.reg_t_requirement < state.equity + 1.0);
    }

    #[test]
    fn test_margin_call_error() {
        let mut portfolio = Portfolio::with_cost_model(100_000.0, CostModel::zero());
        portfolio.set_margin_config(MarginConfig {
            max_leverage: 10.0,
            ..MarginConfig::default()
        });
        let bar = sample_bar();
        // Buy 1,500 shares (~$150k notional) and allow leverage headroom so margin call fires on price drop.
        let order = Order::market("AAPL", Side::Buy, 1500.0, bar.timestamp);
        portfolio.execute_order(&order, &bar).unwrap();

        // A sharp price drop should trigger a maintenance margin call when equity erodes.
        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 40.0);
        let timestamp = bar.timestamp + chrono::Duration::days(1);
        let result = portfolio.record_equity(timestamp, &prices);
        assert!(matches!(result, Err(BacktestError::MarginCall { .. })));
    }

    #[test]
    fn test_leverage_limit_enforced() {
        let mut portfolio = Portfolio::with_cost_model(100_000.0, CostModel::zero());
        portfolio.set_margin_config(MarginConfig {
            max_leverage: 1.0,
            ..MarginConfig::default()
        });
        let bar = sample_bar();
        let order = Order::market("AAPL", Side::Buy, 1500.0, bar.timestamp);
        let result = portfolio.execute_order(&order, &bar);
        assert!(matches!(
            result,
            Err(BacktestError::ConstraintViolation(msg)) if msg.contains("Leverage")
        ));
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

    #[test]
    fn test_volume_participation_limit_no_limit() {
        // Test that without a limit, full order quantity is allowed
        let mut cost_model = CostModel::zero();
        cost_model.max_volume_participation = None;
        let mut portfolio = Portfolio::with_cost_model(1000000.0, cost_model); // $1M capital
        // Disable margin to avoid leverage violations
        portfolio.set_margin_config(MarginConfig {
            enabled: false,
            ..MarginConfig::default()
        });

        let bar = Bar::new(
            chrono::Utc::now(),
            100.0,
            105.0,
            99.0,
            102.0,
            10000.0, // Bar volume
        );

        // Order for 5000 shares (50% of bar volume) - should fill completely
        let order = Order::market("AAPL", Side::Buy, 5000.0, bar.timestamp);
        let trade = portfolio.execute_order(&order, &bar).unwrap();

        assert!(trade.is_some());
        assert_eq!(portfolio.position("AAPL").unwrap().quantity, 5000.0);
    }

    #[test]
    fn test_volume_participation_limit_enforced() {
        // Test that with a 10% limit, orders are capped at 10% of bar volume
        let mut cost_model = CostModel::zero();
        cost_model.max_volume_participation = Some(0.10); // 10% limit
        let mut portfolio = Portfolio::with_cost_model(100000.0, cost_model);

        let bar = Bar::new(
            chrono::Utc::now(),
            100.0,
            105.0,
            99.0,
            102.0,
            10000.0, // Bar volume
        );

        // Order for 5000 shares (50% of bar volume) - should only fill 1000 (10%)
        let order = Order::market("AAPL", Side::Buy, 5000.0, bar.timestamp);
        let trade = portfolio.execute_order(&order, &bar).unwrap();

        assert!(trade.is_some());
        // Should only fill 10% of bar volume = 1000 shares
        assert_eq!(portfolio.position("AAPL").unwrap().quantity, 1000.0);
    }

    #[test]
    fn test_volume_participation_limit_below_limit() {
        // Test that orders below the limit fill completely
        let mut cost_model = CostModel::zero();
        cost_model.max_volume_participation = Some(0.10); // 10% limit
        let mut portfolio = Portfolio::with_cost_model(100000.0, cost_model);

        let bar = Bar::new(
            chrono::Utc::now(),
            100.0,
            105.0,
            99.0,
            102.0,
            10000.0, // Bar volume
        );

        // Order for 500 shares (5% of bar volume) - should fill completely
        let order = Order::market("AAPL", Side::Buy, 500.0, bar.timestamp);
        let trade = portfolio.execute_order(&order, &bar).unwrap();

        assert!(trade.is_some());
        assert_eq!(portfolio.position("AAPL").unwrap().quantity, 500.0);
    }

    #[test]
    fn test_volume_participation_limit_zero_volume() {
        // Test that zero volume bars reject orders with participation limits
        let mut cost_model = CostModel::zero();
        cost_model.max_volume_participation = Some(0.10); // 10% limit
        let mut portfolio = Portfolio::with_cost_model(100000.0, cost_model);

        let bar = Bar::new(
            chrono::Utc::now(),
            100.0,
            105.0,
            99.0,
            102.0,
            0.0, // Zero volume
        );

        // Order should not fill since bar volume is zero and limit applies
        let order = Order::market("AAPL", Side::Buy, 100.0, bar.timestamp);
        let trade = portfolio.execute_order(&order, &bar).unwrap();

        assert!(trade.is_none());
        assert!(!portfolio.has_position("AAPL"));
    }

    #[test]
    fn test_volume_participation_limit_sell_orders() {
        // Test that volume participation limits apply to sell orders too
        let mut cost_model = CostModel::zero();
        cost_model.max_volume_participation = Some(0.20); // 20% limit from the start
        let mut portfolio = Portfolio::with_cost_model(1000000.0, cost_model); // $1M capital
        // Disable margin to avoid leverage violations
        portfolio.set_margin_config(MarginConfig {
            enabled: false,
            ..MarginConfig::default()
        });

        // First, buy shares (will be limited by volume participation)
        let buy_bar = Bar::new(
            chrono::Utc::now(),
            100.0,
            105.0,
            99.0,
            102.0,
            25000.0, // Large volume to allow the buy
        );
        // Buy 5000 shares - with 20% limit on 25000 volume = 5000 allowed
        let buy_order = Order::market("AAPL", Side::Buy, 5000.0, buy_bar.timestamp);
        portfolio.execute_order(&buy_order, &buy_bar).unwrap();
        assert_eq!(portfolio.position("AAPL").unwrap().quantity, 5000.0);

        // Now try to sell all shares on a bar with lower volume
        let sell_bar = Bar::new(
            chrono::Utc::now() + chrono::Duration::days(1),
            102.0,
            106.0,
            101.0,
            104.0,
            10000.0, // Bar volume
        );

        // Try to sell all 5000 shares (50% of volume) - should only sell 2000 (20%)
        let sell_order = Order::market("AAPL", Side::Sell, 5000.0, sell_bar.timestamp);
        portfolio.execute_order(&sell_order, &sell_bar).unwrap();

        // Should still hold 3000 shares after selling 2000
        assert_eq!(portfolio.position("AAPL").unwrap().quantity, 3000.0);
    }

    #[test]
    fn test_volume_participation_limit_fractional() {
        // Test that fractional participation rates work correctly
        let mut cost_model = CostModel::zero();
        cost_model.max_volume_participation = Some(0.05); // 5% limit
        let mut portfolio = Portfolio::with_cost_model(100000.0, cost_model);

        let bar = Bar::new(
            chrono::Utc::now(),
            50.0,
            52.0,
            49.0,
            51.0,
            1000.0, // Bar volume
        );

        // Order for 200 shares (20% of bar volume) - should only fill 50 (5%)
        let order = Order::market("AAPL", Side::Buy, 200.0, bar.timestamp);
        let trade = portfolio.execute_order(&order, &bar).unwrap();

        assert!(trade.is_some());
        assert_eq!(portfolio.position("AAPL").unwrap().quantity, 50.0);
    }
}

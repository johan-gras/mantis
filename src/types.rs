//! Core data types for the backtest engine.

use chrono::{DateTime, Datelike, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};
use uuid::Uuid;

/// OHLCV bar representing a single time period of market data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Rolling volume statistics used for execution modeling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct VolumeProfile {
    pub avg_daily_volume: f64,
    pub avg_bar_volume: f64,
}

impl VolumeProfile {
    /// Create a new volume profile from explicit averages.
    pub fn new(avg_daily_volume: f64, avg_bar_volume: f64) -> Self {
        Self {
            avg_daily_volume,
            avg_bar_volume,
        }
    }

    /// Build a volume profile from historical bars.
    pub fn from_bars(bars: &[Bar]) -> Option<Self> {
        if bars.is_empty() {
            return None;
        }

        let total_volume: f64 = bars.iter().map(|b| b.volume).sum();
        let avg_bar_volume = total_volume / bars.len() as f64;

        let mut daily_totals: HashMap<NaiveDate, f64> = HashMap::new();
        for bar in bars {
            let date = bar.timestamp.date_naive();
            *daily_totals.entry(date).or_default() += bar.volume;
        }

        let avg_daily_volume = if daily_totals.is_empty() {
            0.0
        } else {
            daily_totals.values().sum::<f64>() / daily_totals.len() as f64
        };

        Some(Self::new(avg_daily_volume, avg_bar_volume))
    }

    /// Return the best-effort reference volume for impact calculations.
    pub fn reference_volume(&self) -> f64 {
        if self.avg_bar_volume > 0.0 {
            self.avg_bar_volume
        } else {
            self.avg_daily_volume
        }
    }
}

/// Execution price model for market orders.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionPrice {
    /// Fill at the opening price of the bar.
    #[default]
    Open,
    /// Fill at the closing price of the bar.
    Close,
    /// Fill using the bar's typical price as a VWAP approximation.
    Vwap,
    /// Fill at the average of open and close (TWAP approximation).
    Twap,
    /// Fill at a random price within the bar's range.
    RandomInRange,
    /// Fill at the midpoint between the bar's high and low.
    Midpoint,
}

/// Supported asset classes in the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum AssetClass {
    /// Traditional equities where notional = price * quantity.
    #[default]
    Equity,
    /// Exchange-traded futures contracts with multipliers and margin.
    Future {
        multiplier: f64,
        tick_size: f64,
        margin_requirement: f64,
    },
    /// Crypto assets with explicit decimal precision.
    Crypto {
        base_precision: u8,
        quote_precision: u8,
    },
    /// FX pairs quoted in lots and pip sizes.
    Forex { pip_size: f64, lot_size: f64 },
    /// Listed options referencing an underlying symbol.
    Option { underlying: String, multiplier: f64 },
}

impl AssetClass {
    /// Determine the price tick/pip constraints for this asset class.
    fn price_increment(&self) -> Option<f64> {
        match self {
            AssetClass::Future { tick_size, .. } => Some(*tick_size),
            AssetClass::Forex { pip_size, .. } => Some(*pip_size),
            _ => None,
        }
    }
}

/// Configuration describing how a symbol should be treated by the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssetConfig {
    pub symbol: String,
    pub asset_class: AssetClass,
    pub currency: String,
    pub exchange: Option<String>,
}

impl AssetConfig {
    /// Create a new configuration for a symbol.
    pub fn new(symbol: impl Into<String>, asset_class: AssetClass) -> Self {
        Self {
            symbol: symbol.into(),
            asset_class,
            currency: "USD".to_string(),
            exchange: None,
        }
    }

    /// Convenience constructor for equity symbols.
    pub fn equity(symbol: impl Into<String>) -> Self {
        Self::new(symbol, AssetClass::Equity)
    }

    /// Create a futures configuration with sensible defaults.
    pub fn future(symbol: impl Into<String>, multiplier: f64, tick_size: f64, margin: f64) -> Self {
        Self::new(
            symbol,
            AssetClass::Future {
                multiplier,
                tick_size,
                margin_requirement: margin,
            },
        )
    }

    /// Normalize a raw quantity to match asset precision or lot size rules.
    pub fn normalize_quantity(&self, quantity: f64) -> f64 {
        match &self.asset_class {
            AssetClass::Crypto { base_precision, .. } => {
                let factor = 10_f64.powi(*base_precision as i32);
                (quantity * factor).round() / factor
            }
            _ => quantity,
        }
    }

    /// Normalize price to valid tick/pip increments when defined.
    pub fn normalize_price(&self, price: f64) -> f64 {
        match self.asset_class.price_increment() {
            Some(increment) if increment > 0.0 => (price / increment).round() * increment,
            _ => price,
        }
    }

    /// Determine the notional multiplier for this asset class.
    pub fn notional_multiplier(&self) -> f64 {
        match &self.asset_class {
            AssetClass::Future { multiplier, .. } => *multiplier,
            AssetClass::Option { multiplier, .. } => *multiplier,
            _ => 1.0,
        }
    }

    /// Margin requirement as a percentage of notional, if any.
    pub fn margin_requirement(&self) -> Option<f64> {
        match &self.asset_class {
            AssetClass::Future {
                margin_requirement, ..
            } => Some(*margin_requirement),
            _ => None,
        }
    }
}

/// Data frequency for proper annualization of metrics.
///
/// Different bar frequencies require different annualization factors:
/// - Daily data: 252 trading days per year (stocks) or 365 days (24/7 markets)
/// - Intraday data: varies based on trading hours and bar frequency
///
/// The `trading_hours_24` flag indicates 24/7 markets (crypto) vs traditional markets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum DataFrequency {
    /// 1-second bars
    Second1,
    /// 5-second bars
    Second5,
    /// 10-second bars
    Second10,
    /// 15-second bars
    Second15,
    /// 30-second bars
    Second30,
    /// 1-minute bars
    Minute1,
    /// 5-minute bars
    Minute5,
    /// 15-minute bars
    Minute15,
    /// 30-minute bars
    Minute30,
    /// 1-hour bars
    Hour1,
    /// 4-hour bars
    Hour4,
    /// Daily bars (default)
    #[default]
    Day,
    /// Weekly bars
    Week,
    /// Monthly bars
    Month,
}

impl DataFrequency {
    /// Get the interval in seconds for this frequency.
    pub fn to_seconds(&self) -> i64 {
        match self {
            DataFrequency::Second1 => 1,
            DataFrequency::Second5 => 5,
            DataFrequency::Second10 => 10,
            DataFrequency::Second15 => 15,
            DataFrequency::Second30 => 30,
            DataFrequency::Minute1 => 60,
            DataFrequency::Minute5 => 300,
            DataFrequency::Minute15 => 900,
            DataFrequency::Minute30 => 1800,
            DataFrequency::Hour1 => 3600,
            DataFrequency::Hour4 => 14400,
            DataFrequency::Day => 86400,
            DataFrequency::Week => 604800,
            DataFrequency::Month => 2592000, // 30 days approximation
        }
    }

    /// Get the annualization factor for this frequency.
    ///
    /// The annualization factor is the number of periods per year, which is used
    /// to annualize returns, volatility, and ratios like Sharpe/Sortino.
    ///
    /// For traditional markets (stocks, futures, forex):
    /// - 252 trading days per year
    /// - 6.5 trading hours per day
    /// - Excludes weekends and holidays
    ///
    /// For 24/7 markets (crypto):
    /// - 365 days per year
    /// - 24 hours per day
    /// - Includes all days
    pub fn annualization_factor(&self, trading_hours_24: bool) -> f64 {
        if trading_hours_24 {
            // 24/7 markets (crypto)
            match self {
                DataFrequency::Second1 => 31_536_000.0, // 365 * 24 * 60 * 60
                DataFrequency::Second5 => 6_307_200.0,  // 31_536_000 / 5
                DataFrequency::Second10 => 3_153_600.0, // 31_536_000 / 10
                DataFrequency::Second15 => 2_102_400.0, // 31_536_000 / 15
                DataFrequency::Second30 => 1_051_200.0, // 31_536_000 / 30
                DataFrequency::Minute1 => 525_600.0,    // 365 * 24 * 60
                DataFrequency::Minute5 => 105_120.0,    // 525_600 / 5
                DataFrequency::Minute15 => 35_040.0,    // 525_600 / 15
                DataFrequency::Minute30 => 17_520.0,    // 525_600 / 30
                DataFrequency::Hour1 => 8_760.0,        // 365 * 24
                DataFrequency::Hour4 => 2_190.0,        // 8_760 / 4
                DataFrequency::Day => 365.0,
                DataFrequency::Week => 52.0,
                DataFrequency::Month => 12.0,
            }
        } else {
            // Traditional markets (stocks, futures, forex)
            // 252 trading days, 6.5 hours per trading day
            match self {
                DataFrequency::Second1 => 5_896_800.0, // 252 * 6.5 * 60 * 60
                DataFrequency::Second5 => 1_179_360.0, // 5_896_800 / 5
                DataFrequency::Second10 => 589_680.0,  // 5_896_800 / 10
                DataFrequency::Second15 => 393_120.0,  // 5_896_800 / 15
                DataFrequency::Second30 => 196_560.0,  // 5_896_800 / 30
                DataFrequency::Minute1 => 98_280.0,    // 252 * 6.5 * 60
                DataFrequency::Minute5 => 19_656.0,    // 98_280 / 5
                DataFrequency::Minute15 => 6_552.0,    // 98_280 / 15
                DataFrequency::Minute30 => 3_276.0,    // 98_280 / 30
                DataFrequency::Hour1 => 1_638.0,       // 252 * 6.5
                DataFrequency::Hour4 => 409.5,         // 1_638 / 4
                DataFrequency::Day => 252.0,
                DataFrequency::Week => 52.0,
                DataFrequency::Month => 12.0,
            }
        }
    }

    /// Detect data frequency from a series of bars by analyzing timestamp gaps.
    ///
    /// Returns the most common frequency detected from the bar timestamps.
    /// If bars contain fewer than 2 bars, returns Day as default.
    pub fn detect(bars: &[Bar]) -> Self {
        if bars.len() < 2 {
            return DataFrequency::Day;
        }

        // Calculate time differences between consecutive bars
        let mut gaps: Vec<i64> = bars
            .windows(2)
            .map(|w| (w[1].timestamp - w[0].timestamp).num_seconds())
            .filter(|&g| g > 0) // Filter out zero or negative gaps
            .collect();

        if gaps.is_empty() {
            return DataFrequency::Day;
        }

        // Find median gap (more robust than mean for irregular data)
        gaps.sort_unstable();
        let median_gap = gaps[gaps.len() / 2];

        // Match to closest standard frequency
        DataFrequency::from_seconds(median_gap)
    }

    /// Convert seconds to the closest DataFrequency variant.
    fn from_seconds(seconds: i64) -> Self {
        // Define frequency boundaries (midpoints between adjacent frequencies)
        // Order matters: check from smallest to largest
        if seconds <= 3 {
            DataFrequency::Second1
        } else if seconds <= 7 {
            DataFrequency::Second5
        } else if seconds <= 12 {
            DataFrequency::Second10
        } else if seconds <= 22 {
            DataFrequency::Second15
        } else if seconds <= 45 {
            DataFrequency::Second30
        } else if seconds <= 150 {
            DataFrequency::Minute1
        } else if seconds <= 450 {
            DataFrequency::Minute5
        } else if seconds <= 1350 {
            DataFrequency::Minute15
        } else if seconds <= 2700 {
            DataFrequency::Minute30
        } else if seconds <= 9000 {
            DataFrequency::Hour1
        } else if seconds <= 43200 {
            DataFrequency::Hour4
        } else if seconds <= 259200 {
            DataFrequency::Day
        } else if seconds <= 1209600 {
            DataFrequency::Week
        } else {
            DataFrequency::Month
        }
    }

    /// Check if this frequency requires 24/7 trading hours based on typical use.
    /// Note: This is a heuristic. For accurate results, use explicit configuration.
    pub fn is_likely_crypto(bars: &[Bar]) -> bool {
        if bars.len() < 10 {
            return false;
        }

        // Check if bars exist on weekends (Saturday=6, Sunday=0 in chrono)
        let weekend_bars = bars
            .iter()
            .filter(|b| {
                let weekday = b.timestamp.weekday().num_days_from_sunday();
                weekday == 0 || weekday == 6
            })
            .count();

        // If more than 10% of bars are on weekends, likely 24/7 market
        weekend_bars as f64 / bars.len() as f64 > 0.10
    }

    /// Get a human-readable description of this frequency.
    pub fn description(&self) -> &'static str {
        match self {
            DataFrequency::Second1 => "1-second",
            DataFrequency::Second5 => "5-second",
            DataFrequency::Second10 => "10-second",
            DataFrequency::Second15 => "15-second",
            DataFrequency::Second30 => "30-second",
            DataFrequency::Minute1 => "1-minute",
            DataFrequency::Minute5 => "5-minute",
            DataFrequency::Minute15 => "15-minute",
            DataFrequency::Minute30 => "30-minute",
            DataFrequency::Hour1 => "hourly",
            DataFrequency::Hour4 => "4-hour",
            DataFrequency::Day => "daily",
            DataFrequency::Week => "weekly",
            DataFrequency::Month => "monthly",
        }
    }
}

impl fmt::Display for DataFrequency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

impl Bar {
    /// Create a new bar with validation.
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Validate that bar data is consistent.
    pub fn validate(&self) -> bool {
        self.high >= self.low
            && self.high >= self.open
            && self.high >= self.close
            && self.low <= self.open
            && self.low <= self.close
            && self.open > 0.0
            && self.close > 0.0
            && self.volume >= 0.0
    }

    /// Calculate the typical price (HLC average).
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the bar range.
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the body size (absolute difference between open and close).
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if the bar is bullish (close > open).
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Order side (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

/// Strategy for choosing which tax lot is closed when offsetting a position.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum LotSelectionMethod {
    #[serde(rename = "fifo")]
    #[default]
    FIFO,
    #[serde(rename = "lifo")]
    LIFO,
    #[serde(rename = "highest-cost")]
    HighestCost,
    #[serde(rename = "lowest-cost")]
    LowestCost,
    #[serde(rename = "specific-lot")]
    SpecificLot(Uuid),
}

/// Order type for execution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderType {
    /// Execute at current market price.
    Market,
    /// Execute at specified limit price or better.
    Limit(LimitPrice),
    /// Stop order - triggers when price crosses stop level.
    Stop(StopPrice),
    /// Stop-limit order - becomes limit order when stop is triggered.
    StopLimit { stop: StopPrice, limit: LimitPrice },
}

/// Wrapper for limit price to ensure type safety.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LimitPrice(pub f64);

/// Wrapper for stop price to ensure type safety.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StopPrice(pub f64);

/// An order to be submitted to the backtest engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub symbol: String,
    pub side: Side,
    pub quantity: f64,
    pub order_type: OrderType,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lot_selection: Option<LotSelectionMethod>,
}

impl Order {
    /// Create a new market order.
    pub fn market(
        symbol: impl Into<String>,
        side: Side,
        quantity: f64,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            quantity,
            order_type: OrderType::Market,
            timestamp,
            lot_selection: None,
        }
    }

    /// Create a new limit order.
    pub fn limit(
        symbol: impl Into<String>,
        side: Side,
        quantity: f64,
        limit_price: f64,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            quantity,
            order_type: OrderType::Limit(LimitPrice(limit_price)),
            timestamp,
            lot_selection: None,
        }
    }

    /// Create a new stop order.
    pub fn stop(
        symbol: impl Into<String>,
        side: Side,
        quantity: f64,
        stop_price: f64,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            quantity,
            order_type: OrderType::Stop(StopPrice(stop_price)),
            timestamp,
            lot_selection: None,
        }
    }

    /// Override the lot selection strategy for this order.
    pub fn with_lot_selection(mut self, method: LotSelectionMethod) -> Self {
        self.lot_selection = Some(method);
        self
    }

    /// Validate the order.
    pub fn validate(&self) -> bool {
        self.quantity > 0.0
            && !self.symbol.is_empty()
            && match self.order_type {
                OrderType::Limit(LimitPrice(p)) => p > 0.0,
                OrderType::Stop(StopPrice(p)) => p > 0.0,
                OrderType::StopLimit {
                    stop: StopPrice(s),
                    limit: LimitPrice(l),
                } => s > 0.0 && l > 0.0,
                OrderType::Market => true,
            }
    }
}

/// A completed trade (filled order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub side: Side,
    pub quantity: f64,
    pub entry_price: f64,
    pub entry_time: DateTime<Utc>,
    pub exit_price: Option<f64>,
    pub exit_time: Option<DateTime<Utc>>,
    pub commission: f64,
    pub slippage: f64,
}

impl Trade {
    /// Create a new open trade.
    pub fn open(
        symbol: impl Into<String>,
        side: Side,
        quantity: f64,
        entry_price: f64,
        entry_time: DateTime<Utc>,
        commission: f64,
        slippage: f64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            quantity,
            entry_price,
            entry_time,
            exit_price: None,
            exit_time: None,
            commission,
            slippage,
        }
    }

    /// Close the trade.
    pub fn close(&mut self, exit_price: f64, exit_time: DateTime<Utc>, exit_commission: f64) {
        self.exit_price = Some(exit_price);
        self.exit_time = Some(exit_time);
        self.commission += exit_commission;
    }

    /// Check if trade is closed.
    pub fn is_closed(&self) -> bool {
        self.exit_price.is_some()
    }

    /// Calculate the gross P&L (before costs).
    pub fn gross_pnl(&self) -> Option<f64> {
        self.exit_price.map(|exit| {
            let direction = match self.side {
                Side::Buy => 1.0,
                Side::Sell => -1.0,
            };
            direction * (exit - self.entry_price) * self.quantity
        })
    }

    /// Calculate the net P&L (after costs).
    pub fn net_pnl(&self) -> Option<f64> {
        self.gross_pnl()
            .map(|gross| gross - self.commission - self.slippage * self.quantity)
    }

    /// Calculate return percentage.
    pub fn return_pct(&self) -> Option<f64> {
        self.exit_price.map(|exit| {
            let direction = match self.side {
                Side::Buy => 1.0,
                Side::Sell => -1.0,
            };
            direction * (exit - self.entry_price) / self.entry_price * 100.0
        })
    }

    /// Calculate holding period in bars.
    pub fn holding_period(&self) -> Option<chrono::Duration> {
        self.exit_time
            .map(|exit| exit.signed_duration_since(self.entry_time))
    }
}

/// Current position in a symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_entry_price: f64,
    pub side: Side,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

impl Position {
    /// Create a new position.
    pub fn new(symbol: impl Into<String>, side: Side, quantity: f64, entry_price: f64) -> Self {
        Self {
            symbol: symbol.into(),
            quantity,
            avg_entry_price: entry_price,
            side,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        }
    }

    /// Update unrealized P&L based on current price.
    pub fn update_pnl(&mut self, current_price: f64) {
        let direction = match self.side {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        };
        self.unrealized_pnl = direction * (current_price - self.avg_entry_price) * self.quantity;
    }

    /// Calculate market value of position.
    pub fn market_value(&self, current_price: f64) -> f64 {
        self.quantity * current_price
    }

    /// Check if position is long.
    pub fn is_long(&self) -> bool {
        matches!(self.side, Side::Buy)
    }

    /// Check if position is short.
    pub fn is_short(&self) -> bool {
        matches!(self.side, Side::Sell)
    }
}

/// Tracks a specific entry lot for precise cost-basis accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxLot {
    pub id: Uuid,
    pub quantity: f64,
    pub cost_basis: f64,
    pub acquired_date: DateTime<Utc>,
    pub side: Side,
}

impl TaxLot {
    /// Create a new lot tied to an execution.
    pub fn new(side: Side, quantity: f64, cost_basis: f64, acquired_date: DateTime<Utc>) -> Self {
        Self {
            id: Uuid::new_v4(),
            quantity,
            cost_basis,
            acquired_date,
            side,
        }
    }

    /// Reduce the lot by a specified quantity, returning the amount actually removed.
    pub fn consume(&mut self, qty: f64) -> f64 {
        if qty <= 0.0 {
            return 0.0;
        }
        let taken = self.quantity.min(qty);
        self.quantity -= taken;
        taken
    }

    /// True when the lot has been fully closed.
    pub fn is_empty(&self) -> bool {
        self.quantity <= f64::EPSILON
    }
}

/// Equity snapshot at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub timestamp: DateTime<Utc>,
    pub equity: f64,
    pub cash: f64,
    pub positions_value: f64,
    pub drawdown: f64,
    pub drawdown_pct: f64,
}

/// Signal generated by a strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Signal {
    /// Enter long position.
    Long,
    /// Enter short position.
    Short,
    /// Exit current position.
    Exit,
    /// Do nothing.
    #[default]
    Hold,
}

/// Strategy validation verdict based on out-of-sample performance.
///
/// The verdict classifies strategies based on their OOS/IS degradation ratio:
/// - `Robust`: Degradation > 0.80 (excellent - likely to work in production)
/// - `Borderline`: Degradation 0.60-0.80 (acceptable - proceed with caution)
/// - `LikelyOverfit`: Degradation < 0.60 (concerning - probably won't work live)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Verdict {
    /// Strategy shows excellent out-of-sample performance relative to in-sample.
    /// OOS/IS ratio > 0.80, indicating robust performance likely to persist.
    Robust,
    /// Strategy shows acceptable but not exceptional out-of-sample performance.
    /// OOS/IS ratio 0.60-0.80, indicating some overfitting risk.
    #[default]
    Borderline,
    /// Strategy shows significant degradation out-of-sample.
    /// OOS/IS ratio < 0.60, indicating high probability of overfitting.
    LikelyOverfit,
}

impl Verdict {
    /// Classify based on OOS/IS degradation ratio.
    ///
    /// # Arguments
    /// * `degradation_ratio` - The ratio of OOS performance to IS performance (0.0-1.0+).
    ///   Values above 1.0 indicate OOS outperformed IS (rare but possible).
    ///
    /// # Thresholds
    /// - > 0.80: Robust
    /// - 0.60-0.80: Borderline
    /// - < 0.60: LikelyOverfit
    pub fn from_degradation_ratio(degradation_ratio: f64) -> Self {
        if degradation_ratio > 0.80 {
            Verdict::Robust
        } else if degradation_ratio >= 0.60 {
            Verdict::Borderline
        } else {
            Verdict::LikelyOverfit
        }
    }

    /// Classify based on multiple robustness criteria.
    ///
    /// This is a more nuanced classification that considers:
    /// - OOS/IS degradation ratio
    /// - Whether OOS returns are positive
    /// - Walk-forward efficiency
    ///
    /// # Arguments
    /// * `degradation_ratio` - OOS/IS ratio (typically Sharpe or return based)
    /// * `oos_positive` - Whether out-of-sample returns are positive
    /// * `efficiency` - Walk-forward efficiency (combined OOS / combined IS)
    pub fn from_criteria(degradation_ratio: f64, oos_positive: bool, efficiency: f64) -> Self {
        // Negative OOS returns are always a red flag
        if !oos_positive {
            return Verdict::LikelyOverfit;
        }

        // Very low efficiency indicates serious issues
        if efficiency < 0.40 {
            return Verdict::LikelyOverfit;
        }

        // Use degradation ratio as primary classifier
        if degradation_ratio > 0.80 && efficiency >= 0.60 {
            Verdict::Robust
        } else if degradation_ratio >= 0.60 || efficiency >= 0.50 {
            Verdict::Borderline
        } else {
            Verdict::LikelyOverfit
        }
    }

    /// Returns true if the verdict indicates a potentially viable strategy.
    pub fn is_acceptable(&self) -> bool {
        matches!(self, Verdict::Robust | Verdict::Borderline)
    }

    /// Returns a human-readable description of the verdict.
    pub fn description(&self) -> &'static str {
        match self {
            Verdict::Robust => "Strategy appears robust. Out-of-sample performance is strong relative to in-sample.",
            Verdict::Borderline => "Strategy is borderline. Consider more data or simpler model before production use.",
            Verdict::LikelyOverfit => "Strategy is likely overfit. Significant performance degradation out-of-sample.",
        }
    }

    /// Returns a short label for display purposes.
    pub fn label(&self) -> &'static str {
        match self {
            Verdict::Robust => "robust",
            Verdict::Borderline => "borderline",
            Verdict::LikelyOverfit => "likely_overfit",
        }
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// =============================================================================
// Corporate Actions
// =============================================================================

/// Type of dividend payment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum DividendType {
    /// Regular cash dividend.
    #[default]
    Cash,
    /// Stock dividend (shares instead of cash).
    Stock,
    /// Special one-time dividend.
    Special,
}

/// Type of corporate action affecting share price and/or quantity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CorporateActionType {
    /// Stock split - ratio > 1 means more shares (e.g., 2.0 for 2-for-1 split).
    /// Price is divided by ratio, quantity multiplied by ratio.
    Split {
        /// Split ratio (e.g., 2.0 for 2-for-1 split, 3.0 for 3-for-1).
        ratio: f64,
    },
    /// Reverse split - ratio < 1 means fewer shares (e.g., 0.1 for 1-for-10 reverse).
    /// Price is divided by ratio, quantity multiplied by ratio.
    ReverseSplit {
        /// Reverse split ratio (e.g., 0.1 for 1-for-10 reverse split).
        ratio: f64,
    },
    /// Dividend payment.
    Dividend {
        /// Dividend amount per share.
        amount: f64,
        /// Type of dividend.
        div_type: DividendType,
    },
    /// Spin-off creates new company shares from existing company.
    SpinOff {
        /// Ratio of new shares received per existing share.
        ratio: f64,
        /// Symbol of the new spun-off company.
        new_symbol: String,
    },
}

/// A corporate action event with timing information.
///
/// Corporate actions affect historical price data and must be accounted for
/// when backtesting to avoid look-ahead bias and ensure accurate returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorporateAction {
    /// Symbol this action applies to.
    pub symbol: String,
    /// Type of corporate action.
    pub action_type: CorporateActionType,
    /// Ex-dividend/ex-split date - the date on or after which the stock trades
    /// without the entitlement to the dividend/split.
    pub ex_date: DateTime<Utc>,
    /// Record date - shareholders as of this date receive the action.
    /// Optional as not all corporate actions have explicit record dates.
    pub record_date: Option<DateTime<Utc>>,
    /// Payment/effective date - when the action takes effect.
    /// For dividends, when payment is made. For splits, when new shares appear.
    pub pay_date: Option<DateTime<Utc>>,
}

impl CorporateAction {
    /// Create a new stock split action.
    pub fn split(symbol: impl Into<String>, ratio: f64, ex_date: DateTime<Utc>) -> Self {
        Self {
            symbol: symbol.into(),
            action_type: CorporateActionType::Split { ratio },
            ex_date,
            record_date: None,
            pay_date: None,
        }
    }

    /// Create a new reverse split action.
    pub fn reverse_split(symbol: impl Into<String>, ratio: f64, ex_date: DateTime<Utc>) -> Self {
        Self {
            symbol: symbol.into(),
            action_type: CorporateActionType::ReverseSplit { ratio },
            ex_date,
            record_date: None,
            pay_date: None,
        }
    }

    /// Create a new cash dividend action.
    pub fn cash_dividend(symbol: impl Into<String>, amount: f64, ex_date: DateTime<Utc>) -> Self {
        Self {
            symbol: symbol.into(),
            action_type: CorporateActionType::Dividend {
                amount,
                div_type: DividendType::Cash,
            },
            ex_date,
            record_date: None,
            pay_date: None,
        }
    }

    /// Create a new dividend action with specified type.
    pub fn dividend(
        symbol: impl Into<String>,
        amount: f64,
        div_type: DividendType,
        ex_date: DateTime<Utc>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            action_type: CorporateActionType::Dividend { amount, div_type },
            ex_date,
            record_date: None,
            pay_date: None,
        }
    }

    /// Create a new spin-off action.
    pub fn spin_off(
        symbol: impl Into<String>,
        ratio: f64,
        new_symbol: impl Into<String>,
        ex_date: DateTime<Utc>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            action_type: CorporateActionType::SpinOff {
                ratio,
                new_symbol: new_symbol.into(),
            },
            ex_date,
            record_date: None,
            pay_date: None,
        }
    }

    /// Set the record date.
    pub fn with_record_date(mut self, record_date: DateTime<Utc>) -> Self {
        self.record_date = Some(record_date);
        self
    }

    /// Set the pay date.
    pub fn with_pay_date(mut self, pay_date: DateTime<Utc>) -> Self {
        self.pay_date = Some(pay_date);
        self
    }

    /// Get the adjustment factor for this action.
    ///
    /// For splits: returns the ratio (e.g., 2.0 for 2-for-1 split).
    /// For reverse splits: returns the ratio (e.g., 0.1 for 1-for-10).
    /// For dividends and spin-offs: returns 1.0 (no direct price adjustment).
    pub fn adjustment_factor(&self) -> f64 {
        match &self.action_type {
            CorporateActionType::Split { ratio } => *ratio,
            CorporateActionType::ReverseSplit { ratio } => *ratio,
            CorporateActionType::Dividend { .. } => 1.0,
            CorporateActionType::SpinOff { .. } => 1.0,
        }
    }

    /// Check if this action requires price adjustment (splits only).
    pub fn requires_price_adjustment(&self) -> bool {
        matches!(
            self.action_type,
            CorporateActionType::Split { .. } | CorporateActionType::ReverseSplit { .. }
        )
    }

    /// Check if this is a dividend action.
    pub fn is_dividend(&self) -> bool {
        matches!(self.action_type, CorporateActionType::Dividend { .. })
    }

    /// Get dividend amount if this is a dividend action.
    pub fn dividend_amount(&self) -> Option<f64> {
        match &self.action_type {
            CorporateActionType::Dividend { amount, .. } => Some(*amount),
            _ => None,
        }
    }
}

/// Method for adjusting prices for dividends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DividendAdjustMethod {
    /// Subtract dividend from prices before ex-date (proportional adjustment).
    /// This is the standard method used by most data providers.
    #[default]
    Proportional,
    /// Subtract the absolute dividend amount from prices.
    Absolute,
    /// No adjustment for dividends.
    None,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};

    fn sample_timestamp() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 1, 15, 9, 30, 0).unwrap()
    }

    #[test]
    fn test_volume_profile_from_bars() {
        let bars = vec![
            Bar::new(sample_timestamp(), 100.0, 102.0, 98.0, 101.0, 1000.0),
            Bar::new(
                sample_timestamp() + Duration::hours(1),
                101.0,
                103.0,
                99.0,
                102.0,
                2000.0,
            ),
            Bar::new(
                sample_timestamp() + Duration::days(1),
                103.0,
                105.0,
                102.0,
                104.0,
                3000.0,
            ),
        ];

        let profile = VolumeProfile::from_bars(&bars).unwrap();
        assert!((profile.avg_bar_volume - 2000.0).abs() < f64::EPSILON);
        assert!((profile.avg_daily_volume - 3000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_volume_profile_reference_volume() {
        let profile = VolumeProfile::new(10_000.0, 0.0);
        assert!((profile.reference_volume() - 10_000.0).abs() < f64::EPSILON);

        let profile = VolumeProfile::new(10_000.0, 1_000.0);
        assert!((profile.reference_volume() - 1_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bar_validation() {
        let valid_bar = Bar::new(sample_timestamp(), 100.0, 105.0, 98.0, 102.0, 1000.0);
        assert!(valid_bar.validate());

        // High below low - invalid
        let invalid_bar = Bar::new(sample_timestamp(), 100.0, 95.0, 98.0, 102.0, 1000.0);
        assert!(!invalid_bar.validate());

        // Negative volume - invalid
        let invalid_bar2 = Bar::new(sample_timestamp(), 100.0, 105.0, 98.0, 102.0, -100.0);
        assert!(!invalid_bar2.validate());
    }

    #[test]
    fn test_bar_calculations() {
        let bar = Bar::new(sample_timestamp(), 100.0, 110.0, 90.0, 105.0, 1000.0);

        assert!((bar.typical_price() - 101.666666).abs() < 0.001);
        assert!((bar.range() - 20.0).abs() < f64::EPSILON);
        assert!((bar.body() - 5.0).abs() < f64::EPSILON);
        assert!(bar.is_bullish());
    }

    #[test]
    fn test_order_validation() {
        let valid_order = Order::market("AAPL", Side::Buy, 100.0, sample_timestamp());
        assert!(valid_order.validate());

        let invalid_order = Order::market("", Side::Buy, 100.0, sample_timestamp());
        assert!(!invalid_order.validate());

        let invalid_qty = Order::market("AAPL", Side::Buy, -10.0, sample_timestamp());
        assert!(!invalid_qty.validate());
    }

    #[test]
    fn test_trade_pnl() {
        let mut trade = Trade::open(
            "AAPL",
            Side::Buy,
            100.0,
            150.0,
            sample_timestamp(),
            1.0,
            0.01,
        );

        assert!(!trade.is_closed());
        assert!(trade.gross_pnl().is_none());

        let exit_time = Utc.with_ymd_and_hms(2024, 1, 16, 9, 30, 0).unwrap();
        trade.close(160.0, exit_time, 1.0);

        assert!(trade.is_closed());
        assert!((trade.gross_pnl().unwrap() - 1000.0).abs() < f64::EPSILON);
        // Net P&L = 1000 - 2 (commission) - 1 (slippage * qty) = 997
        assert!((trade.net_pnl().unwrap() - 997.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trade_short_pnl() {
        let mut trade = Trade::open(
            "AAPL",
            Side::Sell,
            100.0,
            160.0,
            sample_timestamp(),
            1.0,
            0.0,
        );

        let exit_time = Utc.with_ymd_and_hms(2024, 1, 16, 9, 30, 0).unwrap();
        trade.close(150.0, exit_time, 1.0);

        // Short: sold at 160, bought back at 150, profit = 10 * 100 = 1000
        assert!((trade.gross_pnl().unwrap() - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_position_pnl() {
        let mut position = Position::new("AAPL", Side::Buy, 100.0, 150.0);
        position.update_pnl(160.0);

        assert!((position.unrealized_pnl - 1000.0).abs() < f64::EPSILON);
        assert!(position.is_long());
    }

    // =========================================================================
    // Corporate Actions Tests
    // =========================================================================

    #[test]
    fn test_corporate_action_split() {
        let ex_date = Utc.with_ymd_and_hms(2024, 6, 10, 0, 0, 0).unwrap();
        let action = CorporateAction::split("AAPL", 4.0, ex_date);

        assert_eq!(action.symbol, "AAPL");
        assert_eq!(action.ex_date, ex_date);
        assert!((action.adjustment_factor() - 4.0).abs() < f64::EPSILON);
        assert!(action.requires_price_adjustment());
        assert!(!action.is_dividend());
        assert!(action.dividend_amount().is_none());
    }

    #[test]
    fn test_corporate_action_reverse_split() {
        let ex_date = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
        let action = CorporateAction::reverse_split("SIRI", 0.1, ex_date);

        assert_eq!(action.symbol, "SIRI");
        assert!((action.adjustment_factor() - 0.1).abs() < f64::EPSILON);
        assert!(action.requires_price_adjustment());
    }

    #[test]
    fn test_corporate_action_dividend() {
        let ex_date = Utc.with_ymd_and_hms(2024, 2, 9, 0, 0, 0).unwrap();
        let record_date = Utc.with_ymd_and_hms(2024, 2, 12, 0, 0, 0).unwrap();
        let pay_date = Utc.with_ymd_and_hms(2024, 2, 15, 0, 0, 0).unwrap();

        let action = CorporateAction::cash_dividend("MSFT", 0.75, ex_date)
            .with_record_date(record_date)
            .with_pay_date(pay_date);

        assert_eq!(action.symbol, "MSFT");
        assert!((action.adjustment_factor() - 1.0).abs() < f64::EPSILON);
        assert!(!action.requires_price_adjustment());
        assert!(action.is_dividend());
        assert!((action.dividend_amount().unwrap() - 0.75).abs() < f64::EPSILON);
        assert_eq!(action.record_date, Some(record_date));
        assert_eq!(action.pay_date, Some(pay_date));
    }

    #[test]
    fn test_corporate_action_special_dividend() {
        let ex_date = Utc.with_ymd_and_hms(2024, 12, 1, 0, 0, 0).unwrap();
        let action = CorporateAction::dividend("COST", 15.0, DividendType::Special, ex_date);

        assert!(action.is_dividend());
        match &action.action_type {
            CorporateActionType::Dividend { amount, div_type } => {
                assert!((*amount - 15.0).abs() < f64::EPSILON);
                assert_eq!(*div_type, DividendType::Special);
            }
            _ => panic!("Expected Dividend action type"),
        }
    }

    #[test]
    fn test_corporate_action_spinoff() {
        let ex_date = Utc.with_ymd_and_hms(2024, 4, 3, 0, 0, 0).unwrap();
        let action = CorporateAction::spin_off("GE", 0.25, "GEV", ex_date);

        assert_eq!(action.symbol, "GE");
        assert!(!action.requires_price_adjustment());
        assert!(!action.is_dividend());

        match &action.action_type {
            CorporateActionType::SpinOff { ratio, new_symbol } => {
                assert!((*ratio - 0.25).abs() < f64::EPSILON);
                assert_eq!(new_symbol, "GEV");
            }
            _ => panic!("Expected SpinOff action type"),
        }
    }

    #[test]
    fn test_corporate_action_serialization() {
        let ex_date = Utc.with_ymd_and_hms(2024, 6, 10, 0, 0, 0).unwrap();
        let action = CorporateAction::split("AAPL", 4.0, ex_date);

        let json = serde_json::to_string(&action).unwrap();
        let deserialized: CorporateAction = serde_json::from_str(&json).unwrap();

        assert_eq!(action, deserialized);
    }

    #[test]
    fn test_dividend_type_default() {
        let default_type = DividendType::default();
        assert_eq!(default_type, DividendType::Cash);
    }

    // =========================================================================
    // Verdict Tests
    // =========================================================================

    #[test]
    fn test_verdict_from_degradation_ratio_robust() {
        // > 0.80 should be Robust
        assert_eq!(Verdict::from_degradation_ratio(0.85), Verdict::Robust);
        assert_eq!(Verdict::from_degradation_ratio(0.95), Verdict::Robust);
        assert_eq!(Verdict::from_degradation_ratio(1.2), Verdict::Robust);
    }

    #[test]
    fn test_verdict_from_degradation_ratio_borderline() {
        // 0.60-0.80 should be Borderline
        assert_eq!(Verdict::from_degradation_ratio(0.60), Verdict::Borderline);
        assert_eq!(Verdict::from_degradation_ratio(0.70), Verdict::Borderline);
        assert_eq!(Verdict::from_degradation_ratio(0.80), Verdict::Borderline);
    }

    #[test]
    fn test_verdict_from_degradation_ratio_likely_overfit() {
        // < 0.60 should be LikelyOverfit
        assert_eq!(
            Verdict::from_degradation_ratio(0.59),
            Verdict::LikelyOverfit
        );
        assert_eq!(
            Verdict::from_degradation_ratio(0.30),
            Verdict::LikelyOverfit
        );
        assert_eq!(Verdict::from_degradation_ratio(0.0), Verdict::LikelyOverfit);
        assert_eq!(
            Verdict::from_degradation_ratio(-0.5),
            Verdict::LikelyOverfit
        );
    }

    #[test]
    fn test_verdict_from_criteria_negative_oos() {
        // Negative OOS returns should always be LikelyOverfit
        assert_eq!(
            Verdict::from_criteria(0.90, false, 0.80),
            Verdict::LikelyOverfit
        );
    }

    #[test]
    fn test_verdict_from_criteria_low_efficiency() {
        // Very low efficiency (< 0.40) should be LikelyOverfit
        assert_eq!(
            Verdict::from_criteria(0.90, true, 0.35),
            Verdict::LikelyOverfit
        );
    }

    #[test]
    fn test_verdict_from_criteria_robust() {
        // High degradation ratio and good efficiency should be Robust
        assert_eq!(Verdict::from_criteria(0.85, true, 0.70), Verdict::Robust);
    }

    #[test]
    fn test_verdict_from_criteria_borderline() {
        // Moderate degradation or efficiency should be Borderline
        assert_eq!(
            Verdict::from_criteria(0.65, true, 0.55),
            Verdict::Borderline
        );
    }

    #[test]
    fn test_verdict_is_acceptable() {
        assert!(Verdict::Robust.is_acceptable());
        assert!(Verdict::Borderline.is_acceptable());
        assert!(!Verdict::LikelyOverfit.is_acceptable());
    }

    #[test]
    fn test_verdict_label() {
        assert_eq!(Verdict::Robust.label(), "robust");
        assert_eq!(Verdict::Borderline.label(), "borderline");
        assert_eq!(Verdict::LikelyOverfit.label(), "likely_overfit");
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", Verdict::Robust), "robust");
        assert_eq!(format!("{}", Verdict::Borderline), "borderline");
        assert_eq!(format!("{}", Verdict::LikelyOverfit), "likely_overfit");
    }

    #[test]
    fn test_verdict_default() {
        // Default should be Borderline (most conservative assumption)
        assert_eq!(Verdict::default(), Verdict::Borderline);
    }

    #[test]
    fn test_verdict_serialization() {
        let verdict = Verdict::Robust;
        let json = serde_json::to_string(&verdict).unwrap();
        let deserialized: Verdict = serde_json::from_str(&json).unwrap();
        assert_eq!(verdict, deserialized);

        let verdict = Verdict::LikelyOverfit;
        let json = serde_json::to_string(&verdict).unwrap();
        let deserialized: Verdict = serde_json::from_str(&json).unwrap();
        assert_eq!(verdict, deserialized);
    }

    // =========================================================================
    // DataFrequency Tests
    // =========================================================================

    #[test]
    fn test_data_frequency_default() {
        assert_eq!(DataFrequency::default(), DataFrequency::Day);
    }

    #[test]
    fn test_data_frequency_to_seconds() {
        assert_eq!(DataFrequency::Second1.to_seconds(), 1);
        assert_eq!(DataFrequency::Minute1.to_seconds(), 60);
        assert_eq!(DataFrequency::Minute5.to_seconds(), 300);
        assert_eq!(DataFrequency::Hour1.to_seconds(), 3600);
        assert_eq!(DataFrequency::Hour4.to_seconds(), 14400);
        assert_eq!(DataFrequency::Day.to_seconds(), 86400);
        assert_eq!(DataFrequency::Week.to_seconds(), 604800);
    }

    #[test]
    fn test_data_frequency_annualization_factor_traditional() {
        // Traditional markets (stocks, futures, forex)
        let factor = DataFrequency::Day.annualization_factor(false);
        assert!((factor - 252.0).abs() < f64::EPSILON);

        let factor = DataFrequency::Hour1.annualization_factor(false);
        assert!((factor - 1638.0).abs() < f64::EPSILON);

        let factor = DataFrequency::Minute1.annualization_factor(false);
        assert!((factor - 98_280.0).abs() < f64::EPSILON);

        let factor = DataFrequency::Week.annualization_factor(false);
        assert!((factor - 52.0).abs() < f64::EPSILON);

        let factor = DataFrequency::Month.annualization_factor(false);
        assert!((factor - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_data_frequency_annualization_factor_24h() {
        // 24/7 markets (crypto)
        let factor = DataFrequency::Day.annualization_factor(true);
        assert!((factor - 365.0).abs() < f64::EPSILON);

        let factor = DataFrequency::Hour1.annualization_factor(true);
        assert!((factor - 8_760.0).abs() < f64::EPSILON);

        let factor = DataFrequency::Minute1.annualization_factor(true);
        assert!((factor - 525_600.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_data_frequency_detect_daily() {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let bars: Vec<Bar> = (0..10)
            .map(|i| Bar::new(base + Duration::days(i), 100.0, 105.0, 95.0, 102.0, 1000.0))
            .collect();

        assert_eq!(DataFrequency::detect(&bars), DataFrequency::Day);
    }

    #[test]
    fn test_data_frequency_detect_minute() {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 9, 30, 0).unwrap();
        let bars: Vec<Bar> = (0..100)
            .map(|i| {
                Bar::new(
                    base + Duration::minutes(i),
                    100.0,
                    105.0,
                    95.0,
                    102.0,
                    1000.0,
                )
            })
            .collect();

        assert_eq!(DataFrequency::detect(&bars), DataFrequency::Minute1);
    }

    #[test]
    fn test_data_frequency_detect_5_minute() {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 9, 30, 0).unwrap();
        let bars: Vec<Bar> = (0..100)
            .map(|i| {
                Bar::new(
                    base + Duration::minutes(i * 5),
                    100.0,
                    105.0,
                    95.0,
                    102.0,
                    1000.0,
                )
            })
            .collect();

        assert_eq!(DataFrequency::detect(&bars), DataFrequency::Minute5);
    }

    #[test]
    fn test_data_frequency_detect_hourly() {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let bars: Vec<Bar> = (0..50)
            .map(|i| Bar::new(base + Duration::hours(i), 100.0, 105.0, 95.0, 102.0, 1000.0))
            .collect();

        assert_eq!(DataFrequency::detect(&bars), DataFrequency::Hour1);
    }

    #[test]
    fn test_data_frequency_detect_insufficient_data() {
        // With only 1 bar, should return default (Day)
        let bars = vec![Bar::new(
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            100.0,
            105.0,
            95.0,
            102.0,
            1000.0,
        )];

        assert_eq!(DataFrequency::detect(&bars), DataFrequency::Day);

        // Empty bars also returns Day
        let empty: Vec<Bar> = vec![];
        assert_eq!(DataFrequency::detect(&empty), DataFrequency::Day);
    }

    #[test]
    fn test_data_frequency_is_likely_crypto() {
        // Create bars with weekend data (crypto marker)
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(); // Monday
        let bars: Vec<Bar> = (0..30)
            .map(|i| Bar::new(base + Duration::days(i), 100.0, 105.0, 95.0, 102.0, 1000.0))
            .collect();

        // This should have weekend bars (Jan 6-7, 13-14, 20-21, 27-28)
        assert!(DataFrequency::is_likely_crypto(&bars));

        // Create bars without weekend data (traditional market)
        let weekday_bars: Vec<Bar> = (0..20)
            .filter_map(|i| {
                let dt = base + Duration::days(i);
                let weekday = dt.weekday().num_days_from_sunday();
                // Skip weekends
                if weekday == 0 || weekday == 6 {
                    None
                } else {
                    Some(Bar::new(dt, 100.0, 105.0, 95.0, 102.0, 1000.0))
                }
            })
            .collect();

        assert!(!DataFrequency::is_likely_crypto(&weekday_bars));
    }

    #[test]
    fn test_data_frequency_description() {
        assert_eq!(DataFrequency::Day.description(), "daily");
        assert_eq!(DataFrequency::Minute1.description(), "1-minute");
        assert_eq!(DataFrequency::Hour4.description(), "4-hour");
        assert_eq!(DataFrequency::Week.description(), "weekly");
    }

    #[test]
    fn test_data_frequency_display() {
        assert_eq!(format!("{}", DataFrequency::Day), "daily");
        assert_eq!(format!("{}", DataFrequency::Minute5), "5-minute");
        assert_eq!(format!("{}", DataFrequency::Hour1), "hourly");
    }

    #[test]
    fn test_data_frequency_serialization() {
        let freq = DataFrequency::Minute15;
        let json = serde_json::to_string(&freq).unwrap();
        let deserialized: DataFrequency = serde_json::from_str(&json).unwrap();
        assert_eq!(freq, deserialized);

        // Test kebab-case serialization
        let freq = DataFrequency::Hour4;
        let json = serde_json::to_string(&freq).unwrap();
        assert!(json.contains("hour4") || json.contains("Hour4"));
    }
}

//! Core data types for the backtest engine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

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

/// Supported asset classes in the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AssetClass {
    /// Traditional equities where notional = price * quantity.
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

impl Default for AssetClass {
    fn default() -> Self {
        AssetClass::Equity
    }
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
        }
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
    use chrono::TimeZone;

    fn sample_timestamp() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 1, 15, 9, 30, 0).unwrap()
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
}

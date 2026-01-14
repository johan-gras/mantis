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
}

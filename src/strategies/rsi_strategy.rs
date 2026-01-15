//! RSI-based Strategy.
//!
//! Uses the Relative Strength Index to identify overbought and oversold conditions.

use crate::data::rsi;
use crate::strategy::{Strategy, StrategyContext};
use crate::types::Signal;

/// RSI Strategy.
///
/// # Parameters
/// - `period`: RSI calculation period (default: 14)
/// - `oversold`: RSI level considered oversold (default: 30)
/// - `overbought`: RSI level considered overbought (default: 70)
///
/// # Signals
/// - Long: RSI crosses above oversold level
/// - Short: RSI crosses below overbought level
/// - Exit: RSI returns to neutral zone
#[derive(Debug, Clone)]
pub struct RsiStrategy {
    period: usize,
    oversold: f64,
    overbought: f64,
    prev_rsi: Option<f64>,
}

impl RsiStrategy {
    /// Create a new RSI strategy.
    pub fn new(period: usize, oversold: f64, overbought: f64) -> Self {
        assert!(period > 0, "Period must be positive");
        assert!(
            oversold < overbought,
            "Oversold must be less than overbought"
        );
        assert!(
            oversold >= 0.0 && overbought <= 100.0,
            "Thresholds must be between 0-100"
        );

        Self {
            period,
            oversold,
            overbought,
            prev_rsi: None,
        }
    }

    /// Create with default parameters (14, 30, 70).
    pub fn default_params() -> Self {
        Self::new(14, 30.0, 70.0)
    }

    /// Create with aggressive parameters for more signals.
    pub fn aggressive() -> Self {
        Self::new(7, 25.0, 75.0)
    }

    /// Create with conservative parameters for fewer signals.
    pub fn conservative() -> Self {
        Self::new(21, 20.0, 80.0)
    }
}

impl Strategy for RsiStrategy {
    fn name(&self) -> &str {
        "RSI Strategy"
    }

    fn init(&mut self) {
        self.prev_rsi = None;
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let history = ctx.history();
        let current_rsi = match rsi(history, self.period) {
            Some(r) => r,
            None => return Signal::Hold,
        };

        let signal = match self.prev_rsi {
            Some(prev) => {
                if ctx.is_flat() {
                    // Entry conditions
                    if prev <= self.oversold && current_rsi > self.oversold {
                        // RSI crossed above oversold - bullish
                        Signal::Long
                    } else if prev >= self.overbought && current_rsi < self.overbought {
                        // RSI crossed below overbought - bearish
                        Signal::Short
                    } else {
                        Signal::Hold
                    }
                } else if ctx.is_long() {
                    // Exit long when RSI reaches overbought
                    if current_rsi >= self.overbought {
                        Signal::Exit
                    } else {
                        Signal::Hold
                    }
                } else {
                    // Exit short when RSI reaches oversold
                    if current_rsi <= self.oversold {
                        Signal::Exit
                    } else {
                        Signal::Hold
                    }
                }
            }
            None => Signal::Hold,
        };

        self.prev_rsi = Some(current_rsi);
        signal
    }

    fn warmup_period(&self) -> usize {
        self.period + 1
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("period".to_string(), self.period.to_string()),
            ("oversold".to_string(), format!("{:.0}", self.oversold)),
            ("overbought".to_string(), format!("{:.0}", self.overbought)),
        ]
    }
}

/// RSI Divergence Strategy.
///
/// Looks for divergences between price and RSI.
#[derive(Debug, Clone)]
pub struct RsiDivergence {
    period: usize,
    lookback: usize,
    prev_rsi: Option<f64>,
    prev_price: Option<f64>,
}

impl RsiDivergence {
    pub fn new(period: usize, lookback: usize) -> Self {
        Self {
            period,
            lookback,
            prev_rsi: None,
            prev_price: None,
        }
    }
}

impl Strategy for RsiDivergence {
    fn name(&self) -> &str {
        "RSI Divergence"
    }

    fn init(&mut self) {
        self.prev_rsi = None;
        self.prev_price = None;
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let history = ctx.history();
        let current_rsi = match rsi(history, self.period) {
            Some(r) => r,
            None => return Signal::Hold,
        };
        let current_price = ctx.current_bar().close;

        // Get price and RSI from lookback bars ago
        let past_bar = match ctx.bar_at(self.lookback) {
            Some(b) => b,
            None => return Signal::Hold,
        };

        let past_history = &ctx.bars[..=ctx.bar_index - self.lookback];
        let past_rsi = match rsi(past_history, self.period) {
            Some(r) => r,
            None => return Signal::Hold,
        };
        let past_price = past_bar.close;

        let signal = if ctx.is_flat() {
            // Bullish divergence: price makes lower low, RSI makes higher low
            if current_price < past_price && current_rsi > past_rsi {
                Signal::Long
            }
            // Bearish divergence: price makes higher high, RSI makes lower high
            else if current_price > past_price && current_rsi < past_rsi {
                Signal::Short
            } else {
                Signal::Hold
            }
        } else {
            // Simple exit on RSI extremes
            if (ctx.is_long() && current_rsi > 70.0) || (ctx.is_short() && current_rsi < 30.0) {
                Signal::Exit
            } else {
                Signal::Hold
            }
        };

        self.prev_rsi = Some(current_rsi);
        self.prev_price = Some(current_price);

        signal
    }

    fn warmup_period(&self) -> usize {
        self.period + self.lookback + 1
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("period".to_string(), self.period.to_string()),
            ("lookback".to_string(), self.lookback.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;
    use chrono::{TimeZone, Utc};

    fn create_oscillating_bars() -> Vec<Bar> {
        // Create bars that oscillate to trigger RSI signals
        let mut bars = Vec::new();
        for i in 0..30 {
            let phase = (i as f64 * 0.5).sin();
            let base = 100.0 + phase * 10.0;
            bars.push(Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i),
                base - 1.0,
                base + 2.0,
                base - 2.0,
                base + 1.0,
                1000.0,
            ));
        }
        bars
    }

    #[test]
    fn test_rsi_strategy_creation() {
        let strategy = RsiStrategy::new(14, 30.0, 70.0);
        assert_eq!(strategy.period, 14);
        assert_eq!(strategy.oversold, 30.0);
        assert_eq!(strategy.overbought, 70.0);
    }

    #[test]
    fn test_rsi_warmup() {
        let strategy = RsiStrategy::new(14, 30.0, 70.0);
        assert_eq!(strategy.warmup_period(), 15);
    }

    #[test]
    fn test_rsi_parameters() {
        let strategy = RsiStrategy::aggressive();
        let params = strategy.parameters();
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].1, "7");
    }

    #[test]
    fn test_rsi_on_bars() {
        let mut strategy = RsiStrategy::default_params();
        strategy.init();

        let bars = create_oscillating_bars();

        // Run through all bars
        for i in 0..bars.len() {
            let ctx = StrategyContext {
                bar_index: i,
                bars: &bars,
                position: 0.0,
                cash: 100000.0,
                equity: 100000.0,
                symbol: "TEST",
                volume_profile: None,
            };
            let _ = strategy.on_bar(&ctx);
        }

        // Strategy should have updated prev_rsi
        assert!(strategy.prev_rsi.is_some());
    }
}

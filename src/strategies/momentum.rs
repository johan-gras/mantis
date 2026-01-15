//! Momentum Strategy.
//!
//! This strategy buys when the price has positive momentum over a lookback
//! period and exits when momentum turns negative.

use crate::strategy::{Strategy, StrategyContext};
use crate::types::Signal;

/// Momentum Strategy.
///
/// # Parameters
/// - `lookback`: Number of bars to measure momentum over (default: 20)
/// - `threshold`: Minimum momentum percentage to trigger entry (default: 0.0)
///
/// # Signals
/// - Long: Price is above its level N bars ago by at least threshold%
/// - Exit: Price falls below its level N bars ago
#[derive(Debug, Clone)]
pub struct MomentumStrategy {
    lookback: usize,
    threshold: f64,
}

impl MomentumStrategy {
    /// Create a new Momentum strategy.
    pub fn new(lookback: usize, threshold: f64) -> Self {
        assert!(lookback > 0, "Lookback must be positive");
        Self {
            lookback,
            threshold,
        }
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(20, 0.0)
    }

    /// Calculate momentum as percentage change.
    fn calculate_momentum(&self, ctx: &StrategyContext) -> Option<f64> {
        let current_price = ctx.current_bar().close;
        let past_bar = ctx.bar_at(self.lookback)?;
        let past_price = past_bar.close;

        if past_price == 0.0 {
            return None;
        }

        Some((current_price - past_price) / past_price * 100.0)
    }
}

impl Strategy for MomentumStrategy {
    fn name(&self) -> &str {
        "Momentum"
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let momentum = match self.calculate_momentum(ctx) {
            Some(m) => m,
            None => return Signal::Hold,
        };

        if ctx.is_flat() {
            if momentum > self.threshold {
                return Signal::Long;
            }
        } else if (ctx.is_long() && momentum < -self.threshold)
            || (ctx.is_short() && momentum > self.threshold)
        {
            return Signal::Exit;
        }

        Signal::Hold
    }

    fn warmup_period(&self) -> usize {
        self.lookback + 1
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("lookback".to_string(), self.lookback.to_string()),
            ("threshold".to_string(), format!("{:.2}%", self.threshold)),
        ]
    }
}

/// Rate of Change (ROC) Strategy - variant of momentum.
#[derive(Debug, Clone)]
pub struct RocStrategy {
    period: usize,
    entry_threshold: f64,
    exit_threshold: f64,
}

impl RocStrategy {
    /// Create a new ROC strategy.
    pub fn new(period: usize, entry_threshold: f64, exit_threshold: f64) -> Self {
        Self {
            period,
            entry_threshold,
            exit_threshold,
        }
    }

    fn calculate_roc(&self, ctx: &StrategyContext) -> Option<f64> {
        let current = ctx.current_bar().close;
        let past = ctx.bar_at(self.period)?.close;

        if past == 0.0 {
            return None;
        }

        Some((current - past) / past * 100.0)
    }
}

impl Strategy for RocStrategy {
    fn name(&self) -> &str {
        "Rate of Change"
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let roc = match self.calculate_roc(ctx) {
            Some(r) => r,
            None => return Signal::Hold,
        };

        if ctx.is_flat() && roc > self.entry_threshold {
            Signal::Long
        } else if ctx.is_long() && roc < self.exit_threshold {
            Signal::Exit
        } else {
            Signal::Hold
        }
    }

    fn warmup_period(&self) -> usize {
        self.period + 1
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("period".to_string(), self.period.to_string()),
            (
                "entry_threshold".to_string(),
                format!("{:.2}%", self.entry_threshold),
            ),
            (
                "exit_threshold".to_string(),
                format!("{:.2}%", self.exit_threshold),
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;
    use chrono::{TimeZone, Utc};

    fn create_test_bars() -> Vec<Bar> {
        vec![
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                100.0,
                102.0,
                99.0,
                101.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
                101.0,
                103.0,
                100.0,
                102.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
                102.0,
                105.0,
                101.0,
                104.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
                104.0,
                108.0,
                103.0,
                107.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(),
                107.0,
                112.0,
                106.0,
                110.0,
                1000.0,
            ),
        ]
    }

    #[test]
    fn test_momentum_strategy() {
        let mut strategy = MomentumStrategy::new(2, 0.0);
        let bars = create_test_bars();

        let ctx = StrategyContext {
            bar_index: 4,
            bars: &bars,
            position: 0.0,
            cash: 100000.0,
            equity: 100000.0,
            symbol: "TEST",
            volume_profile: None,
            timeframe_manager: None,
        };

        let signal = strategy.on_bar(&ctx);
        // Price went from 104 to 110 over 2 bars - positive momentum
        assert!(matches!(signal, Signal::Long));
    }

    #[test]
    fn test_momentum_warmup() {
        let strategy = MomentumStrategy::new(20, 1.0);
        assert_eq!(strategy.warmup_period(), 21);
    }
}

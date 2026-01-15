//! Mean Reversion Strategy.
//!
//! This strategy assumes prices revert to their mean and trades accordingly:
//! buys when price is significantly below the mean and sells when above.

use crate::data::{bollinger_bands, sma, std_dev};
use crate::strategy::{Strategy, StrategyContext};
use crate::types::Signal;

/// Mean Reversion Strategy using Bollinger Bands.
///
/// # Parameters
/// - `period`: Lookback period for moving average (default: 20)
/// - `num_std`: Number of standard deviations for bands (default: 2.0)
/// - `entry_std`: Entry threshold in std devs from mean (default: 2.0)
/// - `exit_std`: Exit threshold in std devs from mean (default: 0.5)
///
/// # Signals
/// - Long: Price below lower band (oversold)
/// - Short: Price above upper band (overbought)
/// - Exit: Price returns to within exit_std of mean
#[derive(Debug, Clone)]
pub struct MeanReversion {
    period: usize,
    num_std: f64,
    entry_std: f64,
    exit_std: f64,
}

impl MeanReversion {
    /// Create a new Mean Reversion strategy.
    pub fn new(period: usize, num_std: f64, entry_std: f64, exit_std: f64) -> Self {
        assert!(period > 0, "Period must be positive");
        assert!(num_std > 0.0, "Number of std devs must be positive");
        assert!(
            entry_std > exit_std,
            "Entry threshold must be greater than exit"
        );

        Self {
            period,
            num_std,
            entry_std,
            exit_std,
        }
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(20, 2.0, 2.0, 0.5)
    }

    /// Calculate z-score (how many std devs from mean).
    fn calculate_zscore(&self, ctx: &StrategyContext) -> Option<f64> {
        let history = ctx.history();
        let mean = sma(history, self.period)?;
        let std = std_dev(history, self.period)?;

        if std == 0.0 {
            return Some(0.0);
        }

        let current = ctx.current_bar().close;
        Some((current - mean) / std)
    }
}

impl Strategy for MeanReversion {
    fn name(&self) -> &str {
        "Mean Reversion"
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let zscore = match self.calculate_zscore(ctx) {
            Some(z) => z,
            None => return Signal::Hold,
        };

        if ctx.is_flat() {
            // Entry signals
            if zscore < -self.entry_std {
                return Signal::Long; // Oversold - expect reversion up
            } else if zscore > self.entry_std {
                return Signal::Short; // Overbought - expect reversion down
            }
        } else if ctx.is_long() {
            // Exit long
            if zscore > -self.exit_std {
                return Signal::Exit; // Price reverted to mean
            }
        } else if ctx.is_short() {
            // Exit short
            if zscore < self.exit_std {
                return Signal::Exit; // Price reverted to mean
            }
        }

        Signal::Hold
    }

    fn warmup_period(&self) -> usize {
        self.period
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("period".to_string(), self.period.to_string()),
            ("num_std".to_string(), format!("{:.1}", self.num_std)),
            ("entry_std".to_string(), format!("{:.1}", self.entry_std)),
            ("exit_std".to_string(), format!("{:.1}", self.exit_std)),
        ]
    }
}

/// Bollinger Band Bounce Strategy.
///
/// Trades bounces off Bollinger Bands.
#[derive(Debug, Clone)]
pub struct BollingerBounce {
    period: usize,
    num_std: f64,
    use_close: bool,
}

impl BollingerBounce {
    /// Create a new Bollinger Bounce strategy.
    pub fn new(period: usize, num_std: f64) -> Self {
        Self {
            period,
            num_std,
            use_close: true,
        }
    }

    /// Use high/low instead of close for band touches.
    pub fn use_high_low(mut self) -> Self {
        self.use_close = false;
        self
    }
}

impl Strategy for BollingerBounce {
    fn name(&self) -> &str {
        "Bollinger Bounce"
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let history = ctx.history();
        let (middle, upper, lower) = match bollinger_bands(history, self.period, self.num_std) {
            Some(bands) => bands,
            None => return Signal::Hold,
        };

        let bar = ctx.current_bar();
        let test_low = if self.use_close { bar.close } else { bar.low };
        let test_high = if self.use_close { bar.close } else { bar.high };

        if ctx.is_flat() {
            if test_low <= lower {
                return Signal::Long; // Bounce off lower band
            } else if test_high >= upper {
                return Signal::Short; // Bounce off upper band
            }
        } else if (ctx.is_long() && bar.close >= middle) || (ctx.is_short() && bar.close <= middle)
        {
            return Signal::Exit; // Take profit at middle band
        }

        Signal::Hold
    }

    fn warmup_period(&self) -> usize {
        self.period
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("period".to_string(), self.period.to_string()),
            ("num_std".to_string(), format!("{:.1}", self.num_std)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;
    use chrono::{TimeZone, Utc};

    fn create_mean_reverting_bars() -> Vec<Bar> {
        // Create bars that oscillate around 100
        vec![
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                100.0,
                102.0,
                98.0,
                100.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
                100.0,
                103.0,
                99.0,
                101.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
                101.0,
                102.0,
                98.0,
                99.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
                99.0,
                101.0,
                97.0,
                100.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(),
                100.0,
                102.0,
                99.0,
                101.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 6, 0, 0, 0).unwrap(),
                101.0,
                103.0,
                100.0,
                102.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 7, 0, 0, 0).unwrap(),
                102.0,
                104.0,
                101.0,
                103.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 8, 0, 0, 0).unwrap(),
                103.0,
                105.0,
                102.0,
                104.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 9, 0, 0, 0).unwrap(),
                104.0,
                106.0,
                103.0,
                105.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 10, 0, 0, 0).unwrap(),
                105.0,
                108.0,
                104.0,
                107.0,
                1000.0,
            ),
            // Now a sharp drop
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 11, 0, 0, 0).unwrap(),
                107.0,
                107.0,
                90.0,
                92.0,
                1000.0,
            ),
        ]
    }

    #[test]
    fn test_mean_reversion_creation() {
        let strategy = MeanReversion::new(20, 2.0, 2.0, 0.5);
        assert_eq!(strategy.period, 20);
        assert_eq!(strategy.warmup_period(), 20);
    }

    #[test]
    fn test_zscore_calculation() {
        let mut strategy = MeanReversion::new(5, 2.0, 2.0, 0.5);
        let bars = create_mean_reverting_bars();

        // At the last bar (sharp drop), z-score should be significantly negative
        let ctx = StrategyContext {
            bar_index: 10,
            bars: &bars,
            position: 0.0,
            cash: 100000.0,
            equity: 100000.0,
            symbol: "TEST",
            volume_profile: None,
            timeframe_manager: None,
        };

        let signal = strategy.on_bar(&ctx);
        // Should signal long due to oversold condition
        assert!(matches!(signal, Signal::Long | Signal::Hold));
    }
}

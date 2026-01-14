//! MACD-based Strategy.
//!
//! Uses the Moving Average Convergence Divergence indicator to identify
//! trend changes and momentum.

use crate::data::macd;
use crate::strategy::{Strategy, StrategyContext};
use crate::types::Signal;

/// MACD Strategy.
///
/// # Parameters
/// - `fast_period`: Fast EMA period (default: 12)
/// - `slow_period`: Slow EMA period (default: 26)
/// - `signal_period`: Signal line EMA period (default: 9)
///
/// # Signals
/// - Long: MACD line crosses above signal line (bullish crossover)
/// - Short: MACD line crosses below signal line (bearish crossover)
/// - Exit: Opposite crossover occurs
#[derive(Debug, Clone)]
pub struct MacdStrategy {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    prev_macd: Option<f64>,
    prev_signal: Option<f64>,
    use_histogram_zero: bool,
}

impl MacdStrategy {
    /// Create a new MACD strategy with custom parameters.
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        assert!(fast_period < slow_period, "Fast period must be less than slow period");
        assert!(signal_period > 0, "Signal period must be positive");

        Self {
            fast_period,
            slow_period,
            signal_period,
            prev_macd: None,
            prev_signal: None,
            use_histogram_zero: false,
        }
    }

    /// Create with default parameters (12/26/9).
    pub fn default_params() -> Self {
        Self::new(12, 26, 9)
    }

    /// Use histogram zero-line crossings instead of MACD/signal crossings.
    pub fn with_histogram_mode(mut self) -> Self {
        self.use_histogram_zero = true;
        self
    }
}

impl Strategy for MacdStrategy {
    fn name(&self) -> &str {
        "MACD Strategy"
    }

    fn init(&mut self) {
        self.prev_macd = None;
        self.prev_signal = None;
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let history = ctx.history();
        let (macd_line, signal_line, histogram) = match macd(history, self.fast_period, self.slow_period, self.signal_period) {
            Some(values) => values,
            None => return Signal::Hold,
        };

        let signal = match (self.prev_macd, self.prev_signal) {
            (Some(prev_macd), Some(prev_signal)) => {
                if self.use_histogram_zero {
                    // Use histogram crossing zero
                    let prev_histogram = prev_macd - prev_signal;

                    if ctx.is_flat() {
                        if prev_histogram <= 0.0 && histogram > 0.0 {
                            Signal::Long
                        } else if prev_histogram >= 0.0 && histogram < 0.0 {
                            Signal::Short
                        } else {
                            Signal::Hold
                        }
                    } else if ctx.is_long() {
                        if histogram < 0.0 {
                            Signal::Exit
                        } else {
                            Signal::Hold
                        }
                    } else {
                        // Short position
                        if histogram > 0.0 {
                            Signal::Exit
                        } else {
                            Signal::Hold
                        }
                    }
                } else {
                    // Use MACD/signal line crossovers
                    let was_below = prev_macd < prev_signal;
                    let is_above = macd_line > signal_line;

                    if ctx.is_flat() {
                        if was_below && is_above {
                            Signal::Long // Bullish crossover
                        } else if !was_below && !is_above {
                            Signal::Short // Bearish crossover
                        } else {
                            Signal::Hold
                        }
                    } else if ctx.is_long() {
                        if !was_below && !is_above {
                            Signal::Exit // Exit on bearish crossover
                        } else {
                            Signal::Hold
                        }
                    } else {
                        // Short position
                        if was_below && is_above {
                            Signal::Exit // Exit on bullish crossover
                        } else {
                            Signal::Hold
                        }
                    }
                }
            }
            _ => Signal::Hold,
        };

        self.prev_macd = Some(macd_line);
        self.prev_signal = Some(signal_line);

        signal
    }

    fn warmup_period(&self) -> usize {
        self.slow_period + self.signal_period
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("fast_period".to_string(), self.fast_period.to_string()),
            ("slow_period".to_string(), self.slow_period.to_string()),
            ("signal_period".to_string(), self.signal_period.to_string()),
            ("mode".to_string(), if self.use_histogram_zero { "histogram".to_string() } else { "crossover".to_string() }),
        ]
    }
}

/// MACD with trend filter.
/// Only takes signals in the direction of the overall trend (based on MACD above/below zero).
#[derive(Debug, Clone)]
pub struct MacdTrendStrategy {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    prev_macd: Option<f64>,
    prev_signal: Option<f64>,
}

impl MacdTrendStrategy {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            signal_period,
            prev_macd: None,
            prev_signal: None,
        }
    }

    pub fn default_params() -> Self {
        Self::new(12, 26, 9)
    }
}

impl Strategy for MacdTrendStrategy {
    fn name(&self) -> &str {
        "MACD Trend"
    }

    fn init(&mut self) {
        self.prev_macd = None;
        self.prev_signal = None;
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let history = ctx.history();
        let (macd_line, signal_line, _histogram) = match macd(history, self.fast_period, self.slow_period, self.signal_period) {
            Some(values) => values,
            None => return Signal::Hold,
        };

        let signal = match (self.prev_macd, self.prev_signal) {
            (Some(prev_macd), Some(prev_signal)) => {
                let was_below = prev_macd < prev_signal;
                let is_above = macd_line > signal_line;
                let is_bullish_trend = macd_line > 0.0;

                if ctx.is_flat() {
                    // Only go long in bullish trend (MACD > 0)
                    if was_below && is_above && is_bullish_trend {
                        Signal::Long
                    }
                    // Only go short in bearish trend (MACD < 0)
                    else if !was_below && !is_above && !is_bullish_trend {
                        Signal::Short
                    } else {
                        Signal::Hold
                    }
                } else if ctx.is_long() {
                    // Exit long on bearish crossover
                    if !was_below && !is_above {
                        Signal::Exit
                    } else {
                        Signal::Hold
                    }
                } else {
                    // Exit short on bullish crossover
                    if was_below && is_above {
                        Signal::Exit
                    } else {
                        Signal::Hold
                    }
                }
            }
            _ => Signal::Hold,
        };

        self.prev_macd = Some(macd_line);
        self.prev_signal = Some(signal_line);

        signal
    }

    fn warmup_period(&self) -> usize {
        self.slow_period + self.signal_period
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("fast_period".to_string(), self.fast_period.to_string()),
            ("slow_period".to_string(), self.slow_period.to_string()),
            ("signal_period".to_string(), self.signal_period.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;
    use chrono::{TimeZone, Utc};

    fn create_trending_bars(count: usize, trend: f64) -> Vec<Bar> {
        (0..count)
            .map(|i| {
                let base = 100.0 + trend * i as f64;
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    base,
                    base + 2.0,
                    base - 1.0,
                    base + 1.0,
                    1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_macd_strategy_creation() {
        let strategy = MacdStrategy::new(12, 26, 9);
        assert_eq!(strategy.fast_period, 12);
        assert_eq!(strategy.slow_period, 26);
        assert_eq!(strategy.signal_period, 9);
    }

    #[test]
    fn test_macd_warmup() {
        let strategy = MacdStrategy::default_params();
        assert_eq!(strategy.warmup_period(), 35); // 26 + 9
    }

    #[test]
    fn test_macd_parameters() {
        let strategy = MacdStrategy::default_params().with_histogram_mode();
        let params = strategy.parameters();
        assert_eq!(params.len(), 4);
        assert_eq!(params[3].1, "histogram");
    }

    #[test]
    fn test_macd_on_trending_data() {
        let mut strategy = MacdStrategy::default_params();
        strategy.init();

        let bars = create_trending_bars(60, 0.5); // Uptrend

        // Run through bars to build up history
        for i in 0..bars.len() {
            let ctx = StrategyContext {
                bar_index: i,
                bars: &bars,
                position: 0.0,
                cash: 100000.0,
                equity: 100000.0,
                symbol: "TEST",
            };
            let _ = strategy.on_bar(&ctx);
        }

        // Should have set prev values
        assert!(strategy.prev_macd.is_some());
        assert!(strategy.prev_signal.is_some());
    }

    #[test]
    #[should_panic]
    fn test_invalid_periods() {
        MacdStrategy::new(26, 12, 9); // fast > slow should panic
    }
}

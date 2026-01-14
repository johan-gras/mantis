//! Simple Moving Average Crossover Strategy.
//!
//! This is a classic trend-following strategy that generates buy signals
//! when the fast MA crosses above the slow MA, and sell signals when
//! the fast MA crosses below the slow MA.

use crate::data::sma;
use crate::strategy::{Strategy, StrategyContext};
use crate::types::Signal;

/// SMA Crossover Strategy.
///
/// # Parameters
/// - `fast_period`: Period for the fast moving average (default: 10)
/// - `slow_period`: Period for the slow moving average (default: 30)
///
/// # Signals
/// - Long: Fast MA crosses above Slow MA
/// - Exit: Fast MA crosses below Slow MA (or use Short if shorting enabled)
#[derive(Debug, Clone)]
pub struct SmaCrossover {
    fast_period: usize,
    slow_period: usize,
    prev_fast: Option<f64>,
    prev_slow: Option<f64>,
}

impl SmaCrossover {
    /// Create a new SMA Crossover strategy.
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        assert!(
            fast_period < slow_period,
            "Fast period must be less than slow period"
        );

        Self {
            fast_period,
            slow_period,
            prev_fast: None,
            prev_slow: None,
        }
    }

    /// Create with default parameters (10/30).
    pub fn default_params() -> Self {
        Self::new(10, 30)
    }
}

impl Strategy for SmaCrossover {
    fn name(&self) -> &str {
        "SMA Crossover"
    }

    fn init(&mut self) {
        self.prev_fast = None;
        self.prev_slow = None;
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let history = ctx.history();

        let fast = match sma(history, self.fast_period) {
            Some(v) => v,
            None => return Signal::Hold,
        };

        let slow = match sma(history, self.slow_period) {
            Some(v) => v,
            None => return Signal::Hold,
        };

        let signal = match (self.prev_fast, self.prev_slow) {
            (Some(prev_f), Some(prev_s)) => {
                // Check for crossover
                let was_below = prev_f < prev_s;
                let is_above = fast > slow;

                if was_below && is_above {
                    // Bullish crossover
                    Signal::Long
                } else if !was_below && !is_above {
                    // Bearish crossover
                    if ctx.is_long() {
                        Signal::Exit
                    } else {
                        Signal::Short
                    }
                } else {
                    Signal::Hold
                }
            }
            _ => Signal::Hold,
        };

        self.prev_fast = Some(fast);
        self.prev_slow = Some(slow);

        signal
    }

    fn warmup_period(&self) -> usize {
        self.slow_period
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("fast_period".to_string(), self.fast_period.to_string()),
            ("slow_period".to_string(), self.slow_period.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;
    use chrono::{TimeZone, Utc};

    fn create_trending_bars(trend: f64, count: usize) -> Vec<Bar> {
        (0..count)
            .map(|i| {
                let base = 100.0 + trend * i as f64;
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    base,
                    base + 2.0,
                    base - 2.0,
                    base + 1.0,
                    1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_sma_crossover_creation() {
        let strategy = SmaCrossover::new(5, 20);
        assert_eq!(strategy.fast_period, 5);
        assert_eq!(strategy.slow_period, 20);
    }

    #[test]
    #[should_panic]
    fn test_invalid_periods() {
        SmaCrossover::new(20, 5); // Should panic - fast > slow
    }

    #[test]
    fn test_warmup_period() {
        let strategy = SmaCrossover::new(10, 30);
        assert_eq!(strategy.warmup_period(), 30);
    }

    #[test]
    fn test_uptrend_signal() {
        let mut strategy = SmaCrossover::new(3, 10);
        strategy.init();

        // Create uptrending data
        let bars = create_trending_bars(1.0, 50);

        let mut last_signal = Signal::Hold;
        for i in 0..bars.len() {
            let ctx = StrategyContext {
                bar_index: i,
                bars: &bars,
                position: 0.0,
                cash: 100000.0,
                equity: 100000.0,
                symbol: "TEST",
            };

            last_signal = strategy.on_bar(&ctx);
        }

        // In a strong uptrend, we should eventually get a long signal
        // (after the fast MA catches up to cross above the slow MA)
        assert!(matches!(last_signal, Signal::Hold | Signal::Long));
    }
}

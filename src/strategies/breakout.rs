//! Breakout Strategy.
//!
//! This strategy trades breakouts from price channels or consolidation ranges.

use crate::strategy::{Strategy, StrategyContext};
use crate::types::Signal;

/// Donchian Channel Breakout Strategy.
///
/// # Parameters
/// - `entry_period`: Period for entry channel (default: 20)
/// - `exit_period`: Period for exit channel (default: 10)
///
/// # Signals
/// - Long: Price breaks above highest high of entry_period bars
/// - Short: Price breaks below lowest low of entry_period bars
/// - Exit: Price breaks exit channel in opposite direction
#[derive(Debug, Clone)]
pub struct BreakoutStrategy {
    entry_period: usize,
    exit_period: usize,
}

impl BreakoutStrategy {
    /// Create a new Breakout strategy.
    pub fn new(entry_period: usize, exit_period: usize) -> Self {
        assert!(entry_period > 0, "Entry period must be positive");
        assert!(exit_period > 0, "Exit period must be positive");

        Self {
            entry_period,
            exit_period,
        }
    }

    /// Create with default parameters (Turtle Trading style: 20/10).
    pub fn default_params() -> Self {
        Self::new(20, 10)
    }

    /// Calculate the highest high over n bars.
    fn highest_high(&self, ctx: &StrategyContext, period: usize) -> Option<f64> {
        if ctx.bar_index < period {
            return None;
        }

        let start = ctx.bar_index - period;
        let highest = ctx.bars[start..ctx.bar_index]
            .iter()
            .map(|b| b.high)
            .fold(f64::NEG_INFINITY, f64::max);

        Some(highest)
    }

    /// Calculate the lowest low over n bars.
    fn lowest_low(&self, ctx: &StrategyContext, period: usize) -> Option<f64> {
        if ctx.bar_index < period {
            return None;
        }

        let start = ctx.bar_index - period;
        let lowest = ctx.bars[start..ctx.bar_index]
            .iter()
            .map(|b| b.low)
            .fold(f64::INFINITY, f64::min);

        Some(lowest)
    }
}

impl Strategy for BreakoutStrategy {
    fn name(&self) -> &str {
        "Donchian Breakout"
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let current = ctx.current_bar();

        if ctx.is_flat() {
            // Entry conditions
            let entry_high = match self.highest_high(ctx, self.entry_period) {
                Some(h) => h,
                None => return Signal::Hold,
            };
            let entry_low = match self.lowest_low(ctx, self.entry_period) {
                Some(l) => l,
                None => return Signal::Hold,
            };

            if current.high > entry_high {
                return Signal::Long; // Breakout above resistance
            } else if current.low < entry_low {
                return Signal::Short; // Breakdown below support
            }
        } else {
            // Exit conditions
            let exit_high = match self.highest_high(ctx, self.exit_period) {
                Some(h) => h,
                None => return Signal::Hold,
            };
            let exit_low = match self.lowest_low(ctx, self.exit_period) {
                Some(l) => l,
                None => return Signal::Hold,
            };

            if ctx.is_long() && current.low < exit_low {
                return Signal::Exit; // Exit long on breakdown
            } else if ctx.is_short() && current.high > exit_high {
                return Signal::Exit; // Exit short on breakout
            }
        }

        Signal::Hold
    }

    fn warmup_period(&self) -> usize {
        self.entry_period.max(self.exit_period)
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("entry_period".to_string(), self.entry_period.to_string()),
            ("exit_period".to_string(), self.exit_period.to_string()),
        ]
    }
}

/// Range Breakout Strategy.
///
/// Waits for a consolidation range and trades the breakout.
#[derive(Debug, Clone)]
pub struct RangeBreakout {
    lookback: usize,
    threshold_pct: f64,
    in_range: bool,
    range_high: f64,
    range_low: f64,
}

impl RangeBreakout {
    /// Create a new Range Breakout strategy.
    pub fn new(lookback: usize, threshold_pct: f64) -> Self {
        Self {
            lookback,
            threshold_pct,
            in_range: false,
            range_high: 0.0,
            range_low: 0.0,
        }
    }

    fn is_consolidating(&self, ctx: &StrategyContext) -> Option<(bool, f64, f64)> {
        if ctx.bar_index < self.lookback {
            return None;
        }

        let start = ctx.bar_index - self.lookback;
        let bars = &ctx.bars[start..=ctx.bar_index];

        let high = bars
            .iter()
            .map(|b| b.high)
            .fold(f64::NEG_INFINITY, f64::max);
        let low = bars.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);

        let range_pct = (high - low) / low * 100.0;
        let is_tight = range_pct < self.threshold_pct;

        Some((is_tight, high, low))
    }
}

impl Strategy for RangeBreakout {
    fn name(&self) -> &str {
        "Range Breakout"
    }

    fn init(&mut self) {
        self.in_range = false;
        self.range_high = 0.0;
        self.range_low = 0.0;
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let current = ctx.current_bar();

        if ctx.is_flat() {
            if let Some((is_tight, high, low)) = self.is_consolidating(ctx) {
                if is_tight {
                    self.in_range = true;
                    self.range_high = high;
                    self.range_low = low;
                } else if self.in_range {
                    // Check for breakout
                    if current.close > self.range_high {
                        self.in_range = false;
                        return Signal::Long;
                    } else if current.close < self.range_low {
                        self.in_range = false;
                        return Signal::Short;
                    }
                }
            }
        }

        Signal::Hold
    }

    fn warmup_period(&self) -> usize {
        self.lookback + 1
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("lookback".to_string(), self.lookback.to_string()),
            (
                "threshold_pct".to_string(),
                format!("{:.1}%", self.threshold_pct),
            ),
        ]
    }
}

/// ATR-based Volatility Breakout.
#[derive(Debug, Clone)]
pub struct AtrBreakout {
    atr_period: usize,
    multiplier: f64,
    prev_close: Option<f64>,
}

impl AtrBreakout {
    pub fn new(atr_period: usize, multiplier: f64) -> Self {
        Self {
            atr_period,
            multiplier,
            prev_close: None,
        }
    }

    fn calculate_atr(&self, ctx: &StrategyContext) -> Option<f64> {
        use crate::data::atr;
        atr(ctx.history(), self.atr_period)
    }
}

impl Strategy for AtrBreakout {
    fn name(&self) -> &str {
        "ATR Breakout"
    }

    fn init(&mut self) {
        self.prev_close = None;
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let current = ctx.current_bar();
        let atr_val = match self.calculate_atr(ctx) {
            Some(a) => a,
            None => return Signal::Hold,
        };

        let signal = match self.prev_close {
            Some(prev) => {
                let threshold = atr_val * self.multiplier;

                if ctx.is_flat() {
                    if current.close > prev + threshold {
                        Signal::Long
                    } else if current.close < prev - threshold {
                        Signal::Short
                    } else {
                        Signal::Hold
                    }
                } else if ctx.is_long() {
                    if current.close < prev - threshold {
                        Signal::Exit
                    } else {
                        Signal::Hold
                    }
                } else if current.close > prev + threshold {
                    Signal::Exit
                } else {
                    Signal::Hold
                }
            }
            None => Signal::Hold,
        };

        self.prev_close = Some(current.close);
        signal
    }

    fn warmup_period(&self) -> usize {
        self.atr_period + 1
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("atr_period".to_string(), self.atr_period.to_string()),
            ("multiplier".to_string(), format!("{:.1}", self.multiplier)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;
    use chrono::{TimeZone, Utc};

    fn create_trending_bars() -> Vec<Bar> {
        (0..30)
            .map(|i| {
                let base = 100.0 + i as f64 * 0.5;
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i),
                    base - 1.0,
                    base + 2.0,
                    base - 2.0,
                    base + 1.0,
                    1000.0,
                )
            })
            .collect()
    }

    fn create_breakout_bars() -> Vec<Bar> {
        let mut bars: Vec<Bar> = (0..20)
            .map(|i| {
                // Consolidation phase
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i),
                    99.0,
                    101.0,
                    99.0,
                    100.0,
                    1000.0,
                )
            })
            .collect();

        // Add breakout bar
        bars.push(Bar::new(
            Utc.with_ymd_and_hms(2024, 1, 21, 0, 0, 0).unwrap(),
            100.0,
            115.0,
            100.0,
            112.0,
            2000.0,
        ));

        bars
    }

    #[test]
    fn test_breakout_strategy_creation() {
        let strategy = BreakoutStrategy::new(20, 10);
        assert_eq!(strategy.entry_period, 20);
        assert_eq!(strategy.exit_period, 10);
    }

    #[test]
    fn test_breakout_warmup() {
        let strategy = BreakoutStrategy::new(20, 10);
        assert_eq!(strategy.warmup_period(), 20);
    }

    #[test]
    fn test_highest_high() {
        let strategy = BreakoutStrategy::new(5, 3);
        let bars = create_trending_bars();

        let ctx = StrategyContext {
            bar_index: 10,
            bars: &bars,
            position: 0.0,
            cash: 100000.0,
            equity: 100000.0,
            symbol: "TEST",
        };

        let high = strategy.highest_high(&ctx, 5);
        assert!(high.is_some());
    }

    #[test]
    fn test_breakout_signal() {
        let mut strategy = BreakoutStrategy::new(10, 5);
        let bars = create_breakout_bars();

        // Check signal on breakout bar
        let ctx = StrategyContext {
            bar_index: 20,
            bars: &bars,
            position: 0.0,
            cash: 100000.0,
            equity: 100000.0,
            symbol: "TEST",
        };

        let signal = strategy.on_bar(&ctx);
        assert!(matches!(signal, Signal::Long));
    }
}

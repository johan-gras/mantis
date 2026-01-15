//! Strategy trait and related utilities.

use crate::types::{Bar, Order, Signal, VolumeProfile};

/// Context provided to strategies during backtest execution.
#[derive(Debug)]
pub struct StrategyContext<'a> {
    /// Current bar index.
    pub bar_index: usize,
    /// Current and historical bars up to this point.
    pub bars: &'a [Bar],
    /// Current position quantity (positive for long, negative for short, zero for flat).
    pub position: f64,
    /// Current cash balance.
    pub cash: f64,
    /// Current portfolio equity (cash + positions value).
    pub equity: f64,
    /// Symbol being traded.
    pub symbol: &'a str,
    /// Historical volume statistics for the symbol when available.
    pub volume_profile: Option<VolumeProfile>,
}

impl<'a> StrategyContext<'a> {
    /// Get the current bar.
    pub fn current_bar(&self) -> &Bar {
        &self.bars[self.bar_index]
    }

    /// Get the previous bar, if available.
    pub fn prev_bar(&self) -> Option<&Bar> {
        if self.bar_index > 0 {
            Some(&self.bars[self.bar_index - 1])
        } else {
            None
        }
    }

    /// Get a bar at a specific lookback (0 = current, 1 = previous, etc.).
    pub fn bar_at(&self, lookback: usize) -> Option<&Bar> {
        if lookback <= self.bar_index {
            Some(&self.bars[self.bar_index - lookback])
        } else {
            None
        }
    }

    /// Get the closing prices for the last n bars.
    pub fn closes(&self, n: usize) -> Vec<f64> {
        let start = self.bar_index.saturating_sub(n - 1);
        self.bars[start..=self.bar_index]
            .iter()
            .map(|b| b.close)
            .collect()
    }

    /// Get all bars up to and including current.
    pub fn history(&self) -> &[Bar] {
        &self.bars[..=self.bar_index]
    }

    /// Check if we have a position.
    pub fn has_position(&self) -> bool {
        self.position.abs() > f64::EPSILON
    }

    /// Check if we're long.
    pub fn is_long(&self) -> bool {
        self.position > f64::EPSILON
    }

    /// Check if we're short.
    pub fn is_short(&self) -> bool {
        self.position < -f64::EPSILON
    }

    /// Check if we're flat (no position).
    pub fn is_flat(&self) -> bool {
        !self.has_position()
    }

    /// Get the configured volume profile for this symbol, if available.
    pub fn volume_profile(&self) -> Option<VolumeProfile> {
        self.volume_profile
    }
}

/// Trait that all trading strategies must implement.
pub trait Strategy: Send + Sync {
    /// Returns the name of the strategy.
    fn name(&self) -> &str;

    /// Called once at the start of the backtest.
    fn init(&mut self) {}

    /// Generate a trading signal based on current market state.
    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal;

    /// Optionally generate specific orders instead of signals.
    /// If this returns Some(orders), the signals from on_bar are ignored.
    fn generate_orders(&mut self, _ctx: &StrategyContext) -> Option<Vec<Order>> {
        None
    }

    /// Called after each trade is executed.
    fn on_trade(&mut self, _ctx: &StrategyContext, _order: &Order) {}

    /// Called at the end of the backtest.
    fn on_finish(&mut self) {}

    /// Minimum bars needed before the strategy can generate signals.
    fn warmup_period(&self) -> usize {
        0
    }

    /// Get strategy parameters as key-value pairs for logging.
    fn parameters(&self) -> Vec<(String, String)> {
        vec![]
    }
}

/// A simple wrapper to run multiple strategies on the same data.
pub struct StrategyEnsemble {
    strategies: Vec<Box<dyn Strategy>>,
}

impl StrategyEnsemble {
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    pub fn add(&mut self, strategy: Box<dyn Strategy>) {
        self.strategies.push(strategy);
    }

    pub fn strategies(&self) -> &[Box<dyn Strategy>] {
        &self.strategies
    }

    pub fn strategies_mut(&mut self) -> &mut [Box<dyn Strategy>] {
        &mut self.strategies
    }
}

impl Default for StrategyEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    struct TestStrategy {
        signal_count: usize,
    }

    impl Strategy for TestStrategy {
        fn name(&self) -> &str {
            "TestStrategy"
        }

        fn on_bar(&mut self, _ctx: &StrategyContext) -> Signal {
            self.signal_count += 1;
            Signal::Hold
        }

        fn warmup_period(&self) -> usize {
            5
        }
    }

    fn create_test_bars() -> Vec<Bar> {
        (0..10)
            .map(|i| {
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, i + 1, 0, 0, 0).unwrap(),
                    100.0 + i as f64,
                    105.0 + i as f64,
                    98.0 + i as f64,
                    102.0 + i as f64,
                    1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_strategy_context() {
        let bars = create_test_bars();
        let ctx = StrategyContext {
            bar_index: 5,
            bars: &bars,
            position: 100.0,
            cash: 10000.0,
            equity: 20000.0,
            symbol: "TEST",
            volume_profile: None,
        };

        assert!(ctx.has_position());
        assert!(ctx.is_long());
        assert!(!ctx.is_short());
        assert!(!ctx.is_flat());

        assert_eq!(ctx.current_bar().open, 105.0);
        assert_eq!(ctx.prev_bar().unwrap().open, 104.0);
        assert_eq!(ctx.bar_at(2).unwrap().open, 103.0);

        let closes = ctx.closes(3);
        assert_eq!(closes.len(), 3);
    }

    #[test]
    fn test_strategy_trait() {
        let mut strategy = TestStrategy { signal_count: 0 };
        assert_eq!(strategy.name(), "TestStrategy");
        assert_eq!(strategy.warmup_period(), 5);

        let bars = create_test_bars();
        let ctx = StrategyContext {
            bar_index: 5,
            bars: &bars,
            position: 0.0,
            cash: 10000.0,
            equity: 10000.0,
            symbol: "TEST",
            volume_profile: None,
        };

        let signal = strategy.on_bar(&ctx);
        assert_eq!(signal, Signal::Hold);
        assert_eq!(strategy.signal_count, 1);
    }
}

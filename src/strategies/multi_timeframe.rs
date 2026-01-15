//! Multi-timeframe trend following strategy.
//!
//! Uses daily SMA for trend direction and hourly RSI for entry timing.
//! Demonstrates multi-timeframe strategy interface.

use crate::data::{rsi, sma, ResampleInterval};
use crate::strategy::{Strategy, StrategyContext};
use crate::types::Signal;

/// Multi-timeframe strategy: Daily trend + Hourly entries
///
/// # Strategy Logic
/// - Uses daily 50-period SMA for trend direction
/// - Uses hourly 14-period RSI for entry timing
/// - Long when: price > daily SMA AND hourly RSI < 30 (oversold)
/// - Short when: price < daily SMA AND hourly RSI > 70 (overbought)
///
/// # Example
/// ```ignore
/// use mantis::strategies::MultiTimeframeStrategy;
/// use mantis::engine::{Engine, BacktestConfig};
/// use mantis::data::load_csv;
///
/// let mut engine = Engine::new(BacktestConfig::default());
/// // Load minute-level data
/// let bars = load_csv("data/SPY_1min.csv", &Default::default())?;
/// engine.add_data("SPY", bars);
///
/// let mut strategy = MultiTimeframeStrategy::new(50, 14, 30.0, 70.0);
/// let result = engine.run(&mut strategy, "SPY")?;
/// println!("Return: {:.2}%", result.total_return_pct);
/// ```
pub struct MultiTimeframeStrategy {
    /// Daily SMA period for trend
    daily_sma_period: usize,
    /// Hourly RSI period for entries
    hourly_rsi_period: usize,
    /// RSI oversold threshold (buy signal)
    rsi_oversold: f64,
    /// RSI overbought threshold (sell signal)
    rsi_overbought: f64,
}

impl MultiTimeframeStrategy {
    /// Create a new multi-timeframe strategy.
    ///
    /// # Arguments
    /// * `daily_sma_period` - SMA period for daily trend (e.g., 50)
    /// * `hourly_rsi_period` - RSI period for hourly timing (e.g., 14)
    /// * `rsi_oversold` - RSI threshold for oversold (e.g., 30)
    /// * `rsi_overbought` - RSI threshold for overbought (e.g., 70)
    pub fn new(
        daily_sma_period: usize,
        hourly_rsi_period: usize,
        rsi_oversold: f64,
        rsi_overbought: f64,
    ) -> Self {
        Self {
            daily_sma_period,
            hourly_rsi_period,
            rsi_oversold,
            rsi_overbought,
        }
    }

    /// Create default parameters (50-day SMA, 14-hour RSI, 30/70 levels).
    pub fn default_params() -> Self {
        Self::new(50, 14, 30.0, 70.0)
    }
}

impl Strategy for MultiTimeframeStrategy {
    fn name(&self) -> &str {
        "Multi-Timeframe (Daily Trend + Hourly Entry)"
    }

    fn requested_timeframes(&self) -> Vec<ResampleInterval> {
        vec![
            ResampleInterval::Hour(1),  // For hourly RSI
            ResampleInterval::Day,       // For daily SMA
        ]
    }

    fn warmup_period(&self) -> usize {
        // Need enough data for daily SMA calculation
        // Assuming base data is minute-level: 50 days * 390 minutes/day
        self.daily_sma_period * 390
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        // Check if multi-timeframe is available
        if !ctx.has_multi_timeframe() {
            // Fallback: use base timeframe only (not recommended)
            return Signal::Hold;
        }

        // Get daily bars for trend
        let daily_bars = match ctx.bars_at(ResampleInterval::Day) {
            Some(bars) if bars.len() >= self.daily_sma_period => bars,
            _ => return Signal::Hold, // Not enough daily bars yet
        };

        // Calculate daily SMA
        let daily_sma_value = match sma(daily_bars, self.daily_sma_period) {
            Some(value) => value,
            None => return Signal::Hold,
        };

        // Get hourly bars for timing
        let hourly_bars = match ctx.bars_at(ResampleInterval::Hour(1)) {
            Some(bars) if bars.len() > self.hourly_rsi_period => bars,
            _ => return Signal::Hold, // Not enough hourly bars yet
        };

        // Calculate hourly RSI
        let hourly_rsi_value = match rsi(hourly_bars, self.hourly_rsi_period) {
            Some(value) => value,
            None => return Signal::Hold,
        };

        // Current price
        let current_price = ctx.current_bar().close;

        // Determine trend from daily SMA
        let uptrend = current_price > daily_sma_value;
        let downtrend = current_price < daily_sma_value;

        // Entry signals based on trend + RSI
        if uptrend && hourly_rsi_value < self.rsi_oversold && !ctx.has_position() {
            // Uptrend + oversold RSI → Buy
            Signal::Long
        } else if downtrend && hourly_rsi_value > self.rsi_overbought && !ctx.has_position() {
            // Downtrend + overbought RSI → Short
            Signal::Short
        } else if ctx.has_position() {
            // Exit logic: opposite trend
            if (ctx.is_long() && downtrend) || (ctx.is_short() && uptrend) {
                Signal::Exit
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        }
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("daily_sma_period".to_string(), self.daily_sma_period.to_string()),
            ("hourly_rsi_period".to_string(), self.hourly_rsi_period.to_string()),
            ("rsi_oversold".to_string(), format!("{:.1}", self.rsi_oversold)),
            ("rsi_overbought".to_string(), format!("{:.1}", self.rsi_overbought)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;
    use chrono::{TimeZone, Utc};

    fn create_minute_bars(count: usize, start_price: f64) -> Vec<Bar> {
        let base_time = Utc.with_ymd_and_hms(2024, 1, 1, 9, 30, 0).unwrap();
        (0..count)
            .map(|i| {
                let price = start_price + (i as f64 * 0.01); // Slight uptrend
                Bar {
                    timestamp: base_time + chrono::Duration::minutes(i as i64),
                    open: price,
                    high: price + 0.05,
                    low: price - 0.05,
                    close: price,
                    volume: 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_strategy_creation() {
        let strategy = MultiTimeframeStrategy::new(50, 14, 30.0, 70.0);
        assert_eq!(strategy.name(), "Multi-Timeframe (Daily Trend + Hourly Entry)");
        assert_eq!(strategy.warmup_period(), 50 * 390);
    }

    #[test]
    fn test_requested_timeframes() {
        let strategy = MultiTimeframeStrategy::default_params();
        let timeframes = strategy.requested_timeframes();
        assert_eq!(timeframes.len(), 2);
        assert!(timeframes.contains(&ResampleInterval::Hour(1)));
        assert!(timeframes.contains(&ResampleInterval::Day));
    }

    #[test]
    fn test_signal_without_multi_timeframe() {
        let strategy = MultiTimeframeStrategy::default_params();
        let bars = create_minute_bars(100, 100.0);

        let ctx = StrategyContext {
            bar_index: 50,
            bars: &bars,
            position: 0.0,
            cash: 10000.0,
            equity: 10000.0,
            symbol: "TEST",
            volume_profile: None,
            timeframe_manager: None, // No multi-timeframe
        };

        let mut strategy = strategy;
        let signal = strategy.on_bar(&ctx);
        assert_eq!(signal, Signal::Hold); // Should hold without multi-timeframe
    }

    #[test]
    fn test_parameters() {
        let strategy = MultiTimeframeStrategy::new(50, 14, 30.0, 70.0);
        let params = strategy.parameters();
        assert_eq!(params.len(), 4);
        assert!(params.iter().any(|(k, v)| k == "daily_sma_period" && v == "50"));
        assert!(params.iter().any(|(k, v)| k == "hourly_rsi_period" && v == "14"));
    }
}

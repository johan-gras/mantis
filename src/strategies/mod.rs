//! Example trading strategies.
//!
//! This module provides several ready-to-use trading strategies:
//!
//! - [`SmaCrossover`]: Simple Moving Average crossover
//! - [`MomentumStrategy`]: Price momentum / rate of change
//! - [`MeanReversion`]: Bollinger Band mean reversion
//! - [`RsiStrategy`]: RSI overbought/oversold
//! - [`BreakoutStrategy`]: Donchian channel breakout
//! - [`MacdStrategy`]: MACD crossover strategy
//!
//! Additional strategies for advanced use:
//! - [`RocStrategy`]: Rate of Change variant
//! - [`BollingerBounce`]: Bollinger Band bounce
//! - [`RsiDivergence`]: RSI divergence detection
//! - [`RangeBreakout`]: Consolidation range breakout
//! - [`AtrBreakout`]: ATR-based volatility breakout
//! - [`MacdTrendStrategy`]: MACD with trend filter
//!
//! Machine Learning Strategies:
//! - [`ExternalSignalStrategy`]: Accept signals from ML model predictions
//! - [`ClassificationStrategy`]: Use discrete class predictions (long/hold/short)
//! - [`ConfidenceWeightedStrategy`]: Position sizing based on model confidence
//! - [`TimestampedSignalStrategy`]: Signals indexed by timestamp
//! - [`EnsembleSignalStrategy`]: Combine multiple model predictions
//!
//! Multi-Timeframe Strategies:
//! - [`MultiTimeframeStrategy`]: Daily trend + hourly entry timing

mod breakout;
mod macd_strategy;
mod mean_reversion;
mod ml_strategy;
mod momentum;
mod multi_timeframe;
mod rsi_strategy;
mod sma_crossover;

pub use breakout::{AtrBreakout, BreakoutStrategy, RangeBreakout};
pub use macd_strategy::{MacdStrategy, MacdTrendStrategy};
pub use mean_reversion::{BollingerBounce, MeanReversion};
pub use ml_strategy::{
    AggregationMethod, ClassificationStrategy, ConfidenceWeightedStrategy, EnsembleSignalStrategy,
    ExternalSignalStrategy, TimestampedSignalStrategy,
};
pub use momentum::{MomentumStrategy, RocStrategy};
pub use multi_timeframe::MultiTimeframeStrategy;
pub use rsi_strategy::{RsiDivergence, RsiStrategy};
pub use sma_crossover::SmaCrossover;

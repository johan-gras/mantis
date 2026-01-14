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

mod sma_crossover;
mod momentum;
mod mean_reversion;
mod rsi_strategy;
mod breakout;
mod macd_strategy;
mod ml_strategy;

pub use sma_crossover::SmaCrossover;
pub use momentum::{MomentumStrategy, RocStrategy};
pub use mean_reversion::{MeanReversion, BollingerBounce};
pub use rsi_strategy::{RsiStrategy, RsiDivergence};
pub use breakout::{BreakoutStrategy, RangeBreakout, AtrBreakout};
pub use macd_strategy::{MacdStrategy, MacdTrendStrategy};
pub use ml_strategy::{
    ExternalSignalStrategy,
    ClassificationStrategy,
    ConfidenceWeightedStrategy,
    TimestampedSignalStrategy,
    EnsembleSignalStrategy,
    AggregationMethod,
};

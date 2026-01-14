//! Machine Learning Strategy.
//!
//! This module provides strategies that can accept external signals from
//! machine learning models, enabling seamless integration with deep learning
//! frameworks like PyTorch, TensorFlow, or JAX.
//!
//! # Usage Patterns
//!
//! ## Pattern 1: Pre-computed signals
//!
//! Load predictions from a file or array:
//!
//! ```ignore
//! use ralph_backtest::strategies::ExternalSignalStrategy;
//!
//! let predictions = vec![0.0, 0.5, -0.3, 0.8, -0.2]; // From your ML model
//! let strategy = ExternalSignalStrategy::new(predictions, 0.5);
//! ```
//!
//! ## Pattern 2: Threshold-based entry/exit
//!
//! ```ignore
//! let strategy = ExternalSignalStrategy::new(predictions, 0.5)
//!     .with_exit_threshold(0.0);
//! ```
//!
//! ## Pattern 3: Classification signals
//!
//! ```ignore
//! let class_predictions = vec![1, 0, -1, 1, 0]; // 1=long, 0=hold, -1=short
//! let strategy = ClassificationStrategy::new(class_predictions);
//! ```

use crate::strategy::{Strategy, StrategyContext};
use crate::types::Signal;
use std::collections::HashMap;

/// Strategy that uses pre-computed external signals.
///
/// Ideal for backtesting ML model predictions where the model
/// outputs continuous values (e.g., predicted returns, confidence scores).
#[derive(Debug)]
pub struct ExternalSignalStrategy {
    /// Pre-computed signals indexed by bar index.
    signals: Vec<f64>,
    /// Threshold for long entry (signal > threshold).
    long_threshold: f64,
    /// Threshold for short entry (signal < -threshold).
    short_threshold: f64,
    /// Optional exit threshold (exit when |signal| < exit_threshold).
    exit_threshold: Option<f64>,
    /// Current signal index offset (if signals don't start at bar 0).
    offset: usize,
    /// Name of the strategy.
    name: String,
}

impl ExternalSignalStrategy {
    /// Create a new external signal strategy.
    ///
    /// # Arguments
    /// * `signals` - Pre-computed signal values (e.g., predicted returns)
    /// * `threshold` - Entry threshold (long when signal > threshold, short when < -threshold)
    pub fn new(signals: Vec<f64>, threshold: f64) -> Self {
        Self {
            signals,
            long_threshold: threshold,
            short_threshold: -threshold,
            exit_threshold: None,
            offset: 0,
            name: "External Signal".to_string(),
        }
    }

    /// Set asymmetric thresholds for long and short entries.
    pub fn with_asymmetric_thresholds(mut self, long_threshold: f64, short_threshold: f64) -> Self {
        self.long_threshold = long_threshold;
        self.short_threshold = short_threshold;
        self
    }

    /// Set exit threshold (exit position when |signal| falls below this).
    pub fn with_exit_threshold(mut self, threshold: f64) -> Self {
        self.exit_threshold = Some(threshold);
        self
    }

    /// Set the bar index offset for signals.
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Set a custom name for the strategy.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the signal for a given bar index.
    fn get_signal(&self, bar_index: usize) -> Option<f64> {
        if bar_index < self.offset {
            return None;
        }
        let signal_idx = bar_index - self.offset;
        self.signals.get(signal_idx).copied()
    }
}

impl Strategy for ExternalSignalStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let signal_value = match self.get_signal(ctx.bar_index) {
            Some(s) => s,
            None => return Signal::Hold,
        };

        // Check exit first if we have a position
        if let Some(exit_thresh) = self.exit_threshold {
            if ctx.has_position() && signal_value.abs() < exit_thresh {
                return Signal::Exit;
            }
        }

        // Entry signals
        if ctx.is_flat() {
            if signal_value > self.long_threshold {
                return Signal::Long;
            } else if signal_value < self.short_threshold {
                return Signal::Short;
            }
        } else if ctx.is_long() {
            // Exit long if signal turns bearish
            if signal_value < self.short_threshold {
                return Signal::Exit;
            }
        } else if ctx.is_short() {
            // Exit short if signal turns bullish
            if signal_value > self.long_threshold {
                return Signal::Exit;
            }
        }

        Signal::Hold
    }

    fn warmup_period(&self) -> usize {
        self.offset
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("long_threshold".to_string(), format!("{:.4}", self.long_threshold)),
            ("short_threshold".to_string(), format!("{:.4}", self.short_threshold)),
            ("signals_count".to_string(), self.signals.len().to_string()),
        ]
    }
}

/// Strategy for classification model outputs.
///
/// Expects discrete class predictions: 1 (long), 0 (hold), -1 (short).
#[derive(Debug)]
pub struct ClassificationStrategy {
    /// Pre-computed class predictions.
    predictions: Vec<i8>,
    /// Bar index offset.
    offset: usize,
    /// Whether to allow position reversals without explicit exit.
    allow_reversals: bool,
    /// Name of the strategy.
    name: String,
}

impl ClassificationStrategy {
    /// Create a new classification strategy.
    ///
    /// # Arguments
    /// * `predictions` - Class predictions (1=long, 0=hold/exit, -1=short)
    pub fn new(predictions: Vec<i8>) -> Self {
        Self {
            predictions,
            offset: 0,
            allow_reversals: false,
            name: "Classification Model".to_string(),
        }
    }

    /// Set the bar index offset.
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Allow direct position reversals (long to short without exit).
    pub fn with_reversals(mut self) -> Self {
        self.allow_reversals = true;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the prediction for a given bar index.
    fn get_prediction(&self, bar_index: usize) -> Option<i8> {
        if bar_index < self.offset {
            return None;
        }
        let pred_idx = bar_index - self.offset;
        self.predictions.get(pred_idx).copied()
    }
}

impl Strategy for ClassificationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let pred = match self.get_prediction(ctx.bar_index) {
            Some(p) => p,
            None => return Signal::Hold,
        };

        match pred {
            1 => {
                // Long signal
                if ctx.is_flat() {
                    Signal::Long
                } else if ctx.is_short() && self.allow_reversals {
                    Signal::Long // Will close short and go long
                } else {
                    Signal::Hold
                }
            }
            -1 => {
                // Short signal
                if ctx.is_flat() {
                    Signal::Short
                } else if ctx.is_long() && self.allow_reversals {
                    Signal::Short // Will close long and go short
                } else {
                    Signal::Hold
                }
            }
            0 => {
                // Hold/Exit signal
                if ctx.has_position() {
                    Signal::Exit
                } else {
                    Signal::Hold
                }
            }
            _ => Signal::Hold,
        }
    }

    fn warmup_period(&self) -> usize {
        self.offset
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("predictions_count".to_string(), self.predictions.len().to_string()),
            ("allow_reversals".to_string(), self.allow_reversals.to_string()),
        ]
    }
}

/// Strategy that combines model predictions with risk management.
///
/// Provides position sizing based on prediction confidence.
#[derive(Debug)]
pub struct ConfidenceWeightedStrategy {
    /// Pre-computed signals (expected returns or scores).
    signals: Vec<f64>,
    /// Confidence values for each signal.
    confidences: Vec<f64>,
    /// Entry threshold for signal.
    signal_threshold: f64,
    /// Minimum confidence to trade.
    confidence_threshold: f64,
    /// Bar index offset.
    offset: usize,
    /// Current recommended position size multiplier.
    position_size_multiplier: f64,
}

impl ConfidenceWeightedStrategy {
    /// Create a new confidence-weighted strategy.
    ///
    /// # Arguments
    /// * `signals` - Predicted signals (e.g., expected returns)
    /// * `confidences` - Confidence levels for each signal (0.0 to 1.0)
    /// * `signal_threshold` - Signal threshold for entry
    /// * `confidence_threshold` - Minimum confidence to trade
    pub fn new(
        signals: Vec<f64>,
        confidences: Vec<f64>,
        signal_threshold: f64,
        confidence_threshold: f64,
    ) -> Self {
        assert_eq!(signals.len(), confidences.len(), "Signals and confidences must have same length");

        Self {
            signals,
            confidences,
            signal_threshold,
            confidence_threshold,
            offset: 0,
            position_size_multiplier: 1.0,
        }
    }

    /// Set the bar index offset.
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Get the recommended position size multiplier.
    ///
    /// Call this after on_bar() to get the confidence-weighted position size.
    pub fn position_size_multiplier(&self) -> f64 {
        self.position_size_multiplier
    }

    /// Get signal and confidence for a given bar index.
    fn get_signal_and_confidence(&self, bar_index: usize) -> Option<(f64, f64)> {
        if bar_index < self.offset {
            return None;
        }
        let idx = bar_index - self.offset;
        match (self.signals.get(idx), self.confidences.get(idx)) {
            (Some(&s), Some(&c)) => Some((s, c)),
            _ => None,
        }
    }
}

impl Strategy for ConfidenceWeightedStrategy {
    fn name(&self) -> &str {
        "Confidence Weighted"
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let (signal, confidence) = match self.get_signal_and_confidence(ctx.bar_index) {
            Some(sc) => sc,
            None => return Signal::Hold,
        };

        // Update position size multiplier based on confidence
        self.position_size_multiplier = confidence;

        // Skip low confidence signals
        if confidence < self.confidence_threshold {
            if ctx.has_position() {
                return Signal::Exit;
            }
            return Signal::Hold;
        }

        // Entry/exit logic
        if ctx.is_flat() {
            if signal > self.signal_threshold {
                return Signal::Long;
            } else if signal < -self.signal_threshold {
                return Signal::Short;
            }
        } else if ctx.is_long() {
            if signal < -self.signal_threshold || confidence < self.confidence_threshold {
                return Signal::Exit;
            }
        } else if ctx.is_short() {
            if signal > self.signal_threshold || confidence < self.confidence_threshold {
                return Signal::Exit;
            }
        }

        Signal::Hold
    }

    fn warmup_period(&self) -> usize {
        self.offset
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("signal_threshold".to_string(), format!("{:.4}", self.signal_threshold)),
            ("confidence_threshold".to_string(), format!("{:.4}", self.confidence_threshold)),
            ("signals_count".to_string(), self.signals.len().to_string()),
        ]
    }
}

/// Strategy that uses timestamped signals from an external source.
///
/// Useful when signals are provided as (timestamp, value) pairs.
#[derive(Debug)]
pub struct TimestampedSignalStrategy {
    /// Signals indexed by Unix timestamp.
    signals: HashMap<i64, f64>,
    /// Entry threshold.
    threshold: f64,
    /// Exit threshold.
    exit_threshold: Option<f64>,
    /// Last known signal (for forward fill).
    last_signal: Option<f64>,
    /// Whether to forward fill missing signals.
    forward_fill: bool,
}

impl TimestampedSignalStrategy {
    /// Create a new timestamped signal strategy.
    ///
    /// # Arguments
    /// * `timestamps` - Unix timestamps for signals
    /// * `values` - Signal values corresponding to timestamps
    /// * `threshold` - Entry threshold
    pub fn new(timestamps: Vec<i64>, values: Vec<f64>, threshold: f64) -> Self {
        assert_eq!(timestamps.len(), values.len());

        let signals: HashMap<i64, f64> = timestamps
            .into_iter()
            .zip(values.into_iter())
            .collect();

        Self {
            signals,
            threshold,
            exit_threshold: None,
            last_signal: None,
            forward_fill: true,
        }
    }

    /// Set exit threshold.
    pub fn with_exit_threshold(mut self, threshold: f64) -> Self {
        self.exit_threshold = Some(threshold);
        self
    }

    /// Disable forward fill of missing signals.
    pub fn without_forward_fill(mut self) -> Self {
        self.forward_fill = false;
        self
    }

    /// Get signal for a given bar.
    fn get_signal(&mut self, timestamp: i64) -> Option<f64> {
        if let Some(&signal) = self.signals.get(&timestamp) {
            self.last_signal = Some(signal);
            return Some(signal);
        }

        if self.forward_fill {
            self.last_signal
        } else {
            None
        }
    }
}

impl Strategy for TimestampedSignalStrategy {
    fn name(&self) -> &str {
        "Timestamped Signal"
    }

    fn init(&mut self) {
        self.last_signal = None;
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        let timestamp = ctx.current_bar().timestamp.timestamp();
        let signal_value = match self.get_signal(timestamp) {
            Some(s) => s,
            None => return Signal::Hold,
        };

        // Check exit first
        if let Some(exit_thresh) = self.exit_threshold {
            if ctx.has_position() && signal_value.abs() < exit_thresh {
                return Signal::Exit;
            }
        }

        // Entry signals
        if ctx.is_flat() {
            if signal_value > self.threshold {
                return Signal::Long;
            } else if signal_value < -self.threshold {
                return Signal::Short;
            }
        } else if ctx.is_long() && signal_value < -self.threshold {
            return Signal::Exit;
        } else if ctx.is_short() && signal_value > self.threshold {
            return Signal::Exit;
        }

        Signal::Hold
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("threshold".to_string(), format!("{:.4}", self.threshold)),
            ("signals_count".to_string(), self.signals.len().to_string()),
            ("forward_fill".to_string(), self.forward_fill.to_string()),
        ]
    }
}

/// Builder for creating ensemble strategies from multiple ML models.
#[derive(Debug)]
pub struct EnsembleSignalStrategy {
    /// Individual model signals.
    model_signals: Vec<Vec<f64>>,
    /// Weights for each model.
    weights: Vec<f64>,
    /// Aggregation method.
    aggregation: AggregationMethod,
    /// Entry threshold for aggregated signal.
    threshold: f64,
    /// Offset for signals.
    offset: usize,
}

/// Methods for aggregating multiple model predictions.
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    /// Simple average.
    Mean,
    /// Weighted average using model weights.
    WeightedMean,
    /// Median value.
    Median,
    /// Vote-based (majority sign).
    Vote,
}

impl EnsembleSignalStrategy {
    /// Create a new ensemble strategy.
    pub fn new(model_signals: Vec<Vec<f64>>, threshold: f64) -> Self {
        let n_models = model_signals.len();
        let weights = vec![1.0 / n_models as f64; n_models];

        Self {
            model_signals,
            weights,
            aggregation: AggregationMethod::Mean,
            threshold,
            offset: 0,
        }
    }

    /// Set custom weights for models.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        assert_eq!(weights.len(), self.model_signals.len());
        self.weights = weights;
        self.aggregation = AggregationMethod::WeightedMean;
        self
    }

    /// Set aggregation method.
    pub fn with_aggregation(mut self, method: AggregationMethod) -> Self {
        self.aggregation = method;
        self
    }

    /// Set offset.
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Aggregate signals at a given index.
    fn aggregate_signal(&self, idx: usize) -> Option<f64> {
        let signals: Vec<f64> = self.model_signals
            .iter()
            .filter_map(|s| s.get(idx).copied())
            .collect();

        if signals.is_empty() {
            return None;
        }

        match self.aggregation {
            AggregationMethod::Mean => {
                Some(signals.iter().sum::<f64>() / signals.len() as f64)
            }
            AggregationMethod::WeightedMean => {
                let weighted_sum: f64 = signals.iter()
                    .zip(self.weights.iter())
                    .map(|(s, w)| s * w)
                    .sum();
                let weight_sum: f64 = self.weights.iter().sum();
                Some(weighted_sum / weight_sum)
            }
            AggregationMethod::Median => {
                let mut sorted = signals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    Some((sorted[mid - 1] + sorted[mid]) / 2.0)
                } else {
                    Some(sorted[mid])
                }
            }
            AggregationMethod::Vote => {
                let positive = signals.iter().filter(|&&s| s > 0.0).count();
                let negative = signals.iter().filter(|&&s| s < 0.0).count();
                if positive > negative {
                    Some(positive as f64 / signals.len() as f64)
                } else if negative > positive {
                    Some(-(negative as f64 / signals.len() as f64))
                } else {
                    Some(0.0)
                }
            }
        }
    }
}

impl Strategy for EnsembleSignalStrategy {
    fn name(&self) -> &str {
        "Ensemble Model"
    }

    fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
        if ctx.bar_index < self.offset {
            return Signal::Hold;
        }

        let idx = ctx.bar_index - self.offset;
        let signal = match self.aggregate_signal(idx) {
            Some(s) => s,
            None => return Signal::Hold,
        };

        if ctx.is_flat() {
            if signal > self.threshold {
                return Signal::Long;
            } else if signal < -self.threshold {
                return Signal::Short;
            }
        } else if ctx.is_long() && signal < -self.threshold {
            return Signal::Exit;
        } else if ctx.is_short() && signal > self.threshold {
            return Signal::Exit;
        }

        Signal::Hold
    }

    fn warmup_period(&self) -> usize {
        self.offset
    }

    fn parameters(&self) -> Vec<(String, String)> {
        vec![
            ("n_models".to_string(), self.model_signals.len().to_string()),
            ("aggregation".to_string(), format!("{:?}", self.aggregation)),
            ("threshold".to_string(), format!("{:.4}", self.threshold)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;
    use chrono::{TimeZone, Utc};

    fn create_test_bars(count: usize) -> Vec<Bar> {
        (0..count)
            .map(|i| {
                let base = 100.0 + i as f64;
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    base - 1.0,
                    base + 2.0,
                    base - 2.0,
                    base + 0.5,
                    1000.0,
                )
            })
            .collect()
    }

    fn create_context<'a>(bars: &'a [Bar], idx: usize, position: f64) -> StrategyContext<'a> {
        StrategyContext {
            bar_index: idx,
            bars,
            position,
            cash: 100000.0,
            equity: 100000.0,
            symbol: "TEST",
        }
    }

    #[test]
    fn test_external_signal_strategy() {
        let signals = vec![0.0, 0.3, 0.6, -0.6, 0.1];
        let mut strategy = ExternalSignalStrategy::new(signals, 0.5);

        let bars = create_test_bars(5);

        // First bar - signal 0.0 < 0.5, should hold
        let ctx = create_context(&bars, 0, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Hold);

        // Third bar - signal 0.6 > 0.5, should go long
        let ctx = create_context(&bars, 2, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Long);

        // Fourth bar - signal -0.6 < -0.5, should exit when in long position
        let ctx = create_context(&bars, 3, 100.0); // In long position
        assert_eq!(strategy.on_bar(&ctx), Signal::Exit);
    }

    #[test]
    fn test_external_signal_with_exit_threshold() {
        let signals = vec![0.6, 0.3, 0.1];
        let mut strategy = ExternalSignalStrategy::new(signals, 0.5)
            .with_exit_threshold(0.2);

        let bars = create_test_bars(3);

        // First bar - go long
        let ctx = create_context(&bars, 0, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Long);

        // Third bar - signal 0.1 < 0.2 exit threshold, should exit
        let ctx = create_context(&bars, 2, 100.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Exit);
    }

    #[test]
    fn test_classification_strategy() {
        let predictions = vec![0, 1, 1, -1, 0];
        let mut strategy = ClassificationStrategy::new(predictions);

        let bars = create_test_bars(5);

        // First bar - 0 = hold
        let ctx = create_context(&bars, 0, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Hold);

        // Second bar - 1 = long
        let ctx = create_context(&bars, 1, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Long);

        // Fourth bar - -1 = short (when flat)
        let ctx = create_context(&bars, 3, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Short);

        // Fifth bar - 0 = exit (when in position)
        let ctx = create_context(&bars, 4, 100.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Exit);
    }

    #[test]
    fn test_confidence_weighted_strategy() {
        let signals = vec![0.6, 0.7, 0.4];
        let confidences = vec![0.9, 0.3, 0.8];
        let mut strategy = ConfidenceWeightedStrategy::new(signals, confidences, 0.5, 0.5);

        let bars = create_test_bars(3);

        // First bar - high confidence, should trade
        let ctx = create_context(&bars, 0, 0.0);
        let signal = strategy.on_bar(&ctx);
        assert_eq!(signal, Signal::Long);
        assert!((strategy.position_size_multiplier() - 0.9).abs() < 0.01);

        // Second bar - low confidence, should not enter
        let ctx = create_context(&bars, 1, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Hold);
    }

    #[test]
    fn test_ensemble_strategy_mean() {
        let model1 = vec![0.3, 0.5, 0.7];
        let model2 = vec![0.2, 0.6, 0.8];
        let model3 = vec![0.4, 0.4, 0.6];

        let mut strategy = EnsembleSignalStrategy::new(vec![model1, model2, model3], 0.5);

        let bars = create_test_bars(3);

        // First bar - mean of [0.3, 0.2, 0.4] = 0.3, below threshold
        let ctx = create_context(&bars, 0, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Hold);

        // Third bar - mean of [0.7, 0.8, 0.6] = 0.7, above threshold
        let ctx = create_context(&bars, 2, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Long);
    }

    #[test]
    fn test_ensemble_strategy_vote() {
        let model1 = vec![0.3, -0.3, 0.3];
        let model2 = vec![0.2, -0.2, -0.2];
        let model3 = vec![0.4, -0.4, 0.1];

        let mut strategy = EnsembleSignalStrategy::new(vec![model1, model2, model3], 0.5)
            .with_aggregation(AggregationMethod::Vote);

        let bars = create_test_bars(3);

        // First bar - all positive, vote = 1.0
        let ctx = create_context(&bars, 0, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Long);

        // Second bar - all negative, vote = -1.0
        let ctx = create_context(&bars, 1, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Short);
    }

    #[test]
    fn test_timestamped_strategy() {
        let bars = create_test_bars(5);
        let timestamps: Vec<i64> = bars.iter().map(|b| b.timestamp.timestamp()).collect();
        let values = vec![0.1, 0.6, 0.3, -0.6, 0.2];

        let mut strategy = TimestampedSignalStrategy::new(timestamps, values, 0.5);

        // Second bar - signal 0.6 > 0.5
        let ctx = create_context(&bars, 1, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Long);

        // Fourth bar - signal -0.6 < -0.5
        let ctx = create_context(&bars, 3, 0.0);
        assert_eq!(strategy.on_bar(&ctx), Signal::Short);
    }
}

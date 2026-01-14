//! Feature extraction for machine learning workflows.
//!
//! This module provides utilities for extracting features from price data
//! that can be used for training machine learning models, particularly
//! deep learning models for time series forecasting.
//!
//! # Feature Categories
//!
//! - **Price features**: Returns, log returns, price ratios
//! - **Technical features**: RSI, MACD, Bollinger Bands, etc.
//! - **Volume features**: Volume ratios, OBV
//! - **Volatility features**: ATR, realized volatility
//! - **Time features**: Day of week, month, etc.
//!
//! # Example
//!
//! ```
//! use mantis::features::{FeatureExtractor, FeatureConfig};
//! use mantis::data::load_csv;
//!
//! let bars = load_csv("data/sample.csv", &Default::default()).unwrap();
//! let config = FeatureConfig::default();
//! let extractor = FeatureExtractor::new(config);
//! let features = extractor.extract(&bars);
//! ```

use crate::data::{atr, bollinger_bands, ema, macd, rsi, sma, std_dev};
use crate::types::Bar;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for feature extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Lookback periods for returns calculation.
    pub return_periods: Vec<usize>,
    /// Periods for moving averages.
    pub ma_periods: Vec<usize>,
    /// RSI period.
    pub rsi_period: usize,
    /// MACD parameters (fast, slow, signal).
    pub macd_params: (usize, usize, usize),
    /// ATR period.
    pub atr_period: usize,
    /// Bollinger Bands period and std multiplier.
    pub bb_params: (usize, f64),
    /// Whether to include time-based features.
    pub include_time_features: bool,
    /// Whether to include volume features.
    pub include_volume_features: bool,
    /// Whether to normalize features.
    pub normalize: bool,
    /// Lookback window for normalization (rolling z-score).
    pub normalize_window: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            return_periods: vec![1, 5, 10, 20],
            ma_periods: vec![5, 10, 20, 50],
            rsi_period: 14,
            macd_params: (12, 26, 9),
            atr_period: 14,
            bb_params: (20, 2.0),
            include_time_features: true,
            include_volume_features: true,
            normalize: true,
            normalize_window: 252,
        }
    }
}

impl FeatureConfig {
    /// Create a minimal configuration for fast extraction.
    pub fn minimal() -> Self {
        Self {
            return_periods: vec![1, 5],
            ma_periods: vec![10, 20],
            rsi_period: 14,
            macd_params: (12, 26, 9),
            atr_period: 14,
            bb_params: (20, 2.0),
            include_time_features: false,
            include_volume_features: false,
            normalize: false,
            normalize_window: 252,
        }
    }

    /// Create a comprehensive configuration for deep learning.
    pub fn comprehensive() -> Self {
        Self {
            return_periods: vec![1, 2, 3, 5, 10, 20, 40, 60],
            ma_periods: vec![5, 10, 20, 50, 100, 200],
            rsi_period: 14,
            macd_params: (12, 26, 9),
            atr_period: 14,
            bb_params: (20, 2.0),
            include_time_features: true,
            include_volume_features: true,
            normalize: true,
            normalize_window: 252,
        }
    }
}

/// A single row of extracted features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureRow {
    /// Index in the original data.
    pub index: usize,
    /// Timestamp as Unix epoch.
    pub timestamp: i64,
    /// Feature values by name.
    pub features: HashMap<String, f64>,
    /// Target variable (e.g., future return).
    pub target: Option<f64>,
}

/// Feature extractor for ML workflows.
pub struct FeatureExtractor {
    config: FeatureConfig,
}

impl FeatureExtractor {
    /// Create a new feature extractor.
    pub fn new(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_extractor() -> Self {
        Self::new(FeatureConfig::default())
    }

    /// Get the minimum warmup period needed.
    pub fn warmup_period(&self) -> usize {
        let max_ma = self.config.ma_periods.iter().max().copied().unwrap_or(50);
        let max_return = self.config.return_periods.iter().max().copied().unwrap_or(20);
        let macd_warmup = self.config.macd_params.1 + self.config.macd_params.2;

        [max_ma, max_return, macd_warmup, self.config.atr_period, self.config.bb_params.0]
            .iter()
            .max()
            .copied()
            .unwrap_or(50)
    }

    /// Extract features from a slice of bars.
    pub fn extract(&self, bars: &[Bar]) -> Vec<FeatureRow> {
        let warmup = self.warmup_period();
        if bars.len() <= warmup {
            return vec![];
        }

        let mut rows = Vec::with_capacity(bars.len() - warmup);

        for i in warmup..bars.len() {
            let history = &bars[..=i];
            let bar = &bars[i];
            let mut features = HashMap::new();

            // Price features
            self.extract_price_features(&mut features, history, bar);

            // Technical indicators
            self.extract_technical_features(&mut features, history);

            // Volume features
            if self.config.include_volume_features {
                self.extract_volume_features(&mut features, history, bar);
            }

            // Time features
            if self.config.include_time_features {
                self.extract_time_features(&mut features, bar);
            }

            // Normalize if configured
            if self.config.normalize && i >= self.config.normalize_window {
                self.normalize_features(&mut features, bars, i);
            }

            rows.push(FeatureRow {
                index: i,
                timestamp: bar.timestamp.timestamp(),
                features,
                target: None, // Set separately
            });
        }

        rows
    }

    /// Extract features with target variable (future returns).
    pub fn extract_with_target(
        &self,
        bars: &[Bar],
        target_horizon: usize,
    ) -> Vec<FeatureRow> {
        let mut rows = self.extract(bars);

        // Add target variable (forward return)
        for row in &mut rows {
            let target_idx = row.index + target_horizon;
            if target_idx < bars.len() {
                let current_price = bars[row.index].close;
                let future_price = bars[target_idx].close;
                let forward_return = (future_price - current_price) / current_price;
                row.target = Some(forward_return);
            }
        }

        rows
    }

    /// Extract features to a 2D array format (for ML frameworks).
    pub fn extract_matrix(&self, bars: &[Bar]) -> (Vec<Vec<f64>>, Vec<String>) {
        let rows = self.extract(bars);
        if rows.is_empty() {
            return (vec![], vec![]);
        }

        // Get feature names from first row
        let mut feature_names: Vec<String> = rows[0].features.keys().cloned().collect();
        feature_names.sort(); // Consistent ordering

        // Build matrix
        let matrix: Vec<Vec<f64>> = rows
            .iter()
            .map(|row| {
                feature_names
                    .iter()
                    .map(|name| row.features.get(name).copied().unwrap_or(f64::NAN))
                    .collect()
            })
            .collect();

        (matrix, feature_names)
    }

    /// Extract features to CSV format string.
    pub fn to_csv(&self, bars: &[Bar], target_horizon: Option<usize>) -> String {
        let rows = if let Some(horizon) = target_horizon {
            self.extract_with_target(bars, horizon)
        } else {
            self.extract(bars)
        };

        if rows.is_empty() {
            return String::new();
        }

        // Get sorted feature names
        let mut feature_names: Vec<String> = rows[0].features.keys().cloned().collect();
        feature_names.sort();

        // Build header
        let mut header = vec!["index".to_string(), "timestamp".to_string()];
        header.extend(feature_names.clone());
        if target_horizon.is_some() {
            header.push("target".to_string());
        }

        let mut output = header.join(",") + "\n";

        // Build data rows
        for row in &rows {
            let mut values: Vec<String> = vec![
                row.index.to_string(),
                row.timestamp.to_string(),
            ];

            for name in &feature_names {
                let value = row.features.get(name).copied().unwrap_or(f64::NAN);
                values.push(format!("{:.6}", value));
            }

            if target_horizon.is_some() {
                let target = row.target.unwrap_or(f64::NAN);
                values.push(format!("{:.6}", target));
            }

            output.push_str(&values.join(","));
            output.push('\n');
        }

        output
    }

    /// Extract price-based features.
    fn extract_price_features(
        &self,
        features: &mut HashMap<String, f64>,
        history: &[Bar],
        bar: &Bar,
    ) {
        let close = bar.close;

        // Simple return for each period
        for &period in &self.config.return_periods {
            if history.len() > period {
                let past_close = history[history.len() - 1 - period].close;
                let ret = (close - past_close) / past_close;
                features.insert(format!("return_{}", period), ret);

                // Log return
                if past_close > 0.0 && close > 0.0 {
                    let log_ret = (close / past_close).ln();
                    features.insert(format!("log_return_{}", period), log_ret);
                }
            }
        }

        // OHLC ratios
        features.insert("hl_ratio".to_string(), (bar.high - bar.low) / close);
        features.insert("oc_ratio".to_string(), (bar.close - bar.open) / close);
        features.insert("upper_shadow".to_string(), (bar.high - bar.close.max(bar.open)) / close);
        features.insert("lower_shadow".to_string(), (bar.close.min(bar.open) - bar.low) / close);

        // Body direction
        features.insert("body_direction".to_string(), if bar.is_bullish() { 1.0 } else { -1.0 });
    }

    /// Extract technical indicator features.
    fn extract_technical_features(
        &self,
        features: &mut HashMap<String, f64>,
        history: &[Bar],
    ) {
        let close = history.last().map(|b| b.close).unwrap_or(0.0);

        // Moving averages
        for &period in &self.config.ma_periods {
            if let Some(sma_val) = sma(history, period) {
                features.insert(format!("sma_{}", period), sma_val);
                // Price relative to MA
                if sma_val > 0.0 {
                    features.insert(format!("price_sma_{}_ratio", period), close / sma_val);
                }
            }

            if let Some(ema_val) = ema(history, period) {
                features.insert(format!("ema_{}", period), ema_val);
                if ema_val > 0.0 {
                    features.insert(format!("price_ema_{}_ratio", period), close / ema_val);
                }
            }
        }

        // MA crossover features
        if self.config.ma_periods.len() >= 2 {
            let short_period = self.config.ma_periods[0];
            let long_period = self.config.ma_periods[1];

            if let (Some(short_ma), Some(long_ma)) = (sma(history, short_period), sma(history, long_period)) {
                if long_ma > 0.0 {
                    features.insert("ma_crossover_ratio".to_string(), short_ma / long_ma);
                }
            }
        }

        // RSI
        if let Some(rsi_val) = rsi(history, self.config.rsi_period) {
            features.insert("rsi".to_string(), rsi_val);
            features.insert("rsi_normalized".to_string(), (rsi_val - 50.0) / 50.0); // Normalized to [-1, 1]
        }

        // MACD
        let (fast, slow, signal) = self.config.macd_params;
        if let Some((macd_line, signal_line, histogram)) = macd(history, fast, slow, signal) {
            features.insert("macd_line".to_string(), macd_line);
            features.insert("macd_signal".to_string(), signal_line);
            features.insert("macd_histogram".to_string(), histogram);

            // Normalized MACD (relative to price)
            if close > 0.0 {
                features.insert("macd_normalized".to_string(), macd_line / close * 100.0);
            }
        }

        // Bollinger Bands
        let (bb_period, bb_std) = self.config.bb_params;
        if let Some((middle, upper, lower)) = bollinger_bands(history, bb_period, bb_std) {
            features.insert("bb_middle".to_string(), middle);
            features.insert("bb_upper".to_string(), upper);
            features.insert("bb_lower".to_string(), lower);

            // %B (price position within bands)
            let band_width = upper - lower;
            if band_width > 0.0 {
                let percent_b = (close - lower) / band_width;
                features.insert("bb_percent_b".to_string(), percent_b);
                features.insert("bb_width".to_string(), band_width / middle);
            }
        }

        // ATR
        if let Some(atr_val) = atr(history, self.config.atr_period) {
            features.insert("atr".to_string(), atr_val);
            // Normalized ATR
            if close > 0.0 {
                features.insert("atr_normalized".to_string(), atr_val / close);
            }
        }

        // Standard deviation
        if let Some(std_val) = std_dev(history, 20) {
            features.insert("std_dev_20".to_string(), std_val);
            if close > 0.0 {
                features.insert("volatility_20".to_string(), std_val / close);
            }
        }
    }

    /// Extract volume-based features.
    fn extract_volume_features(
        &self,
        features: &mut HashMap<String, f64>,
        history: &[Bar],
        bar: &Bar,
    ) {
        let volume = bar.volume;

        // Volume
        features.insert("volume".to_string(), volume);

        // Volume moving averages
        for &period in &[5, 10, 20] {
            if history.len() >= period {
                let vol_sum: f64 = history[history.len() - period..]
                    .iter()
                    .map(|b| b.volume)
                    .sum();
                let vol_avg = vol_sum / period as f64;
                features.insert(format!("volume_sma_{}", period), vol_avg);

                // Volume ratio
                if vol_avg > 0.0 {
                    features.insert(format!("volume_ratio_{}", period), volume / vol_avg);
                }
            }
        }

        // Dollar volume (price * volume)
        features.insert("dollar_volume".to_string(), bar.close * volume);

        // Volume price trend
        if history.len() >= 2 {
            let prev_bar = &history[history.len() - 2];
            let price_change = bar.close - prev_bar.close;
            features.insert("volume_price_trend".to_string(), price_change.signum() * volume);
        }
    }

    /// Extract time-based features.
    fn extract_time_features(&self, features: &mut HashMap<String, f64>, bar: &Bar) {
        use chrono::{Datelike, Timelike};

        let dt = bar.timestamp;

        // Day of week (0-6, with cyclical encoding)
        let dow = dt.weekday().num_days_from_monday() as f64;
        features.insert("day_of_week".to_string(), dow);
        features.insert("day_of_week_sin".to_string(), (2.0 * std::f64::consts::PI * dow / 7.0).sin());
        features.insert("day_of_week_cos".to_string(), (2.0 * std::f64::consts::PI * dow / 7.0).cos());

        // Month (1-12, with cyclical encoding)
        let month = dt.month() as f64;
        features.insert("month".to_string(), month);
        features.insert("month_sin".to_string(), (2.0 * std::f64::consts::PI * month / 12.0).sin());
        features.insert("month_cos".to_string(), (2.0 * std::f64::consts::PI * month / 12.0).cos());

        // Day of month (1-31)
        let day = dt.day() as f64;
        features.insert("day_of_month".to_string(), day);
        features.insert("day_of_month_sin".to_string(), (2.0 * std::f64::consts::PI * day / 31.0).sin());
        features.insert("day_of_month_cos".to_string(), (2.0 * std::f64::consts::PI * day / 31.0).cos());

        // Hour of day (for intraday data)
        let hour = dt.hour() as f64;
        features.insert("hour".to_string(), hour);
        features.insert("hour_sin".to_string(), (2.0 * std::f64::consts::PI * hour / 24.0).sin());
        features.insert("hour_cos".to_string(), (2.0 * std::f64::consts::PI * hour / 24.0).cos());

        // Quarter
        let quarter = ((month - 1.0) / 3.0).floor() + 1.0;
        features.insert("quarter".to_string(), quarter);

        // Is start/end of month
        features.insert("is_month_start".to_string(), if day <= 5.0 { 1.0 } else { 0.0 });
        features.insert("is_month_end".to_string(), if day >= 25.0 { 1.0 } else { 0.0 });

        // Year progress
        let day_of_year = dt.ordinal() as f64;
        features.insert("year_progress".to_string(), day_of_year / 365.0);
    }

    /// Normalize features using rolling z-score.
    fn normalize_features(
        &self,
        features: &mut HashMap<String, f64>,
        bars: &[Bar],
        current_idx: usize,
    ) {
        // For simplicity, normalize price-based features by recent volatility
        let start_idx = current_idx.saturating_sub(self.config.normalize_window);
        let window_bars = &bars[start_idx..=current_idx];

        if window_bars.len() < 20 {
            return;
        }

        // Calculate mean and std of closes for normalization
        let closes: Vec<f64> = window_bars.iter().map(|b| b.close).collect();
        let mean: f64 = closes.iter().sum::<f64>() / closes.len() as f64;
        let variance: f64 = closes.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / closes.len() as f64;
        let std = variance.sqrt();

        if std > 0.0 {
            // Normalize relevant features
            let keys_to_normalize: Vec<String> = features
                .keys()
                .filter(|k| {
                    k.starts_with("sma_")
                        || k.starts_with("ema_")
                        || k.starts_with("bb_")
                        || k.starts_with("atr")
                })
                .cloned()
                .collect();

            for key in keys_to_normalize {
                if key.starts_with("atr") || key.ends_with("_ratio") || key.ends_with("_normalized") {
                    continue; // Already relative
                }

                if let Some(value) = features.get(&key) {
                    let normalized = (*value - mean) / std;
                    features.insert(format!("{}_zscore", key), normalized);
                }
            }
        }
    }
}

/// Create sequences for time-series models (LSTM, Transformer, etc.).
pub struct SequenceBuilder {
    sequence_length: usize,
    feature_names: Vec<String>,
}

impl SequenceBuilder {
    /// Create a new sequence builder.
    pub fn new(sequence_length: usize) -> Self {
        Self {
            sequence_length,
            feature_names: Vec::new(),
        }
    }

    /// Build sequences from feature rows.
    pub fn build_sequences(
        &mut self,
        rows: &[FeatureRow],
    ) -> (Vec<Vec<Vec<f64>>>, Vec<Option<f64>>) {
        if rows.is_empty() || rows.len() < self.sequence_length {
            return (vec![], vec![]);
        }

        // Get feature names from first row
        if self.feature_names.is_empty() {
            self.feature_names = rows[0].features.keys().cloned().collect();
            self.feature_names.sort();
        }

        let _num_features = self.feature_names.len();
        let num_sequences = rows.len() - self.sequence_length + 1;

        let mut sequences = Vec::with_capacity(num_sequences);
        let mut targets = Vec::with_capacity(num_sequences);

        for i in 0..num_sequences {
            let seq_rows = &rows[i..i + self.sequence_length];

            let sequence: Vec<Vec<f64>> = seq_rows
                .iter()
                .map(|row| {
                    self.feature_names
                        .iter()
                        .map(|name| row.features.get(name).copied().unwrap_or(f64::NAN))
                        .collect()
                })
                .collect();

            let target = seq_rows.last().and_then(|r| r.target);

            sequences.push(sequence);
            targets.push(target);
        }

        (sequences, targets)
    }

    /// Build sequences and export to JSON format for ML frameworks.
    pub fn to_json(&mut self, rows: &[FeatureRow]) -> String {
        let (sequences, targets) = self.build_sequences(rows);

        let data = serde_json::json!({
            "sequence_length": self.sequence_length,
            "num_features": self.feature_names.len(),
            "feature_names": self.feature_names,
            "num_sequences": sequences.len(),
            "sequences": sequences,
            "targets": targets,
        });

        serde_json::to_string_pretty(&data).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get the feature names.
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
}

/// Train/test splitter that respects temporal ordering.
pub struct TimeSeriesSplitter {
    train_ratio: f64,
    validation_ratio: f64,
    gap: usize, // Gap between train/val/test to prevent leakage
}

impl TimeSeriesSplitter {
    /// Create a new splitter with train/val/test ratios.
    pub fn new(train_ratio: f64, validation_ratio: f64) -> Self {
        assert!(train_ratio > 0.0 && train_ratio < 1.0);
        assert!(train_ratio + validation_ratio < 1.0);

        Self {
            train_ratio,
            validation_ratio,
            gap: 1,
        }
    }

    /// Set the gap between splits to prevent data leakage.
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Split data into train/validation/test sets.
    pub fn split<T: Clone>(&self, data: &[T]) -> (Vec<T>, Vec<T>, Vec<T>) {
        let n = data.len();
        let train_end = (n as f64 * self.train_ratio) as usize;
        let val_end = train_end + (n as f64 * self.validation_ratio) as usize;

        let train = data[..train_end].to_vec();
        let val_start = (train_end + self.gap).min(n);
        let validation = if val_start < val_end {
            data[val_start..val_end].to_vec()
        } else {
            vec![]
        };
        let test_start = (val_end + self.gap).min(n);
        let test = if test_start < n {
            data[test_start..].to_vec()
        } else {
            vec![]
        };

        (train, validation, test)
    }

    /// Perform K-fold time series cross-validation (expanding window).
    pub fn time_series_cv<T: Clone>(&self, data: &[T], n_splits: usize) -> Vec<(Vec<T>, Vec<T>)> {
        let n = data.len();
        let min_train_size = n / (n_splits + 1);
        let fold_size = (n - min_train_size) / n_splits;

        let mut splits = Vec::with_capacity(n_splits);

        for i in 0..n_splits {
            let train_end = min_train_size + i * fold_size;
            let test_start = (train_end + self.gap).min(n);
            let test_end = (train_end + fold_size + self.gap).min(n);

            let train = data[..train_end].to_vec();
            let test = data[test_start..test_end].to_vec();

            splits.push((train, test));
        }

        splits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn create_test_bars(count: usize) -> Vec<Bar> {
        (0..count)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.1) + (i as f64 * 0.5).sin() * 5.0;
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    base - 1.0,
                    base + 2.0,
                    base - 2.0,
                    base + 0.5,
                    1000000.0 + (i as f64 * 100.0),
                )
            })
            .collect()
    }

    #[test]
    fn test_feature_extractor_creation() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);
        assert!(extractor.warmup_period() > 0);
    }

    #[test]
    fn test_feature_extraction() {
        let bars = create_test_bars(100);
        let extractor = FeatureExtractor::default_extractor();
        let rows = extractor.extract(&bars);

        assert!(!rows.is_empty());
        assert!(rows[0].features.contains_key("return_1"));
        assert!(rows[0].features.contains_key("rsi"));
    }

    #[test]
    fn test_feature_extraction_with_target() {
        let bars = create_test_bars(100);
        let extractor = FeatureExtractor::default_extractor();
        let rows = extractor.extract_with_target(&bars, 5);

        // Some rows should have targets
        let with_target: Vec<_> = rows.iter().filter(|r| r.target.is_some()).collect();
        assert!(!with_target.is_empty());
    }

    #[test]
    fn test_feature_matrix() {
        let bars = create_test_bars(100);
        let extractor = FeatureExtractor::new(FeatureConfig::minimal());
        let (matrix, names) = extractor.extract_matrix(&bars);

        assert!(!matrix.is_empty());
        assert!(!names.is_empty());
        assert_eq!(matrix[0].len(), names.len());
    }

    #[test]
    fn test_csv_export() {
        let bars = create_test_bars(100);
        let extractor = FeatureExtractor::new(FeatureConfig::minimal());
        let csv = extractor.to_csv(&bars, Some(5));

        assert!(!csv.is_empty());
        assert!(csv.contains("index,timestamp"));
        assert!(csv.contains("target"));
    }

    #[test]
    fn test_sequence_builder() {
        let bars = create_test_bars(100);
        let extractor = FeatureExtractor::new(FeatureConfig::minimal());
        let rows = extractor.extract(&bars);

        let mut builder = SequenceBuilder::new(10);
        let (sequences, targets) = builder.build_sequences(&rows);

        assert!(!sequences.is_empty());
        assert_eq!(sequences.len(), targets.len());
        assert_eq!(sequences[0].len(), 10); // Sequence length
    }

    #[test]
    fn test_time_series_splitter() {
        let data: Vec<i32> = (0..100).collect();
        let splitter = TimeSeriesSplitter::new(0.7, 0.15);
        let (train, val, test) = splitter.split(&data);

        assert!(train.len() > val.len());
        assert!(train.len() > test.len());
        assert!(!train.is_empty());
        assert!(!val.is_empty());
        assert!(!test.is_empty());

        // Verify temporal ordering
        assert!(*train.last().unwrap() < *val.first().unwrap());
        assert!(*val.last().unwrap() < *test.first().unwrap());
    }

    #[test]
    fn test_time_series_cv() {
        let data: Vec<i32> = (0..100).collect();
        let splitter = TimeSeriesSplitter::new(0.7, 0.15);
        let splits = splitter.time_series_cv(&data, 5);

        assert_eq!(splits.len(), 5);

        // Each split should have expanding training set
        for i in 1..splits.len() {
            assert!(splits[i].0.len() > splits[i - 1].0.len());
        }
    }
}

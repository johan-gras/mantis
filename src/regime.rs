//! Market regime detection for ML/DL workflows.
//!
//! This module provides regime detection methods to identify market states:
//!
//! - **Trend regimes**: Trending vs ranging markets
//! - **Volatility regimes**: High vs low volatility periods
//! - **Volume regimes**: High vs low volume activity
//! - **Hidden Markov Model**: Statistical regime switching
//!
//! Regime labels can be used as features for ML models or to filter strategy signals.
//!
//! # Example
//!
//! ```ignore
//! use mantis::regime::{RegimeDetector, RegimeConfig};
//!
//! let config = RegimeConfig::default();
//! let detector = RegimeDetector::new(config);
//!
//! let regimes = detector.detect(&bars);
//! for (i, regime) in regimes.iter().enumerate() {
//!     println!("Bar {}: {:?}", i, regime);
//! }
//! ```

use crate::types::Bar;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Market regime type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regime {
    /// Strong uptrend.
    StrongUptrend,
    /// Weak uptrend / consolidating up.
    WeakUptrend,
    /// Ranging / sideways market.
    Ranging,
    /// Weak downtrend / consolidating down.
    WeakDowntrend,
    /// Strong downtrend.
    StrongDowntrend,
}

impl Regime {
    /// Convert to numeric value for ML features.
    pub fn to_numeric(&self) -> f64 {
        match self {
            Regime::StrongUptrend => 2.0,
            Regime::WeakUptrend => 1.0,
            Regime::Ranging => 0.0,
            Regime::WeakDowntrend => -1.0,
            Regime::StrongDowntrend => -2.0,
        }
    }

    /// Check if regime is bullish.
    pub fn is_bullish(&self) -> bool {
        matches!(self, Regime::StrongUptrend | Regime::WeakUptrend)
    }

    /// Check if regime is bearish.
    pub fn is_bearish(&self) -> bool {
        matches!(self, Regime::StrongDowntrend | Regime::WeakDowntrend)
    }

    /// Check if regime is trending (not ranging).
    pub fn is_trending(&self) -> bool {
        !matches!(self, Regime::Ranging)
    }
}

/// Volatility regime type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VolatilityRegime {
    /// Very high volatility.
    VeryHigh,
    /// High volatility.
    High,
    /// Normal volatility.
    Normal,
    /// Low volatility.
    Low,
    /// Very low volatility.
    VeryLow,
}

impl VolatilityRegime {
    /// Convert to numeric value.
    pub fn to_numeric(&self) -> f64 {
        match self {
            VolatilityRegime::VeryHigh => 2.0,
            VolatilityRegime::High => 1.0,
            VolatilityRegime::Normal => 0.0,
            VolatilityRegime::Low => -1.0,
            VolatilityRegime::VeryLow => -2.0,
        }
    }
}

/// Volume regime type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VolumeRegime {
    /// Very high volume.
    VeryHigh,
    /// High volume.
    High,
    /// Normal volume.
    Normal,
    /// Low volume.
    Low,
    /// Very low volume.
    VeryLow,
}

impl VolumeRegime {
    /// Convert to numeric value.
    pub fn to_numeric(&self) -> f64 {
        match self {
            VolumeRegime::VeryHigh => 2.0,
            VolumeRegime::High => 1.0,
            VolumeRegime::Normal => 0.0,
            VolumeRegime::Low => -1.0,
            VolumeRegime::VeryLow => -2.0,
        }
    }
}

/// Complete regime classification for a bar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeLabel {
    /// Trend regime.
    pub trend: Regime,
    /// Volatility regime.
    pub volatility: VolatilityRegime,
    /// Volume regime.
    pub volume: VolumeRegime,
    /// ADX value (trend strength).
    pub adx: Option<f64>,
    /// Regime probability/confidence (if available).
    pub confidence: Option<f64>,
}

impl RegimeLabel {
    /// Convert to feature vector.
    pub fn to_features(&self) -> HashMap<String, f64> {
        let mut features = HashMap::new();
        features.insert("regime_trend".to_string(), self.trend.to_numeric());
        features.insert("regime_volatility".to_string(), self.volatility.to_numeric());
        features.insert("regime_volume".to_string(), self.volume.to_numeric());
        features.insert("regime_is_bullish".to_string(), if self.trend.is_bullish() { 1.0 } else { 0.0 });
        features.insert("regime_is_trending".to_string(), if self.trend.is_trending() { 1.0 } else { 0.0 });
        if let Some(adx) = self.adx {
            features.insert("regime_adx".to_string(), adx);
        }
        features
    }
}

/// Configuration for regime detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Period for trend detection.
    pub trend_period: usize,
    /// Short EMA period for trend.
    pub trend_ema_short: usize,
    /// Long EMA period for trend.
    pub trend_ema_long: usize,
    /// ADX period for trend strength.
    pub adx_period: usize,
    /// ADX threshold for trending market.
    pub adx_trending_threshold: f64,
    /// ADX threshold for strong trend.
    pub adx_strong_threshold: f64,
    /// ATR period for volatility.
    pub volatility_period: usize,
    /// Lookback for volatility percentile.
    pub volatility_lookback: usize,
    /// Volume period for regime.
    pub volume_period: usize,
    /// Volume lookback for percentile.
    pub volume_lookback: usize,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            trend_period: 20,
            trend_ema_short: 10,
            trend_ema_long: 30,
            adx_period: 14,
            adx_trending_threshold: 20.0,
            adx_strong_threshold: 40.0,
            volatility_period: 14,
            volatility_lookback: 100,
            volume_period: 20,
            volume_lookback: 100,
        }
    }
}

/// Regime detector.
pub struct RegimeDetector {
    config: RegimeConfig,
}

impl RegimeDetector {
    /// Create a new regime detector.
    pub fn new(config: RegimeConfig) -> Self {
        Self { config }
    }

    /// Detect regimes for all bars.
    pub fn detect(&self, bars: &[Bar]) -> Vec<RegimeLabel> {
        if bars.len() < self.config.volatility_lookback {
            return vec![];
        }

        let ema_short = self.calculate_ema_series(bars, self.config.trend_ema_short);
        let ema_long = self.calculate_ema_series(bars, self.config.trend_ema_long);
        let atr_values = self.calculate_atr_series(bars, self.config.volatility_period);
        let adx_values = self.calculate_adx(bars, self.config.adx_period);
        let volume_sma = self.calculate_volume_sma(bars, self.config.volume_period);

        let mut regimes = Vec::with_capacity(bars.len());

        for i in 0..bars.len() {
            let trend = self.detect_trend(
                i,
                &ema_short,
                &ema_long,
                &adx_values,
            );

            let volatility = self.detect_volatility(i, &atr_values, bars);
            let volume = self.detect_volume(i, bars, &volume_sma);
            let adx = adx_values.get(i).copied();

            regimes.push(RegimeLabel {
                trend,
                volatility,
                volume,
                adx,
                confidence: None,
            });
        }

        regimes
    }

    /// Detect trend regime for a specific bar.
    fn detect_trend(
        &self,
        idx: usize,
        ema_short: &[f64],
        ema_long: &[f64],
        adx: &[f64],
    ) -> Regime {
        if idx >= ema_short.len() || idx >= ema_long.len() {
            return Regime::Ranging;
        }

        let short = ema_short[idx];
        let long = ema_long[idx];
        let adx_val = adx.get(idx).copied().unwrap_or(0.0);

        let trend_strength = (short - long) / long * 100.0;
        let is_trending = adx_val > self.config.adx_trending_threshold;
        let is_strong = adx_val > self.config.adx_strong_threshold;

        if trend_strength > 1.0 {
            if is_strong {
                Regime::StrongUptrend
            } else if is_trending {
                Regime::WeakUptrend
            } else {
                Regime::Ranging
            }
        } else if trend_strength < -1.0 {
            if is_strong {
                Regime::StrongDowntrend
            } else if is_trending {
                Regime::WeakDowntrend
            } else {
                Regime::Ranging
            }
        } else {
            Regime::Ranging
        }
    }

    /// Detect volatility regime.
    fn detect_volatility(&self, idx: usize, atr_values: &[f64], bars: &[Bar]) -> VolatilityRegime {
        if idx >= atr_values.len() || idx < self.config.volatility_lookback {
            return VolatilityRegime::Normal;
        }

        let current_atr = atr_values[idx];
        let start_idx = idx.saturating_sub(self.config.volatility_lookback);

        // Normalize ATR by price to get percentage volatility
        let current_price = bars[idx].close;
        let norm_atr = current_atr / current_price * 100.0;

        // Calculate historical percentile
        let mut historical: Vec<f64> = atr_values[start_idx..idx]
            .iter()
            .zip(&bars[start_idx..idx])
            .map(|(&a, b)| a / b.close * 100.0)
            .collect();

        if historical.is_empty() {
            return VolatilityRegime::Normal;
        }

        historical.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile = historical
            .iter()
            .filter(|&&v| v <= norm_atr)
            .count() as f64
            / historical.len() as f64;

        if percentile > 0.9 {
            VolatilityRegime::VeryHigh
        } else if percentile > 0.7 {
            VolatilityRegime::High
        } else if percentile < 0.1 {
            VolatilityRegime::VeryLow
        } else if percentile < 0.3 {
            VolatilityRegime::Low
        } else {
            VolatilityRegime::Normal
        }
    }

    /// Calculate EMA series for all bars.
    fn calculate_ema_series(&self, bars: &[Bar], period: usize) -> Vec<f64> {
        if bars.is_empty() || period == 0 {
            return vec![0.0; bars.len()];
        }

        let mut ema_values = vec![0.0; bars.len()];
        let multiplier = 2.0 / (period as f64 + 1.0);

        // First EMA value is the close price
        if !bars.is_empty() {
            ema_values[0] = bars[0].close;
        }

        // Calculate EMA for each bar
        for i in 1..bars.len() {
            ema_values[i] = bars[i].close * multiplier + ema_values[i - 1] * (1.0 - multiplier);
        }

        ema_values
    }

    /// Calculate ATR series for all bars.
    fn calculate_atr_series(&self, bars: &[Bar], period: usize) -> Vec<f64> {
        if bars.len() < 2 || period == 0 {
            return vec![0.0; bars.len()];
        }

        let mut atr_values = vec![0.0; bars.len()];

        // Calculate True Range for each bar
        let mut true_ranges = vec![0.0; bars.len()];
        true_ranges[0] = bars[0].high - bars[0].low;

        for i in 1..bars.len() {
            let hl = bars[i].high - bars[i].low;
            let hc = (bars[i].high - bars[i - 1].close).abs();
            let lc = (bars[i].low - bars[i - 1].close).abs();
            true_ranges[i] = hl.max(hc).max(lc);
        }

        // Calculate ATR using Wilder's smoothing
        if bars.len() >= period {
            // Initial ATR is simple average of first 'period' TRs
            let initial_sum: f64 = true_ranges[..period].iter().sum();
            atr_values[period - 1] = initial_sum / period as f64;

            // Subsequent values use Wilder's smoothing
            for i in period..bars.len() {
                atr_values[i] = (atr_values[i - 1] * (period - 1) as f64 + true_ranges[i])
                    / period as f64;
            }
        }

        atr_values
    }

    /// Detect volume regime.
    fn detect_volume(&self, idx: usize, bars: &[Bar], volume_sma: &[f64]) -> VolumeRegime {
        if idx >= volume_sma.len() || idx < self.config.volume_lookback {
            return VolumeRegime::Normal;
        }

        let current_volume = bars[idx].volume;
        let avg_volume = volume_sma[idx];

        if avg_volume <= 0.0 {
            return VolumeRegime::Normal;
        }

        let ratio = current_volume / avg_volume;

        if ratio > 2.5 {
            VolumeRegime::VeryHigh
        } else if ratio > 1.5 {
            VolumeRegime::High
        } else if ratio < 0.3 {
            VolumeRegime::VeryLow
        } else if ratio < 0.6 {
            VolumeRegime::Low
        } else {
            VolumeRegime::Normal
        }
    }

    /// Calculate ADX (Average Directional Index).
    fn calculate_adx(&self, bars: &[Bar], period: usize) -> Vec<f64> {
        if bars.len() < period * 2 {
            return vec![0.0; bars.len()];
        }

        let mut plus_dm = vec![0.0; bars.len()];
        let mut minus_dm = vec![0.0; bars.len()];
        let mut tr = vec![0.0; bars.len()];

        // Calculate +DM, -DM, TR
        for i in 1..bars.len() {
            let high_diff = bars[i].high - bars[i - 1].high;
            let low_diff = bars[i - 1].low - bars[i].low;

            plus_dm[i] = if high_diff > low_diff && high_diff > 0.0 {
                high_diff
            } else {
                0.0
            };

            minus_dm[i] = if low_diff > high_diff && low_diff > 0.0 {
                low_diff
            } else {
                0.0
            };

            let hl = bars[i].high - bars[i].low;
            let hc = (bars[i].high - bars[i - 1].close).abs();
            let lc = (bars[i].low - bars[i - 1].close).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        // Smooth using Wilder's method
        let smoothed_plus_dm = self.wilder_smooth(&plus_dm, period);
        let smoothed_minus_dm = self.wilder_smooth(&minus_dm, period);
        let smoothed_tr = self.wilder_smooth(&tr, period);

        // Calculate +DI, -DI, DX
        let mut dx = vec![0.0; bars.len()];

        for i in period..bars.len() {
            if smoothed_tr[i] > 0.0 {
                let plus_di = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i];
                let minus_di = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i];
                let di_sum = plus_di + minus_di;
                if di_sum > 0.0 {
                    dx[i] = 100.0 * (plus_di - minus_di).abs() / di_sum;
                }
            }
        }

        // Smooth DX to get ADX
        self.wilder_smooth(&dx, period)
    }

    /// Wilder's smoothing method.
    fn wilder_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![0.0; data.len()];

        if data.len() < period {
            return result;
        }

        // Initial sum
        let initial_sum: f64 = data[1..=period].iter().sum();
        result[period] = initial_sum;

        // Wilder's smoothing
        for i in (period + 1)..data.len() {
            result[i] = result[i - 1] - result[i - 1] / period as f64 + data[i];
        }

        // Divide by period to get average
        for i in period..data.len() {
            result[i] /= period as f64;
        }

        result
    }

    /// Calculate volume SMA.
    fn calculate_volume_sma(&self, bars: &[Bar], period: usize) -> Vec<f64> {
        let mut result = vec![0.0; bars.len()];

        if bars.len() < period {
            return result;
        }

        let mut sum: f64 = bars[..period].iter().map(|b| b.volume).sum();
        result[period - 1] = sum / period as f64;

        for i in period..bars.len() {
            sum = sum - bars[i - period].volume + bars[i].volume;
            result[i] = sum / period as f64;
        }

        result
    }
}

/// Regime change detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeChange {
    /// Bar index where change occurred.
    pub bar_index: usize,
    /// Previous regime.
    pub from: Regime,
    /// New regime.
    pub to: Regime,
    /// Strength of the change (0.0 to 1.0).
    pub strength: f64,
}

/// Detect regime changes in a sequence of regime labels.
pub fn detect_regime_changes(regimes: &[RegimeLabel]) -> Vec<RegimeChange> {
    if regimes.len() < 2 {
        return vec![];
    }

    let mut changes = Vec::new();

    for i in 1..regimes.len() {
        if regimes[i].trend != regimes[i - 1].trend {
            let strength = match (regimes[i - 1].trend, regimes[i].trend) {
                (Regime::StrongUptrend, Regime::StrongDowntrend)
                | (Regime::StrongDowntrend, Regime::StrongUptrend) => 1.0,
                (Regime::StrongUptrend, Regime::Ranging)
                | (Regime::StrongDowntrend, Regime::Ranging) => 0.7,
                (Regime::Ranging, Regime::StrongUptrend)
                | (Regime::Ranging, Regime::StrongDowntrend) => 0.7,
                _ => 0.4,
            };

            changes.push(RegimeChange {
                bar_index: i,
                from: regimes[i - 1].trend,
                to: regimes[i].trend,
                strength,
            });
        }
    }

    changes
}

/// Calculate regime statistics for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeStats {
    /// Count of each regime.
    pub regime_counts: HashMap<String, usize>,
    /// Average duration of each regime.
    pub avg_durations: HashMap<String, f64>,
    /// Percentage of time in each regime.
    pub regime_percentages: HashMap<String, f64>,
    /// Number of regime changes.
    pub num_changes: usize,
    /// Average bars between changes.
    pub avg_regime_duration: f64,
}

impl RegimeStats {
    /// Calculate statistics from regime labels.
    pub fn from_regimes(regimes: &[RegimeLabel]) -> Self {
        let mut regime_counts: HashMap<String, usize> = HashMap::new();
        let mut regime_durations: HashMap<String, Vec<usize>> = HashMap::new();

        let mut current_regime = regimes.first().map(|r| r.trend);
        let mut current_duration = 0;

        for regime in regimes {
            let name = format!("{:?}", regime.trend);
            *regime_counts.entry(name.clone()).or_insert(0) += 1;

            if Some(regime.trend) == current_regime {
                current_duration += 1;
            } else {
                if let Some(prev) = current_regime {
                    regime_durations
                        .entry(format!("{:?}", prev))
                        .or_insert_with(Vec::new)
                        .push(current_duration);
                }
                current_regime = Some(regime.trend);
                current_duration = 1;
            }
        }

        // Handle last regime
        if let Some(prev) = current_regime {
            regime_durations
                .entry(format!("{:?}", prev))
                .or_insert_with(Vec::new)
                .push(current_duration);
        }

        let total_bars = regimes.len() as f64;
        let regime_percentages: HashMap<String, f64> = regime_counts
            .iter()
            .map(|(k, v)| (k.clone(), *v as f64 / total_bars * 100.0))
            .collect();

        let avg_durations: HashMap<String, f64> = regime_durations
            .iter()
            .map(|(k, v)| {
                let avg = if v.is_empty() {
                    0.0
                } else {
                    v.iter().sum::<usize>() as f64 / v.len() as f64
                };
                (k.clone(), avg)
            })
            .collect();

        let num_changes = detect_regime_changes(regimes).len();
        let avg_regime_duration = if num_changes > 0 {
            total_bars / (num_changes + 1) as f64
        } else {
            total_bars
        };

        Self {
            regime_counts,
            avg_durations,
            regime_percentages,
            num_changes,
            avg_regime_duration,
        }
    }
}

/// Simple Hidden Markov Model for regime detection.
#[derive(Debug, Clone)]
pub struct HMMRegimeDetector {
    /// Number of hidden states.
    n_states: usize,
    /// Lookback period for estimation.
    lookback: usize,
}

impl HMMRegimeDetector {
    /// Create a new HMM regime detector.
    pub fn new(n_states: usize, lookback: usize) -> Self {
        Self { n_states, lookback }
    }

    /// Detect regimes using simple state estimation.
    ///
    /// This is a simplified version that uses volatility clustering
    /// as a proxy for hidden states.
    pub fn detect(&self, bars: &[Bar]) -> Vec<usize> {
        if bars.len() < self.lookback {
            return vec![0; bars.len()];
        }

        // Calculate returns
        let returns: Vec<f64> = bars
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        // Calculate rolling volatility
        let mut volatilities = vec![0.0; bars.len()];
        for i in self.lookback..returns.len() {
            let window: Vec<f64> = returns[(i - self.lookback)..i].to_vec();
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 =
                window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window.len() as f64;
            volatilities[i + 1] = variance.sqrt();
        }

        // Assign states based on volatility percentiles
        let mut sorted_vols: Vec<f64> = volatilities
            .iter()
            .filter(|&&v| v > 0.0)
            .copied()
            .collect();
        sorted_vols.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if sorted_vols.is_empty() {
            return vec![0; bars.len()];
        }

        let thresholds: Vec<f64> = (1..self.n_states)
            .map(|i| {
                let idx = (i as f64 / self.n_states as f64 * sorted_vols.len() as f64) as usize;
                sorted_vols[idx.min(sorted_vols.len() - 1)]
            })
            .collect();

        volatilities
            .iter()
            .map(|&v| {
                if v == 0.0 {
                    0
                } else {
                    let state = thresholds.iter().filter(|&&t| v > t).count();
                    state.min(self.n_states - 1)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn create_trending_bars(count: usize, trend: f64) -> Vec<Bar> {
        let mut price = 100.0;
        (0..count)
            .map(|i| {
                price *= 1.0 + trend;
                let noise = (i as f64 * 0.1).sin() * 0.5;
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    price - 1.0 + noise,
                    price + 2.0,
                    price - 2.0,
                    price + noise,
                    1000000.0 + (i as f64 * 1000.0).sin() * 100000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_regime_to_numeric() {
        assert!((Regime::StrongUptrend.to_numeric() - 2.0).abs() < 0.001);
        assert!((Regime::Ranging.to_numeric() - 0.0).abs() < 0.001);
        assert!((Regime::StrongDowntrend.to_numeric() - (-2.0)).abs() < 0.001);
    }

    #[test]
    fn test_regime_is_bullish() {
        assert!(Regime::StrongUptrend.is_bullish());
        assert!(Regime::WeakUptrend.is_bullish());
        assert!(!Regime::Ranging.is_bullish());
        assert!(!Regime::WeakDowntrend.is_bullish());
    }

    #[test]
    fn test_regime_is_trending() {
        assert!(Regime::StrongUptrend.is_trending());
        assert!(Regime::WeakDowntrend.is_trending());
        assert!(!Regime::Ranging.is_trending());
    }

    #[test]
    fn test_regime_label_to_features() {
        let label = RegimeLabel {
            trend: Regime::StrongUptrend,
            volatility: VolatilityRegime::High,
            volume: VolumeRegime::Normal,
            adx: Some(45.0),
            confidence: None,
        };

        let features = label.to_features();
        assert!((features["regime_trend"] - 2.0).abs() < 0.001);
        assert!((features["regime_is_bullish"] - 1.0).abs() < 0.001);
        assert!((features["regime_adx"] - 45.0).abs() < 0.001);
    }

    #[test]
    fn test_regime_detector_uptrend() {
        let bars = create_trending_bars(150, 0.005); // 0.5% daily gain

        let detector = RegimeDetector::new(RegimeConfig::default());
        let regimes = detector.detect(&bars);

        assert_eq!(regimes.len(), bars.len());

        // Later bars should show uptrend
        let last_regime = regimes.last().unwrap();
        assert!(
            last_regime.trend == Regime::StrongUptrend
                || last_regime.trend == Regime::WeakUptrend
                || last_regime.trend == Regime::Ranging
        );
    }

    #[test]
    fn test_regime_detector_downtrend() {
        let bars = create_trending_bars(150, -0.005); // 0.5% daily loss

        let detector = RegimeDetector::new(RegimeConfig::default());
        let regimes = detector.detect(&bars);

        // Later bars should show downtrend or ranging
        let last_regime = regimes.last().unwrap();
        assert!(
            last_regime.trend == Regime::StrongDowntrend
                || last_regime.trend == Regime::WeakDowntrend
                || last_regime.trend == Regime::Ranging
        );
    }

    #[test]
    fn test_detect_regime_changes() {
        let regimes = vec![
            RegimeLabel {
                trend: Regime::Ranging,
                volatility: VolatilityRegime::Normal,
                volume: VolumeRegime::Normal,
                adx: None,
                confidence: None,
            },
            RegimeLabel {
                trend: Regime::StrongUptrend,
                volatility: VolatilityRegime::High,
                volume: VolumeRegime::High,
                adx: None,
                confidence: None,
            },
            RegimeLabel {
                trend: Regime::StrongUptrend,
                volatility: VolatilityRegime::High,
                volume: VolumeRegime::Normal,
                adx: None,
                confidence: None,
            },
            RegimeLabel {
                trend: Regime::Ranging,
                volatility: VolatilityRegime::Normal,
                volume: VolumeRegime::Low,
                adx: None,
                confidence: None,
            },
        ];

        let changes = detect_regime_changes(&regimes);
        assert_eq!(changes.len(), 2); // Ranging->UpTrend, UpTrend->Ranging
    }

    #[test]
    fn test_regime_stats() {
        let regimes = vec![
            RegimeLabel {
                trend: Regime::StrongUptrend,
                volatility: VolatilityRegime::Normal,
                volume: VolumeRegime::Normal,
                adx: None,
                confidence: None,
            };
            100
        ];

        let stats = RegimeStats::from_regimes(&regimes);

        assert_eq!(stats.regime_counts.get("StrongUptrend"), Some(&100));
        assert!((stats.regime_percentages.get("StrongUptrend").unwrap() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_hmm_regime_detector() {
        let bars = create_trending_bars(200, 0.001);

        let hmm = HMMRegimeDetector::new(3, 20);
        let states = hmm.detect(&bars);

        assert_eq!(states.len(), bars.len());

        // States should be between 0 and n_states-1
        for &state in &states {
            assert!(state < 3);
        }
    }

    #[test]
    fn test_volatility_regime() {
        assert!((VolatilityRegime::VeryHigh.to_numeric() - 2.0).abs() < 0.001);
        assert!((VolatilityRegime::Normal.to_numeric() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_volume_regime() {
        assert!((VolumeRegime::VeryHigh.to_numeric() - 2.0).abs() < 0.001);
        assert!((VolumeRegime::Normal.to_numeric() - 0.0).abs() < 0.001);
    }
}

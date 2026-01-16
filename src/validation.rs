//! Signal validation module.
//!
//! This module provides comprehensive validation for signals used in backtesting,
//! with helpful error messages that include common causes and quick fixes.
//!
//! # Overview
//!
//! Before running a backtest with external signals, it's critical to validate:
//! - Signal length matches data length
//! - No NaN or infinite values
//! - No suspicious patterns (constant, extreme values)
//! - No obvious lookahead bias indicators
//!
//! # Example
//!
//! ```ignore
//! use mantis::validation::{validate_signal, SignalValidationConfig};
//!
//! let signals = vec![0.1, -0.2, 0.3, 0.0, -0.1];
//! let data_len = 5;
//!
//! // Validate with default settings
//! validate_signal(&signals, data_len, &SignalValidationConfig::default())?;
//!
//! // Validate with strict settings (warnings become errors)
//! let strict = SignalValidationConfig::strict();
//! validate_signal(&signals, data_len, &strict)?;
//! ```

use crate::error::{BacktestError, Result};
use tracing::warn;

/// Configuration for signal validation.
#[derive(Debug, Clone)]
pub struct SignalValidationConfig {
    /// Whether to check for NaN values (default: true).
    pub check_nan: bool,
    /// Whether to check for infinite values (default: true).
    pub check_infinite: bool,
    /// Whether to check for constant signals (default: true).
    pub check_constant: bool,
    /// Whether to check for extreme values (default: true).
    pub check_extreme: bool,
    /// Threshold for extreme values (default: 100.0).
    /// Values with absolute value > threshold are considered extreme.
    pub extreme_threshold: f64,
    /// Maximum allowed fraction of extreme values (default: 0.01 = 1%).
    pub max_extreme_fraction: f64,
    /// Treat warnings as errors (default: false).
    /// When true, suspicious patterns cause errors instead of warnings.
    pub strict: bool,
    /// Allow signals with length shorter than data (will be padded with 0).
    /// When true, validates that signal_len <= data_len rather than ==.
    pub allow_shorter_signal: bool,
}

impl Default for SignalValidationConfig {
    fn default() -> Self {
        Self {
            check_nan: true,
            check_infinite: true,
            check_constant: true,
            check_extreme: true,
            extreme_threshold: 100.0,
            max_extreme_fraction: 0.01,
            strict: false,
            allow_shorter_signal: false,
        }
    }
}

impl SignalValidationConfig {
    /// Create a strict validation config that treats warnings as errors.
    pub fn strict() -> Self {
        Self {
            strict: true,
            ..Default::default()
        }
    }

    /// Create a lenient config that only checks for fatal issues.
    pub fn lenient() -> Self {
        Self {
            check_nan: true,
            check_infinite: true,
            check_constant: false,
            check_extreme: false,
            strict: false,
            allow_shorter_signal: true,
            ..Default::default()
        }
    }

    /// Create a config suitable for ML model outputs (allows shorter signals).
    pub fn for_ml() -> Self {
        Self {
            allow_shorter_signal: true,
            ..Default::default()
        }
    }
}

/// Result of signal validation containing any warnings encountered.
#[derive(Debug, Clone, Default)]
pub struct SignalValidationResult {
    /// Warnings that were generated during validation.
    pub warnings: Vec<String>,
    /// Statistics about the signal.
    pub stats: SignalStats,
}

/// Statistics about a signal array.
#[derive(Debug, Clone, Default)]
pub struct SignalStats {
    /// Number of elements in the signal.
    pub length: usize,
    /// Number of NaN values.
    pub nan_count: usize,
    /// Number of infinite values.
    pub infinite_count: usize,
    /// Minimum value (excluding NaN/inf).
    pub min: f64,
    /// Maximum value (excluding NaN/inf).
    pub max: f64,
    /// Mean value (excluding NaN/inf).
    pub mean: f64,
    /// Standard deviation (excluding NaN/inf).
    pub std_dev: f64,
    /// Number of unique values.
    pub unique_count: usize,
    /// Number of extreme values (|x| > threshold).
    pub extreme_count: usize,
}

impl SignalStats {
    /// Calculate statistics from a signal array.
    pub fn from_signal(signal: &[f64], extreme_threshold: f64) -> Self {
        let length = signal.len();
        let mut nan_count = 0;
        let mut infinite_count = 0;
        let mut extreme_count = 0;
        let mut valid_values: Vec<f64> = Vec::with_capacity(length);

        for &v in signal {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                infinite_count += 1;
            } else {
                valid_values.push(v);
                if v.abs() > extreme_threshold {
                    extreme_count += 1;
                }
            }
        }

        let valid_len = valid_values.len();
        let (min, max, mean, std_dev) = if valid_len > 0 {
            let min = valid_values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = valid_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = valid_values.iter().sum();
            let mean = sum / valid_len as f64;

            let variance: f64 = valid_values
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / valid_len as f64;
            let std_dev = variance.sqrt();

            (min, max, mean, std_dev)
        } else {
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
        };

        // Count unique values (with some tolerance for floating point)
        let mut sorted = valid_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let unique_count = if sorted.is_empty() {
            0
        } else {
            let mut count = 1;
            let epsilon = 1e-10;
            for i in 1..sorted.len() {
                if (sorted[i] - sorted[i - 1]).abs() > epsilon {
                    count += 1;
                }
            }
            count
        };

        Self {
            length,
            nan_count,
            infinite_count,
            min,
            max,
            mean,
            std_dev,
            unique_count,
            extreme_count,
        }
    }
}

/// Validate a signal array against data length and check for issues.
///
/// Returns Ok(SignalValidationResult) with any warnings if validation passes,
/// or Err(BacktestError) with helpful diagnostics if validation fails.
///
/// # Arguments
///
/// * `signal` - The signal array to validate
/// * `data_len` - The length of the data array the signal will be used with
/// * `config` - Validation configuration
///
/// # Errors
///
/// Returns an error with helpful diagnostics for:
/// - Signal length mismatch
/// - NaN values in signal
/// - Infinite values in signal
/// - Constant signal (if strict mode)
/// - Extreme values (if strict mode)
///
/// # Example
///
/// ```ignore
/// let signals = vec![0.1, -0.2, 0.3];
/// let result = validate_signal(&signals, 3, &SignalValidationConfig::default())?;
/// for warning in &result.warnings {
///     println!("Warning: {}", warning);
/// }
/// ```
pub fn validate_signal(
    signal: &[f64],
    data_len: usize,
    config: &SignalValidationConfig,
) -> Result<SignalValidationResult> {
    let mut result = SignalValidationResult::default();

    // Check for empty signal
    if signal.is_empty() {
        return Err(BacktestError::InvalidSignal {
            index: 0,
            value_description: "Signal array is empty (0 elements)".to_string(),
            help: crate::error::ErrorHelp {
                common_causes: vec![
                    "Model returned empty predictions",
                    "Signal array was not populated",
                    "Filter removed all data points",
                ],
                quick_fixes: vec![
                    "Ensure your signal has at least one value for each bar in the data",
                    "Check that your model's predict() method returns values",
                    "Verify that filtering conditions don't exclude all data",
                ],
            },
        });
    }

    // Check length mismatch
    if config.allow_shorter_signal {
        if signal.len() > data_len {
            return Err(BacktestError::signal_shape_mismatch(signal.len(), data_len));
        }
    } else if signal.len() != data_len {
        return Err(BacktestError::signal_shape_mismatch(signal.len(), data_len));
    }

    // Calculate statistics
    let stats = SignalStats::from_signal(signal, config.extreme_threshold);
    result.stats = stats.clone();

    // Check for NaN values
    if config.check_nan && stats.nan_count > 0 {
        // Find first NaN index for error message
        let first_nan_idx = signal.iter().position(|x| x.is_nan()).unwrap_or(0);
        return Err(BacktestError::signal_nan(
            first_nan_idx,
            stats.nan_count,
            stats.length,
        ));
    }

    // Check for infinite values
    if config.check_infinite && stats.infinite_count > 0 {
        // Find first infinite value for error message
        let (idx, &value) = signal
            .iter()
            .enumerate()
            .find(|(_, x)| x.is_infinite())
            .unwrap();
        return Err(BacktestError::signal_infinite(idx, value));
    }

    // Check for constant signal
    if config.check_constant && stats.unique_count == 1 && stats.length > 1 {
        let const_value = signal.iter().find(|x| !x.is_nan()).cloned().unwrap_or(0.0);
        if config.strict {
            return Err(BacktestError::signal_constant(const_value, stats.length));
        } else {
            let msg = format!(
                "Signal is constant (value={:.6}) across all {} rows",
                const_value, stats.length
            );
            warn!("{}", msg);
            result.warnings.push(msg);
        }
    }

    // Check for extreme values
    if config.check_extreme && stats.length > 0 {
        let extreme_fraction = stats.extreme_count as f64 / stats.length as f64;
        if extreme_fraction > config.max_extreme_fraction {
            if config.strict {
                return Err(BacktestError::signal_extreme_values(
                    stats.min,
                    stats.max,
                    stats.extreme_count,
                ));
            } else {
                let msg = format!(
                    "Signal has {} extreme values ({:.1}% > threshold {})",
                    stats.extreme_count,
                    extreme_fraction * 100.0,
                    config.extreme_threshold
                );
                warn!("{}", msg);
                result.warnings.push(msg);
            }
        }
    }

    Ok(result)
}

/// Validate multiple signals for a multi-asset backtest.
///
/// # Arguments
///
/// * `signals` - Map of symbol to signal array
/// * `data_lens` - Map of symbol to data length
/// * `config` - Validation configuration
pub fn validate_signals<S: AsRef<str>>(
    signals: &[(S, &[f64])],
    data_lens: &[(S, usize)],
    config: &SignalValidationConfig,
) -> Result<Vec<(String, SignalValidationResult)>> {
    let data_len_map: std::collections::HashMap<&str, usize> =
        data_lens.iter().map(|(s, l)| (s.as_ref(), *l)).collect();

    let mut results = Vec::with_capacity(signals.len());

    for (symbol, signal) in signals {
        let symbol_str = symbol.as_ref();
        let data_len = *data_len_map.get(symbol_str).ok_or_else(|| {
            BacktestError::DataError(format!(
                "No data length provided for symbol: {}",
                symbol_str
            ))
        })?;

        let validation_result = validate_signal(signal, data_len, config)?;
        results.push((symbol_str.to_string(), validation_result));
    }

    Ok(results)
}

/// Quick validation that only checks for fatal issues (NaN, infinity, length mismatch, empty).
///
/// Use this for performance-critical paths where you want minimal overhead.
pub fn validate_signal_quick(signal: &[f64], data_len: usize) -> Result<()> {
    // Empty signal check
    if signal.is_empty() {
        return Err(BacktestError::InvalidSignal {
            index: 0,
            value_description: "Signal array is empty (0 elements)".to_string(),
            help: crate::error::ErrorHelp {
                common_causes: vec![
                    "Model returned empty predictions",
                    "Signal array was not populated",
                    "Filter removed all data points",
                ],
                quick_fixes: vec![
                    "Ensure your signal has at least one value for each bar in the data",
                    "Check that your model's predict() method returns values",
                    "Verify that filtering conditions don't exclude all data",
                ],
            },
        });
    }

    // Length check
    if signal.len() != data_len {
        return Err(BacktestError::signal_shape_mismatch(signal.len(), data_len));
    }

    // NaN/Infinity check (single pass)
    for (i, &v) in signal.iter().enumerate() {
        if v.is_nan() {
            return Err(BacktestError::signal_nan(i, 1, signal.len()));
        }
        if v.is_infinite() {
            return Err(BacktestError::signal_infinite(i, v));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_signal() {
        let signal = vec![0.1, -0.2, 0.3, 0.0, -0.1];
        let result = validate_signal(&signal, 5, &SignalValidationConfig::default());
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.warnings.is_empty());
        assert_eq!(result.stats.length, 5);
        assert_eq!(result.stats.nan_count, 0);
        assert_eq!(result.stats.infinite_count, 0);
    }

    #[test]
    fn test_length_mismatch() {
        let signal = vec![0.1, -0.2, 0.3];
        let result = validate_signal(&signal, 5, &SignalValidationConfig::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BacktestError::SignalShapeMismatch { .. }));
        // Check error message contains expected info
        let msg = format!("{}", err);
        assert!(msg.contains("3 rows"));
        assert!(msg.contains("5 rows"));
        assert!(msg.contains("Common causes"));
        assert!(msg.contains("Quick fix"));
    }

    #[test]
    fn test_nan_detection() {
        let signal = vec![0.1, f64::NAN, 0.3];
        let result = validate_signal(&signal, 3, &SignalValidationConfig::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BacktestError::InvalidSignal { .. }));
        let msg = format!("{}", err);
        assert!(msg.contains("NaN"));
    }

    #[test]
    fn test_infinity_detection() {
        let signal = vec![0.1, f64::INFINITY, 0.3];
        let result = validate_signal(&signal, 3, &SignalValidationConfig::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BacktestError::InvalidSignal { .. }));
        let msg = format!("{}", err);
        assert!(msg.contains("infinity"));
    }

    #[test]
    fn test_negative_infinity_detection() {
        let signal = vec![0.1, f64::NEG_INFINITY, 0.3];
        let result = validate_signal(&signal, 3, &SignalValidationConfig::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("negative infinity"));
    }

    #[test]
    fn test_empty_signal_detection() {
        let signal: Vec<f64> = vec![];
        let result = validate_signal(&signal, 0, &SignalValidationConfig::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BacktestError::InvalidSignal { .. }));
        let msg = format!("{}", err);
        assert!(msg.contains("empty"));
        assert!(msg.contains("Common causes"));
        assert!(msg.contains("Quick fix"));
    }

    #[test]
    fn test_constant_signal_warning() {
        let signal = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let result = validate_signal(&signal, 5, &SignalValidationConfig::default());
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.warnings.is_empty());
        assert!(result.warnings[0].contains("constant"));
    }

    #[test]
    fn test_constant_signal_strict() {
        let signal = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let result = validate_signal(&signal, 5, &SignalValidationConfig::strict());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BacktestError::SuspiciousSignal { .. }));
    }

    #[test]
    fn test_extreme_values_warning() {
        let mut signal = vec![0.1; 100];
        signal[0] = 1000.0; // 1% extreme
        signal[1] = -1000.0; // 2% extreme
        let result = validate_signal(&signal, 100, &SignalValidationConfig::default());
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.warnings.is_empty());
        assert!(result.warnings[0].contains("extreme"));
    }

    #[test]
    fn test_allow_shorter_signal() {
        let signal = vec![0.1, -0.2, 0.3];
        let config = SignalValidationConfig::for_ml();
        let result = validate_signal(&signal, 5, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quick_validation() {
        let signal = vec![0.1, -0.2, 0.3];
        assert!(validate_signal_quick(&signal, 3).is_ok());
        assert!(validate_signal_quick(&signal, 5).is_err());

        let nan_signal = vec![0.1, f64::NAN, 0.3];
        assert!(validate_signal_quick(&nan_signal, 3).is_err());
    }

    #[test]
    fn test_signal_stats() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = SignalStats::from_signal(&signal, 100.0);
        assert_eq!(stats.length, 5);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.infinite_count, 0);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert_eq!(stats.unique_count, 5);
        assert_eq!(stats.extreme_count, 0);
    }

    #[test]
    fn test_error_has_help() {
        let err = BacktestError::signal_shape_mismatch(100, 200);
        assert!(err.has_help());

        let err = BacktestError::DataError("test".to_string());
        assert!(!err.has_help());
    }
}

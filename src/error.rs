//! Error types for the backtest engine.
//!
//! This module provides structured error types with helpful diagnostics
//! including common causes and quick fix suggestions.

use std::fmt;
use thiserror::Error;

/// Structured help message for errors.
///
/// Provides common causes and quick fixes to help users diagnose and resolve issues.
#[derive(Debug, Clone)]
pub struct ErrorHelp {
    /// Common causes of this error.
    pub common_causes: Vec<&'static str>,
    /// Quick fix suggestions.
    pub quick_fixes: Vec<&'static str>,
}

impl fmt::Display for ErrorHelp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.common_causes.is_empty() {
            writeln!(f, "  Common causes:")?;
            for cause in &self.common_causes {
                writeln!(f, "    • {}", cause)?;
            }
        }
        if !self.quick_fixes.is_empty() {
            writeln!(f, "  Quick fix:")?;
            for fix in &self.quick_fixes {
                writeln!(f, "    → {}", fix)?;
            }
        }
        Ok(())
    }
}

/// Main error type for the backtest engine.
#[derive(Error, Debug)]
pub enum BacktestError {
    #[error("Data error: {0}")]
    DataError(String),

    #[error("CSV parsing error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Date parsing error: {0}")]
    DateParseError(#[from] chrono::ParseError),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("Strategy error: {0}")]
    StrategyError(String),

    #[error("Insufficient funds: required {required}, available {available}")]
    InsufficientFunds { required: f64, available: f64 },

    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    #[error("No data loaded")]
    NoData,

    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("TOML parsing error: {0}")]
    TomlError(#[from] toml::de::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Portfolio constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Margin call ({reason}): equity {equity:.2} < required {requirement:.2}")]
    MarginCall {
        equity: f64,
        requirement: f64,
        reason: String,
    },

    #[error("Database error: {0}")]
    DatabaseError(String),

    // ============================================================
    // Helpful Error Types (with structured diagnostics)
    // ============================================================

    /// Signal shape mismatch between signal array and data.
    #[error("Signal shape mismatch: signal has {signal_len} rows but data has {data_len} rows\n{}", .help)]
    SignalShapeMismatch {
        signal_len: usize,
        data_len: usize,
        help: ErrorHelp,
    },

    /// Invalid signal values detected (NaN, infinity, or out of range).
    #[error("Invalid signal at index {index}: {value_description}\n{}", .help)]
    InvalidSignal {
        index: usize,
        value_description: String,
        help: ErrorHelp,
    },

    /// Potential lookahead bias detected.
    #[error("Potential lookahead bias detected: {description}\n{}", .help)]
    LookaheadBias {
        description: String,
        help: ErrorHelp,
    },

    /// Signal contains suspicious values that may indicate data issues.
    #[error("Suspicious signal detected: {description}\n{}", .help)]
    SuspiciousSignal {
        description: String,
        help: ErrorHelp,
    },
}

impl BacktestError {
    /// Create a SignalShapeMismatch error with helpful diagnostics.
    pub fn signal_shape_mismatch(signal_len: usize, data_len: usize) -> Self {
        let help = ErrorHelp {
            common_causes: vec![
                "Weekend/holiday filtering mismatch between signal and data",
                "Date alignment issues (different start/end dates)",
                "Signal generated from different data source than backtest data",
                "Warmup period not accounted for in signal array",
            ],
            quick_fixes: vec![
                "Use `signal.reindex(data.index, fill_value=0)` to match indices",
                "Ensure signal and data cover the same date range",
                "Check that both use the same holiday calendar",
                "Pad signal with zeros for warmup period if needed",
            ],
        };
        Self::SignalShapeMismatch {
            signal_len,
            data_len,
            help,
        }
    }

    /// Create an InvalidSignal error for NaN values.
    pub fn signal_nan(index: usize, nan_count: usize, total_len: usize) -> Self {
        let help = ErrorHelp {
            common_causes: vec![
                "Missing data in source features caused NaN propagation",
                "Division by zero in signal calculation",
                "Indicator warmup period not handled (e.g., first N bars of SMA)",
                "Forward-fill or interpolation not applied to gaps",
            ],
            quick_fixes: vec![
                "Replace NaN with 0: `signal = signal.fillna(0)` or `signal[signal.isnan()] = 0`",
                "Drop rows with NaN: `signal = signal.dropna()`",
                "Check indicator warmup: exclude first N bars where N = longest lookback",
                "Use `df.ffill()` to forward-fill missing values before computing signal",
            ],
        };
        Self::InvalidSignal {
            index,
            value_description: format!(
                "NaN value (found {} NaN values out of {} total)",
                nan_count, total_len
            ),
            help,
        }
    }

    /// Create an InvalidSignal error for infinite values.
    pub fn signal_infinite(index: usize, value: f64) -> Self {
        let help = ErrorHelp {
            common_causes: vec![
                "Division by zero (e.g., volatility = 0, volume = 0)",
                "Log of zero or negative number",
                "Exponential overflow in calculation",
                "Numeric precision issues in long calculations",
            ],
            quick_fixes: vec![
                "Clip extreme values: `signal = signal.clip(-10, 10)`",
                "Add small epsilon to denominators: `x / (vol + 1e-8)`",
                "Check for zero values before division",
                "Use `np.nan_to_num(signal, nan=0, posinf=0, neginf=0)`",
            ],
        };
        let sign = if value > 0.0 { "positive" } else { "negative" };
        Self::InvalidSignal {
            index,
            value_description: format!("{} infinity", sign),
            help,
        }
    }

    /// Create a LookaheadBias error.
    pub fn lookahead_bias(description: impl Into<String>) -> Self {
        let help = ErrorHelp {
            common_causes: vec![
                "Using close price to generate signals that execute at close",
                "Signal uses future data (e.g., tomorrow's return in today's signal)",
                "Point-in-time data not enforced (using restated financials)",
                "Data preprocessing used full dataset statistics",
            ],
            quick_fixes: vec![
                "Ensure signals execute at NEXT bar's open, not current bar",
                "Use only data available at signal generation time",
                "Implement point-in-time database or as-reported data",
                "Use expanding window statistics, not full-sample",
            ],
        };
        Self::LookaheadBias {
            description: description.into(),
            help,
        }
    }

    /// Create a SuspiciousSignal error for constant signals.
    pub fn signal_constant(value: f64, length: usize) -> Self {
        let help = ErrorHelp {
            common_causes: vec![
                "Signal calculation returned same value for all rows",
                "Bug in signal generation code",
                "Data not varying (e.g., all prices identical)",
                "Wrong column used for signal calculation",
            ],
            quick_fixes: vec![
                "Verify signal calculation logic produces varying outputs",
                "Check that input data has expected variation",
                "Print signal statistics to debug: `print(signal.describe())`",
            ],
        };
        Self::SuspiciousSignal {
            description: format!(
                "Signal is constant (value={:.6}) across all {} rows",
                value, length
            ),
            help,
        }
    }

    /// Create a SuspiciousSignal error for signals with extreme values.
    pub fn signal_extreme_values(min: f64, max: f64, extreme_count: usize) -> Self {
        let help = ErrorHelp {
            common_causes: vec![
                "Signal not normalized/standardized",
                "Outliers in input data",
                "Numeric overflow in calculation",
                "Units mismatch (e.g., returns vs prices)",
            ],
            quick_fixes: vec![
                "Normalize signal: `(signal - signal.mean()) / signal.std()`",
                "Winsorize extreme values: `signal.clip(lower=q01, upper=q99)`",
                "Apply rank transformation: `signal.rank(pct=True)`",
                "Check units are consistent (returns should be small decimals)",
            ],
        };
        Self::SuspiciousSignal {
            description: format!(
                "Signal has {} extreme values (min={:.2e}, max={:.2e})",
                extreme_count, min, max
            ),
            help,
        }
    }

    /// Check if this error has structured help information.
    pub fn has_help(&self) -> bool {
        matches!(
            self,
            BacktestError::SignalShapeMismatch { .. }
                | BacktestError::InvalidSignal { .. }
                | BacktestError::LookaheadBias { .. }
                | BacktestError::SuspiciousSignal { .. }
        )
    }
}

/// Result type alias for backtest operations.
pub type Result<T> = std::result::Result<T, BacktestError>;

//! Error types for the backtest engine.

use thiserror::Error;

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
}

/// Result type alias for backtest operations.
pub type Result<T> = std::result::Result<T, BacktestError>;

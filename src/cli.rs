//! Command-line interface for the backtest engine.

use mantis::analytics::ResultFormatter;
use mantis::config::BacktestFileConfig;
use mantis::cost_sensitivity::{
    run_cost_sensitivity_analysis, CostSensitivityAnalysis, CostSensitivityConfig,
};
use mantis::data::{
    data_quality_report, load_csv, load_data, load_parquet, resample, DataConfig, ResampleInterval,
};
use mantis::engine::{BacktestConfig, Engine};
use mantis::error::Result;
use mantis::experiments::{
    default_store_path, ensure_store_directory, ExperimentFilter, ExperimentRecord, ExperimentStore,
};
use mantis::features::{FeatureConfig, FeatureExtractor, TimeSeriesSplitter};
use mantis::monte_carlo::{MonteCarloConfig, MonteCarloResult, MonteCarloSimulator};
use mantis::portfolio::{CostModel, MarginConfig};
use mantis::strategies::{
    BreakoutStrategy, MacdStrategy, MeanReversion, MomentumStrategy, RsiStrategy, SmaCrossover,
};
use mantis::strategy::Strategy;
use mantis::types::{AssetClass, AssetConfig, ExecutionPrice, LotSelectionMethod};
use mantis::walkforward::{
    WalkForwardAnalyzer, WalkForwardConfig, WalkForwardMetric, WalkForwardResult,
};

use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;
use tabled::{builder::Builder, settings::Style};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// Mantis - A high-performance backtesting engine for quantitative trading.
#[derive(Parser)]
#[command(name = "mantis")]
#[command(author = "Johan")]
#[command(version = "1.0.0")]
#[command(about = "A high-performance backtesting engine for trading strategies")]
#[command(long_about = None)]
pub struct Cli {
    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Output format
    #[arg(short, long, value_enum, default_value = "text")]
    pub output: OutputFormat,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run a backtest with a built-in strategy
    Run {
        /// Path to data file (CSV or Parquet)
        #[arg(short, long)]
        data: PathBuf,

        /// Symbol name
        #[arg(short, long, default_value = "SYMBOL")]
        symbol: String,

        /// Strategy to use
        #[arg(short = 'S', long, value_enum, default_value = "sma-crossover")]
        strategy: StrategyType,

        /// Initial capital
        #[arg(short, long, default_value = "100000")]
        capital: f64,

        /// Position size as fraction of equity (0.0-1.0)
        #[arg(short, long, default_value = "0.1")]
        position_size: f64,

        /// Position sizing method (percent, fixed, volatility, signal, risk)
        #[arg(long, value_enum, default_value = "percent")]
        sizing_method: PositionSizingMethodArg,

        /// Fixed dollar amount for 'fixed' sizing method
        #[arg(long)]
        fixed_dollar: Option<f64>,

        /// Target volatility for 'volatility' sizing method (e.g., 0.15 for 15%)
        #[arg(long)]
        target_vol: Option<f64>,

        /// Volatility lookback period for 'volatility' sizing method
        #[arg(long, default_value = "20")]
        vol_lookback: usize,

        /// Risk per trade for 'risk' sizing method (e.g., 0.01 for 1%)
        #[arg(long)]
        risk_per_trade: Option<f64>,

        /// ATR multiplier for stop distance in 'risk' sizing method
        #[arg(long, default_value = "2.0")]
        stop_atr: f64,

        /// ATR period for 'risk' sizing method
        #[arg(long, default_value = "14")]
        atr_period: usize,

        /// Commission percentage (e.g., 0.1 for 0.1%)
        #[arg(long, default_value = "0.1")]
        commission: f64,

        /// Slippage percentage (e.g., 0.1 for 0.1%)
        #[arg(long, default_value = "0.1")]
        slippage: f64,

        /// Maximum volume participation rate (0.0-1.0, e.g., 0.10 = 10% of bar volume)
        #[arg(long)]
        max_volume_participation: Option<f64>,

        /// Execution price model for market orders
        #[arg(long, value_enum, default_value = "open")]
        execution_price: ExecutionPriceArg,

        /// Probability that an order fully fills on each attempt (0.0 - 1.0)
        #[arg(long, default_value = "1.0")]
        fill_probability: f64,

        /// Lifetime of pending limit orders in bars (0 = good-till-cancelled)
        #[arg(long, default_value = "5")]
        limit_order_ttl: usize,

        /// Tax-lot selection policy when closing positions
        #[arg(long = "lot-selection", value_enum, default_value = "fifo")]
        lot_selection: LotSelectionArg,

        /// Allow short selling
        #[arg(long)]
        allow_short: bool,

        /// Maximum leverage (gross exposure / equity)
        #[arg(long = "max-leverage", default_value = "2.0")]
        max_leverage: f64,

        /// Reg T initial margin for long positions (decimal fraction)
        #[arg(long = "regt-long", default_value = "0.5")]
        reg_t_long: f64,

        /// Reg T initial margin for short positions (decimal fraction)
        #[arg(long = "regt-short", default_value = "1.5")]
        reg_t_short: f64,

        /// Maintenance margin for long positions
        #[arg(long = "maintenance-long", default_value = "0.25")]
        maintenance_long: f64,

        /// Maintenance margin for short positions
        #[arg(long = "maintenance-short", default_value = "0.30")]
        maintenance_short: f64,

        /// Annualized interest rate charged on margin borrowing
        #[arg(long = "margin-interest", default_value = "0.03")]
        margin_interest: f64,

        /// Use portfolio margin instead of Reg T
        #[arg(long = "use-portfolio-margin")]
        use_portfolio_margin: bool,

        /// Portfolio margin percentage applied to gross exposure
        #[arg(long = "portfolio-margin-pct", default_value = "0.15")]
        portfolio_margin_pct: f64,

        /// Disable margin (require full cash for purchases)
        #[arg(long = "disable-margin")]
        disable_margin: bool,

        /// Fast MA period (for SMA strategy)
        #[arg(long, default_value = "10")]
        fast_period: usize,

        /// Slow MA period (for SMA strategy)
        #[arg(long, default_value = "30")]
        slow_period: usize,

        /// RSI period
        #[arg(long, default_value = "14")]
        rsi_period: usize,

        /// Momentum lookback period
        #[arg(long, default_value = "20")]
        lookback: usize,

        /// Data file format (auto-detects from extension if not specified)
        #[arg(short, long, value_enum, default_value = "auto")]
        format: DataFormatArg,

        /// Asset class for the provided data (equity, future, crypto, forex, option)
        #[arg(long, value_enum, default_value = "equity")]
        asset_class: AssetClassArg,

        /// Contract or option multiplier (futures/options)
        #[arg(long, default_value = "1.0")]
        multiplier: f64,

        /// Minimum tick size for prices (futures)
        #[arg(long, default_value = "0.01")]
        tick_size: f64,

        /// Margin requirement as a decimal fraction (e.g., 0.1 for 10%)
        #[arg(long, default_value = "0.1")]
        margin_requirement: f64,

        /// Maximum decimals for crypto base asset quantity
        #[arg(long, default_value = "8")]
        base_precision: u8,

        /// Maximum decimals for crypto quote prices
        #[arg(long, default_value = "2")]
        quote_precision: u8,

        /// Pip size for forex pairs (e.g., 0.0001)
        #[arg(long, default_value = "0.0001")]
        pip_size: f64,

        /// Lot size for forex positions
        #[arg(long, default_value = "100000")]
        lot_size: f64,

        /// Underlying symbol for options (defaults to same as symbol)
        #[arg(long)]
        underlying: Option<String>,

        /// Random seed for reproducible execution (None = deterministic from timestamps)
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Run walk-forward optimization with rolling windows
    WalkForward {
        /// Path to data file (CSV or Parquet)
        #[arg(short, long)]
        data: PathBuf,

        /// Symbol name
        #[arg(short, long, default_value = "SYMBOL")]
        symbol: String,

        /// Strategy to evaluate
        #[arg(short = 'S', long, value_enum, default_value = "sma-crossover")]
        strategy: StrategyType,

        /// Initial capital
        #[arg(short, long, default_value = "100000")]
        capital: f64,

        /// Position size as fraction of equity (0.0-1.0)
        #[arg(short, long, default_value = "0.1")]
        position_size: f64,

        /// Commission percentage (e.g., 0.1 for 0.1%)
        #[arg(long, default_value = "0.1")]
        commission: f64,

        /// Slippage percentage (e.g., 0.1 for 0.1%)
        #[arg(long, default_value = "0.1")]
        slippage: f64,

        /// Maximum volume participation rate (0.0-1.0, e.g., 0.10 = 10% of bar volume)
        #[arg(long)]
        max_volume_participation: Option<f64>,

        /// Execution price model for market orders
        #[arg(long, value_enum, default_value = "open")]
        execution_price: ExecutionPriceArg,

        /// Probability that an order fully fills on each attempt (0.0 - 1.0)
        #[arg(long, default_value = "1.0")]
        fill_probability: f64,

        /// Lifetime of pending limit orders in bars (0 = good-till-cancelled)
        #[arg(long, default_value = "5")]
        limit_order_ttl: usize,

        /// Tax-lot selection policy when closing positions
        #[arg(long = "lot-selection", value_enum, default_value = "fifo")]
        lot_selection: LotSelectionArg,

        /// Allow short selling
        #[arg(long)]
        allow_short: bool,

        /// Maximum leverage (gross exposure / equity)
        #[arg(long = "max-leverage", default_value = "2.0")]
        max_leverage: f64,

        /// Reg T initial margin for long positions
        #[arg(long = "regt-long", default_value = "0.5")]
        reg_t_long: f64,

        /// Reg T initial margin for short positions
        #[arg(long = "regt-short", default_value = "1.5")]
        reg_t_short: f64,

        /// Maintenance margin requirement for longs
        #[arg(long = "maintenance-long", default_value = "0.25")]
        maintenance_long: f64,

        /// Maintenance margin requirement for shorts
        #[arg(long = "maintenance-short", default_value = "0.30")]
        maintenance_short: f64,

        /// Annualized interest rate on borrowed capital
        #[arg(long = "margin-interest", default_value = "0.03")]
        margin_interest: f64,

        /// Use portfolio margin instead of Reg T
        #[arg(long = "use-portfolio-margin")]
        use_portfolio_margin: bool,

        /// Portfolio margin percentage applied to exposure
        #[arg(long = "portfolio-margin-pct", default_value = "0.15")]
        portfolio_margin_pct: f64,

        /// Disable margin usage entirely
        #[arg(long = "disable-margin")]
        disable_margin: bool,

        /// Number of walk-forward windows/folds
        #[arg(long = "folds", default_value = "5")]
        folds: usize,

        /// Ratio of data allocated to in-sample optimization (0-1)
        #[arg(long, default_value = "0.7")]
        in_sample_ratio: f64,

        /// Use anchored windows instead of rolling
        #[arg(long)]
        anchored: bool,

        /// Minimum bars per window
        #[arg(long, default_value = "50")]
        min_bars: usize,

        /// Walk-forward optimization metric
        #[arg(long, value_enum, default_value = "sharpe")]
        metric: WalkForwardMetricArg,

        /// Data file format (auto-detects from extension if not specified)
        #[arg(short, long, value_enum, default_value = "auto")]
        format: DataFormatArg,
    },

    /// Optimize strategy parameters
    Optimize {
        /// Path to CSV data file
        #[arg(short, long)]
        data: PathBuf,

        /// Symbol name
        #[arg(short, long, default_value = "SYMBOL")]
        symbol: String,

        /// Strategy to optimize
        #[arg(short = 'S', long, value_enum, default_value = "sma-crossover")]
        strategy: StrategyType,

        /// Initial capital
        #[arg(short, long, default_value = "100000")]
        capital: f64,

        /// Metric to optimize
        #[arg(short, long, value_enum, default_value = "sharpe")]
        metric: OptimizeMetric,
    },

    /// Show information about available strategies
    Strategies,

    /// Validate a data file
    Validate {
        /// Path to CSV data file
        #[arg(short, long)]
        data: PathBuf,
    },

    /// Generate an example configuration file
    Init {
        /// Output path for config file
        #[arg(short, long, default_value = "backtest.toml")]
        output: PathBuf,
    },

    /// Run a backtest from a configuration file
    RunConfig {
        /// Path to TOML configuration file
        #[arg(short, long)]
        config: PathBuf,
    },

    /// Extract features for ML/DL model training
    Features {
        /// Path to CSV data file
        #[arg(short, long)]
        data: PathBuf,

        /// Output directory for feature files
        #[arg(short, long, default_value = "ml_data")]
        output_dir: PathBuf,

        /// Target prediction horizon (bars)
        #[arg(short, long, default_value = "5")]
        target_horizon: usize,

        /// Feature configuration: minimal, default, comprehensive
        #[arg(long, value_enum, default_value = "default")]
        feature_config: FeatureConfigType,

        /// Train/validation/test split ratio (train_pct)
        #[arg(long, default_value = "0.7")]
        train_ratio: f64,

        /// Validation ratio (of total data)
        #[arg(long, default_value = "0.15")]
        validation_ratio: f64,
    },

    /// Resample time-series data to a different interval
    Resample {
        /// Path to input data file (CSV or Parquet)
        #[arg(short, long)]
        input: PathBuf,

        /// Path to output file
        #[arg(short, long)]
        output: PathBuf,

        /// Target resampling interval
        #[arg(short = 'I', long, value_enum)]
        interval: ResampleIntervalArg,

        /// Data file format (auto-detects from extension if not specified)
        #[arg(short, long, value_enum, default_value = "auto")]
        format: DataFormatArg,
    },

    /// Generate a data quality report
    Quality {
        /// Path to data file (CSV or Parquet)
        #[arg(short, long)]
        data: PathBuf,

        /// Expected interval between bars in seconds (e.g., 60 for 1-minute, 86400 for daily)
        #[arg(short = 'I', long)]
        interval_seconds: i64,

        /// Data file format (auto-detects from extension if not specified)
        #[arg(short, long, value_enum, default_value = "auto")]
        format: DataFormatArg,
    },

    /// Run Monte Carlo simulation on a backtest for robustness testing
    MonteCarlo {
        /// Path to data file (CSV or Parquet)
        #[arg(short, long)]
        data: PathBuf,

        /// Symbol name
        #[arg(short, long, default_value = "SYMBOL")]
        symbol: String,

        /// Strategy to use
        #[arg(short = 'S', long, value_enum, default_value = "sma-crossover")]
        strategy: StrategyType,

        /// Initial capital
        #[arg(short, long, default_value = "100000")]
        capital: f64,

        /// Position size as fraction of equity (0.0-1.0)
        #[arg(short, long, default_value = "0.1")]
        position_size: f64,

        /// Commission percentage (e.g., 0.1 for 0.1%)
        #[arg(long, default_value = "0.1")]
        commission: f64,

        /// Slippage percentage (e.g., 0.1 for 0.1%)
        #[arg(long, default_value = "0.1")]
        slippage: f64,

        /// Number of Monte Carlo simulations
        #[arg(long, default_value = "1000")]
        simulations: usize,

        /// Confidence level for intervals (e.g., 0.95 for 95%)
        #[arg(long, default_value = "0.95")]
        confidence: f64,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,

        /// Use trade resampling (bootstrap with replacement)
        #[arg(long, default_value = "true")]
        resample_trades: bool,

        /// Shuffle returns instead of resampling (destroys autocorrelation)
        #[arg(long)]
        shuffle_returns: bool,

        /// Fast MA period (for SMA strategy)
        #[arg(long, default_value = "10")]
        fast_period: usize,

        /// Slow MA period (for SMA strategy)
        #[arg(long, default_value = "30")]
        slow_period: usize,

        /// RSI period
        #[arg(long, default_value = "14")]
        rsi_period: usize,

        /// Momentum lookback period
        #[arg(long, default_value = "20")]
        lookback: usize,

        /// Data file format (auto-detects from extension if not specified)
        #[arg(short, long, value_enum, default_value = "auto")]
        format: DataFormatArg,
    },

    /// Run cost sensitivity analysis to test strategy robustness to transaction costs
    CostSensitivity {
        /// Path to data file (CSV or Parquet)
        #[arg(short, long)]
        data: PathBuf,

        /// Symbol name
        #[arg(short, long, default_value = "SYMBOL")]
        symbol: String,

        /// Strategy to use
        #[arg(short = 'S', long, value_enum, default_value = "sma-crossover")]
        strategy: StrategyType,

        /// Initial capital
        #[arg(short, long, default_value = "100000")]
        capital: f64,

        /// Position size as fraction of equity (0.0-1.0)
        #[arg(short, long, default_value = "0.1")]
        position_size: f64,

        /// Commission percentage (e.g., 0.1 for 0.1%)
        #[arg(long, default_value = "0.1")]
        commission: f64,

        /// Slippage percentage (e.g., 0.1 for 0.1%)
        #[arg(long, default_value = "0.1")]
        slippage: f64,

        /// Cost multipliers to test (comma-separated, e.g., "1.0,2.0,5.0,10.0")
        #[arg(long, default_value = "1.0,2.0,5.0,10.0")]
        multipliers: String,

        /// Include zero-cost baseline (theoretical upper bound)
        #[arg(long)]
        include_zero_cost: bool,

        /// Minimum acceptable Sharpe ratio at 5x costs (robustness threshold)
        #[arg(long, default_value = "0.5")]
        robustness_threshold: f64,

        /// Fast MA period (for SMA strategy)
        #[arg(long, default_value = "10")]
        fast_period: usize,

        /// Slow MA period (for SMA strategy)
        #[arg(long, default_value = "30")]
        slow_period: usize,

        /// RSI period
        #[arg(long, default_value = "14")]
        rsi_period: usize,

        /// Momentum lookback period
        #[arg(long, default_value = "20")]
        lookback: usize,

        /// Data file format (auto-detects from extension if not specified)
        #[arg(short, long, value_enum, default_value = "auto")]
        format: DataFormatArg,
    },

    /// Experiment tracking and management
    #[command(subcommand)]
    Experiments(ExperimentsCommands),
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Csv,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum AssetClassArg {
    Equity,
    Future,
    Crypto,
    Forex,
    Option,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ExecutionPriceArg {
    Open,
    Close,
    Vwap,
    Twap,
    #[value(name = "random-in-range")]
    RandomInRange,
    Midpoint,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum LotSelectionArg {
    Fifo,
    Lifo,
    #[value(name = "highest-cost")]
    HighestCost,
    #[value(name = "lowest-cost")]
    LowestCost,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum StrategyType {
    SmaCrossover,
    Momentum,
    MeanReversion,
    Rsi,
    Breakout,
    Macd,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum OptimizeMetric {
    Sharpe,
    Sortino,
    Return,
    Calmar,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum WalkForwardMetricArg {
    Sharpe,
    Sortino,
    Return,
    Calmar,
    #[value(name = "profit-factor")]
    ProfitFactor,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum FeatureConfigType {
    Minimal,
    Default,
    Comprehensive,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum DataFormatArg {
    /// Auto-detect based on file extension
    Auto,
    /// Force CSV format
    Csv,
    /// Force Parquet format
    Parquet,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum, Default)]
#[value(rename_all = "kebab-case")]
pub enum PositionSizingMethodArg {
    /// Fixed percentage of equity (default)
    #[default]
    Percent,
    /// Fixed dollar amount per trade
    Fixed,
    /// Volatility-targeted sizing
    Volatility,
    /// Signal-scaled sizing
    Signal,
    /// Risk-based sizing using ATR
    Risk,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum ResampleIntervalArg {
    /// 1 minute
    #[value(name = "1m")]
    OneMinute,
    /// 5 minutes
    #[value(name = "5m")]
    FiveMinutes,
    /// 15 minutes
    #[value(name = "15m")]
    FifteenMinutes,
    /// 30 minutes
    #[value(name = "30m")]
    ThirtyMinutes,
    /// 1 hour
    #[value(name = "1h")]
    OneHour,
    /// 4 hours
    #[value(name = "4h")]
    FourHours,
    /// 1 day (daily)
    #[value(name = "1d")]
    OneDay,
    /// 1 week (weekly)
    #[value(name = "1w")]
    OneWeek,
    /// 1 month (monthly)
    #[value(name = "1M")]
    OneMonth,
}

impl ResampleIntervalArg {
    /// Convert CLI argument to ResampleInterval.
    fn to_interval(self) -> ResampleInterval {
        match self {
            ResampleIntervalArg::OneMinute => ResampleInterval::Minute(1),
            ResampleIntervalArg::FiveMinutes => ResampleInterval::Minute(5),
            ResampleIntervalArg::FifteenMinutes => ResampleInterval::Minute(15),
            ResampleIntervalArg::ThirtyMinutes => ResampleInterval::Minute(30),
            ResampleIntervalArg::OneHour => ResampleInterval::Hour(1),
            ResampleIntervalArg::FourHours => ResampleInterval::Hour(4),
            ResampleIntervalArg::OneDay => ResampleInterval::Day,
            ResampleIntervalArg::OneWeek => ResampleInterval::Week,
            ResampleIntervalArg::OneMonth => ResampleInterval::Month,
        }
    }
}

#[derive(Subcommand)]
pub enum ExperimentsCommands {
    /// List all experiments with optional filters
    List {
        /// Filter by strategy name (partial match)
        #[arg(long)]
        strategy: Option<String>,

        /// Filter by minimum Sharpe ratio
        #[arg(long)]
        min_sharpe: Option<f64>,

        /// Filter by maximum drawdown percentage
        #[arg(long)]
        max_drawdown: Option<f64>,

        /// Limit number of results
        #[arg(short, long)]
        limit: Option<usize>,

        /// Sort by field (sharpe_ratio, total_return, timestamp, max_drawdown, num_trades)
        #[arg(long, default_value = "timestamp")]
        sort_by: String,

        /// Sort descending
        #[arg(long)]
        desc: bool,
    },

    /// Show details for a specific experiment
    Show {
        /// Experiment ID
        id: String,
    },

    /// Compare two experiments side-by-side
    Compare {
        /// First experiment ID
        id1: String,
        /// Second experiment ID
        id2: String,
    },

    /// Add tags to an experiment
    Tag {
        /// Experiment ID
        id: String,
        /// Tags to add
        tags: Vec<String>,
    },

    /// Add notes to an experiment
    Note {
        /// Experiment ID
        id: String,
        /// Note text
        note: String,
    },

    /// Delete an experiment
    Delete {
        /// Experiment ID
        id: String,
    },
}

#[derive(Clone, Debug)]
enum StrategyParam {
    Sma {
        fast: usize,
        slow: usize,
    },
    Momentum {
        lookback: usize,
        threshold: f64,
    },
    MeanReversion {
        period: usize,
        num_std: f64,
        entry_std: f64,
        exit_std: f64,
    },
    Rsi {
        period: usize,
        oversold: f64,
        overbought: f64,
    },
    Breakout {
        entry_period: usize,
        exit_period: usize,
    },
    Macd {
        fast: usize,
        slow: usize,
        signal: usize,
    },
}

impl std::hash::Hash for StrategyParam {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            StrategyParam::Sma { fast, slow } => {
                std::mem::discriminant(self).hash(state);
                fast.hash(state);
                slow.hash(state);
            }
            StrategyParam::Momentum {
                lookback,
                threshold,
            } => {
                std::mem::discriminant(self).hash(state);
                lookback.hash(state);
                threshold.to_bits().hash(state);
            }
            StrategyParam::MeanReversion {
                period,
                num_std,
                entry_std,
                exit_std,
            } => {
                std::mem::discriminant(self).hash(state);
                period.hash(state);
                num_std.to_bits().hash(state);
                entry_std.to_bits().hash(state);
                exit_std.to_bits().hash(state);
            }
            StrategyParam::Rsi {
                period,
                oversold,
                overbought,
            } => {
                std::mem::discriminant(self).hash(state);
                period.hash(state);
                oversold.to_bits().hash(state);
                overbought.to_bits().hash(state);
            }
            StrategyParam::Breakout {
                entry_period,
                exit_period,
            } => {
                std::mem::discriminant(self).hash(state);
                entry_period.hash(state);
                exit_period.hash(state);
            }
            StrategyParam::Macd { fast, slow, signal } => {
                std::mem::discriminant(self).hash(state);
                fast.hash(state);
                slow.hash(state);
                signal.hash(state);
            }
        }
    }
}

/// Load data based on format argument.
fn load_data_with_format(
    path: &PathBuf,
    config: &DataConfig,
    format: DataFormatArg,
) -> Result<Vec<mantis::types::Bar>> {
    match format {
        DataFormatArg::Auto => load_data(path, config),
        DataFormatArg::Csv => load_csv(path, config),
        DataFormatArg::Parquet => load_parquet(path, config),
    }
}

impl Cli {
    /// Initialize logging based on verbosity level.
    pub fn init_logging(&self) {
        let level = match self.verbose {
            0 => Level::WARN,
            1 => Level::INFO,
            2 => Level::DEBUG,
            _ => Level::TRACE,
        };

        let subscriber = FmtSubscriber::builder()
            .with_max_level(level)
            .with_target(false)
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set tracing subscriber");
    }
}

/// Run the CLI application.
pub fn run() -> Result<()> {
    let cli = Cli::parse();
    cli.init_logging();

    match &cli.command {
        Commands::Run {
            data,
            symbol,
            strategy,
            capital,
            position_size,
            sizing_method,
            fixed_dollar,
            target_vol,
            vol_lookback,
            risk_per_trade,
            stop_atr,
            atr_period,
            commission,
            slippage,
            execution_price,
            fill_probability,
            limit_order_ttl,
            lot_selection,
            allow_short,
            max_leverage,
            reg_t_long,
            reg_t_short,
            maintenance_long,
            maintenance_short,
            margin_interest,
            use_portfolio_margin,
            portfolio_margin_pct,
            disable_margin,
            fast_period,
            slow_period,
            rsi_period,
            lookback,
            format,
            asset_class,
            multiplier,
            tick_size,
            margin_requirement,
            base_precision,
            quote_precision,
            pip_size,
            lot_size,
            underlying,
            seed,
            max_volume_participation,
        } => run_backtest(
            data,
            symbol,
            *strategy,
            *capital,
            *position_size,
            *sizing_method,
            *fixed_dollar,
            *target_vol,
            *vol_lookback,
            *risk_per_trade,
            *stop_atr,
            *atr_period,
            *commission,
            *slippage,
            *max_volume_participation,
            *execution_price,
            *fill_probability,
            *limit_order_ttl,
            *lot_selection,
            *allow_short,
            *max_leverage,
            *reg_t_long,
            *reg_t_short,
            *maintenance_long,
            *maintenance_short,
            *margin_interest,
            *use_portfolio_margin,
            *portfolio_margin_pct,
            *disable_margin,
            *fast_period,
            *slow_period,
            *rsi_period,
            *lookback,
            *format,
            *asset_class,
            *multiplier,
            *tick_size,
            *margin_requirement,
            *base_precision,
            *quote_precision,
            *pip_size,
            *lot_size,
            underlying.clone(),
            *seed,
            cli.output,
        ),

        Commands::WalkForward {
            data,
            symbol,
            strategy,
            capital,
            position_size,
            commission,
            slippage,
            max_volume_participation,
            execution_price,
            fill_probability,
            limit_order_ttl,
            lot_selection,
            allow_short,
            max_leverage,
            reg_t_long,
            reg_t_short,
            maintenance_long,
            maintenance_short,
            margin_interest,
            use_portfolio_margin,
            portfolio_margin_pct,
            disable_margin,
            folds,
            in_sample_ratio,
            anchored,
            min_bars,
            metric,
            format,
        } => run_walk_forward(
            data,
            symbol,
            *strategy,
            *capital,
            *position_size,
            *commission,
            *slippage,
            *max_volume_participation,
            *execution_price,
            *fill_probability,
            *limit_order_ttl,
            *lot_selection,
            *allow_short,
            *max_leverage,
            *reg_t_long,
            *reg_t_short,
            *maintenance_long,
            *maintenance_short,
            *margin_interest,
            *use_portfolio_margin,
            *portfolio_margin_pct,
            *disable_margin,
            *folds,
            *in_sample_ratio,
            *anchored,
            *min_bars,
            *metric,
            *format,
            cli.output,
        ),

        Commands::Optimize {
            data,
            symbol,
            strategy,
            capital,
            metric,
        } => run_optimization(data, symbol, *strategy, *capital, *metric, cli.output),

        Commands::Strategies => {
            print_strategies();
            Ok(())
        }

        Commands::Validate { data } => validate_data(data),

        Commands::Init { output } => init_config(output),

        Commands::RunConfig { config } => run_from_config(config, cli.output),

        Commands::Features {
            data,
            output_dir,
            target_horizon,
            feature_config,
            train_ratio,
            validation_ratio,
        } => extract_features(
            data,
            output_dir,
            *target_horizon,
            *feature_config,
            *train_ratio,
            *validation_ratio,
        ),

        Commands::Resample {
            input,
            output,
            interval,
            format,
        } => resample_data(input, output, *interval, *format),

        Commands::Quality {
            data,
            interval_seconds,
            format,
        } => run_quality_report(data, *interval_seconds, *format),

        Commands::MonteCarlo {
            data,
            symbol,
            strategy,
            capital,
            position_size,
            commission,
            slippage,
            simulations,
            confidence,
            seed,
            resample_trades,
            shuffle_returns,
            fast_period,
            slow_period,
            rsi_period,
            lookback,
            format,
        } => run_monte_carlo(
            data,
            symbol,
            *strategy,
            *capital,
            *position_size,
            *commission,
            *slippage,
            *simulations,
            *confidence,
            *seed,
            *resample_trades,
            *shuffle_returns,
            *fast_period,
            *slow_period,
            *rsi_period,
            *lookback,
            *format,
            cli.output,
        ),

        Commands::CostSensitivity {
            data,
            symbol,
            strategy,
            capital,
            position_size,
            commission,
            slippage,
            multipliers,
            include_zero_cost,
            robustness_threshold,
            fast_period,
            slow_period,
            rsi_period,
            lookback,
            format,
        } => run_cost_sensitivity(
            data,
            symbol,
            *strategy,
            *capital,
            *position_size,
            *commission,
            *slippage,
            multipliers,
            *include_zero_cost,
            *robustness_threshold,
            *fast_period,
            *slow_period,
            *rsi_period,
            *lookback,
            *format,
            cli.output,
        ),

        Commands::Experiments(experiments_cmd) => match experiments_cmd {
            ExperimentsCommands::List {
                strategy,
                min_sharpe,
                max_drawdown,
                limit,
                sort_by,
                desc,
            } => experiments_list(strategy, *min_sharpe, *max_drawdown, *limit, sort_by, *desc),

            ExperimentsCommands::Show { id } => experiments_show(id),

            ExperimentsCommands::Compare { id1, id2 } => experiments_compare(id1, id2),

            ExperimentsCommands::Tag { id, tags } => experiments_tag(id, tags),

            ExperimentsCommands::Note { id, note } => experiments_note(id, note),

            ExperimentsCommands::Delete { id } => experiments_delete(id),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn run_backtest(
    data_path: &PathBuf,
    symbol: &str,
    strategy_type: StrategyType,
    capital: f64,
    position_size: f64,
    sizing_method: PositionSizingMethodArg,
    fixed_dollar: Option<f64>,
    target_vol: Option<f64>,
    vol_lookback: usize,
    risk_per_trade: Option<f64>,
    stop_atr: f64,
    atr_period: usize,
    commission: f64,
    slippage: f64,
    max_volume_participation: Option<f64>,
    execution_price: ExecutionPriceArg,
    fill_probability: f64,
    limit_order_ttl: usize,
    lot_selection: LotSelectionArg,
    allow_short: bool,
    max_leverage: f64,
    reg_t_long: f64,
    reg_t_short: f64,
    maintenance_long: f64,
    maintenance_short: f64,
    margin_interest: f64,
    use_portfolio_margin: bool,
    portfolio_margin_pct: f64,
    disable_margin: bool,
    fast_period: usize,
    slow_period: usize,
    rsi_period: usize,
    lookback: usize,
    format: DataFormatArg,
    asset_class: AssetClassArg,
    multiplier: f64,
    tick_size: f64,
    margin_requirement: f64,
    base_precision: u8,
    quote_precision: u8,
    pip_size: f64,
    lot_size: f64,
    underlying: Option<String>,
    seed: Option<u64>,
    output: OutputFormat,
) -> Result<()> {
    use mantis::risk::PositionSizingMethod;

    info!("Loading data from: {}", data_path.display());
    let bars = load_data_with_format(data_path, &DataConfig::default(), format)?;

    let fill_probability = fill_probability.clamp(0.0, 1.0);
    let limit_order_ttl_bars = if limit_order_ttl == 0 {
        None
    } else {
        Some(limit_order_ttl)
    };

    // Build position sizing method from CLI args
    let position_sizing_method = match sizing_method {
        PositionSizingMethodArg::Percent => None, // Use position_size field as fallback
        PositionSizingMethodArg::Fixed => {
            let amount = fixed_dollar.unwrap_or(10000.0);
            Some(PositionSizingMethod::FixedDollar(amount))
        }
        PositionSizingMethodArg::Volatility => {
            let vol = target_vol.unwrap_or(0.15);
            Some(PositionSizingMethod::VolatilityTargeted {
                target_vol: vol,
                lookback: vol_lookback,
            })
        }
        PositionSizingMethodArg::Signal => Some(PositionSizingMethod::SignalScaled {
            base_size: position_size,
        }),
        PositionSizingMethodArg::Risk => {
            let risk = risk_per_trade.unwrap_or(0.01);
            Some(PositionSizingMethod::RiskBased {
                risk_per_trade: risk * 100.0, // Convert to percentage (1% -> 1.0)
                stop_atr,
                atr_period,
            })
        }
    };

    let cost_model = CostModel {
        commission_pct: commission / 100.0,
        slippage_pct: slippage / 100.0,
        max_volume_participation,
        ..Default::default()
    };

    let margin = MarginConfig {
        enabled: !disable_margin,
        reg_t_long_initial: reg_t_long,
        reg_t_short_initial: reg_t_short,
        maintenance_long_pct: maintenance_long,
        maintenance_short_pct: maintenance_short,
        max_leverage,
        use_portfolio_margin,
        portfolio_margin_pct,
        interest_rate: margin_interest,
    };

    let config = BacktestConfig {
        initial_capital: capital,
        cost_model,
        margin,
        position_size,
        position_sizing_method,
        allow_short,
        show_progress: true,
        execution_price: execution_price.into(),
        fill_probability,
        limit_order_ttl_bars,
        lot_selection: lot_selection.into(),
        seed,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data(symbol.to_string(), bars);
    let asset_config = build_asset_config(
        symbol,
        asset_class,
        multiplier,
        tick_size,
        margin_requirement,
        base_precision,
        quote_precision,
        pip_size,
        lot_size,
        underlying,
    );
    engine.data_mut().set_asset_config(asset_config);

    let mut strategy: Box<dyn Strategy> = match strategy_type {
        StrategyType::SmaCrossover => Box::new(SmaCrossover::new(fast_period, slow_period)),
        StrategyType::Momentum => Box::new(MomentumStrategy::new(lookback, 0.0)),
        StrategyType::MeanReversion => Box::new(MeanReversion::default_params()),
        StrategyType::Rsi => Box::new(RsiStrategy::new(rsi_period, 30.0, 70.0)),
        StrategyType::Breakout => Box::new(BreakoutStrategy::default_params()),
        StrategyType::Macd => Box::new(MacdStrategy::new(fast_period, slow_period, 9)),
    };

    let start_time = std::time::Instant::now();
    let result = engine.run(strategy.as_mut(), symbol)?;
    let duration_ms = start_time.elapsed().as_millis() as i64;

    // Automatically log experiment to store
    if let Err(e) = log_experiment_to_store(&result, duration_ms) {
        eprintln!("Warning: Failed to log experiment: {}", e);
    } else {
        println!(
            "\nExperiment logged: {}",
            result.experiment_id.to_string().bright_cyan()
        );
    }

    match output {
        OutputFormat::Text => ResultFormatter::print_report(&result),
        OutputFormat::Json => println!("{}", ResultFormatter::to_json(&result)),
        OutputFormat::Csv => {
            println!("{}", ResultFormatter::csv_header());
            println!("{}", ResultFormatter::to_csv_line(&result));
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_walk_forward(
    data_path: &PathBuf,
    symbol: &str,
    strategy_type: StrategyType,
    capital: f64,
    position_size: f64,
    commission: f64,
    slippage: f64,
    max_volume_participation: Option<f64>,
    execution_price: ExecutionPriceArg,
    fill_probability: f64,
    limit_order_ttl: usize,
    lot_selection: LotSelectionArg,
    allow_short: bool,
    max_leverage: f64,
    reg_t_long: f64,
    reg_t_short: f64,
    maintenance_long: f64,
    maintenance_short: f64,
    margin_interest: f64,
    use_portfolio_margin: bool,
    portfolio_margin_pct: f64,
    disable_margin: bool,
    folds: usize,
    in_sample_ratio: f64,
    anchored: bool,
    min_bars: usize,
    metric: WalkForwardMetricArg,
    format: DataFormatArg,
    output: OutputFormat,
) -> Result<()> {
    if folds == 0 {
        return Err(mantis::BacktestError::ConfigError(
            "Number of folds must be greater than zero".to_string(),
        ));
    }

    if !(0.0 < in_sample_ratio && in_sample_ratio < 1.0) {
        return Err(mantis::BacktestError::ConfigError(
            "In-sample ratio must be between 0 and 1".to_string(),
        ));
    }

    info!("Loading data from: {}", data_path.display());
    let bars = load_data_with_format(data_path, &DataConfig::default(), format)?;

    let fill_probability = fill_probability.clamp(0.0, 1.0);
    let limit_order_ttl_bars = if limit_order_ttl == 0 {
        None
    } else {
        Some(limit_order_ttl)
    };

    let cost_model = CostModel {
        commission_pct: commission / 100.0,
        slippage_pct: slippage / 100.0,
        max_volume_participation,
        ..Default::default()
    };

    let margin = MarginConfig {
        enabled: !disable_margin,
        reg_t_long_initial: reg_t_long,
        reg_t_short_initial: reg_t_short,
        maintenance_long_pct: maintenance_long,
        maintenance_short_pct: maintenance_short,
        max_leverage,
        use_portfolio_margin,
        portfolio_margin_pct,
        interest_rate: margin_interest,
    };

    let walk_forward_config = WalkForwardConfig {
        num_windows: folds,
        in_sample_ratio,
        anchored,
        min_bars_per_window: min_bars,
    };

    let backtest_config = BacktestConfig {
        initial_capital: capital,
        cost_model,
        margin,
        position_size,
        allow_short,
        show_progress: false,
        execution_price: execution_price.into(),
        fill_probability,
        limit_order_ttl_bars,
        lot_selection: lot_selection.into(),
        ..Default::default()
    };

    let analyzer = WalkForwardAnalyzer::new(walk_forward_config, backtest_config);

    let Some(params) = default_param_grid(strategy_type) else {
        println!("Walk-forward analysis not implemented for this strategy");
        return Ok(());
    };

    if params.is_empty() {
        println!("No parameter combinations available for walk-forward analysis");
        return Ok(());
    }

    let result = analyzer.run(
        &bars,
        symbol,
        params,
        |param| strategy_from_param(param),
        metric.into(),
    )?;

    match output {
        OutputFormat::Text => print_walk_forward_text(&result),
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&result)?;
            println!("{}", json);
        }
        OutputFormat::Csv => print_walk_forward_csv(&result),
    }

    Ok(())
}

fn print_walk_forward_text(result: &WalkForwardResult) {
    println!("\nWalk-Forward Analysis\n=====================\n");
    for window in &result.windows {
        println!(
            "Window {} (IS {}-{} / OOS {}-{}):",
            window.window.index + 1,
            window.window.is_start.format("%Y-%m-%d"),
            window.window.is_end.format("%Y-%m-%d"),
            window.window.oos_start.format("%Y-%m-%d"),
            window.window.oos_end.format("%Y-%m-%d")
        );
        println!(
            "  In-Sample   : {:>8.2}% return | Sharpe {:>5.2}",
            window.in_sample_result.total_return_pct, window.in_sample_result.sharpe_ratio
        );
        println!(
            "  Out-of-Sample: {:>8.2}% return | Sharpe {:>5.2}",
            window.out_of_sample_result.total_return_pct, window.out_of_sample_result.sharpe_ratio
        );
        println!(
            "  Efficiency  : {:>8.2}%\n",
            window.efficiency_ratio * 100.0
        );
    }

    println!("Summary:\n--------");
    println!("  Windows            : {}", result.windows.len());
    println!("  Avg IS Return      : {:.2}%", result.avg_is_return);
    println!("  Avg OOS Return     : {:.2}%", result.avg_oos_return);
    println!(
        "  Avg Efficiency     : {:.2}%",
        result.avg_efficiency_ratio * 100.0
    );
    println!("  Combined OOS Return: {:.2}%", result.combined_oos_return);
    println!(
        "  WF Efficiency      : {:.2}%",
        result.walk_forward_efficiency * 100.0
    );
    println!("\nOverfitting Detection:");
    println!("  Avg IS Sharpe      : {:.3}", result.avg_is_sharpe);
    println!("  Avg OOS Sharpe     : {:.3}", result.avg_oos_sharpe);
    println!(
        "  OOS/IS Sharpe Ratio: {:.2}%",
        result.oos_sharpe_ratio() * 100.0
    );
    println!(
        "  OOS Sharpe Threshold: {} (>= 60% of IS required)",
        if result.oos_sharpe_threshold_met {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!(
        "  Parameter Stability: {:.2}% (higher is more robust)",
        result.parameter_stability * 100.0
    );

    let basic_robust = result.is_robust(0.5);
    let sharpe_robust = result.is_robust_with_sharpe(0.5, 0.6);

    println!("\nRobustness Assessment:");
    println!(
        "  Basic Test (WF Efficiency >= 50%): {}",
        if basic_robust { "PASSED" } else { "FAILED" }
    );
    println!(
        "  Sharpe Test (OOS >= 60% of IS):     {}",
        if sharpe_robust { "PASSED" } else { "FAILED" }
    );

    if basic_robust && sharpe_robust {
        println!("\nOverall Result: PASSED - Strategy shows robust out-of-sample performance");
    } else if basic_robust {
        println!("\nOverall Result: WARNING - Efficiency passed but Sharpe degraded significantly");
    } else {
        println!("\nOverall Result: FAILED - Strategy shows signs of overfitting");
    }
}

fn print_walk_forward_csv(result: &WalkForwardResult) {
    println!(
        "window,is_start,is_end,oos_start,oos_end,is_return_pct,oos_return_pct,is_sharpe,oos_sharpe,efficiency_pct"
    );
    for window in &result.windows {
        println!(
            "{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
            window.window.index + 1,
            window.window.is_start.format("%Y-%m-%d"),
            window.window.is_end.format("%Y-%m-%d"),
            window.window.oos_start.format("%Y-%m-%d"),
            window.window.oos_end.format("%Y-%m-%d"),
            window.in_sample_result.total_return_pct,
            window.out_of_sample_result.total_return_pct,
            window.in_sample_result.sharpe_ratio,
            window.out_of_sample_result.sharpe_ratio,
            window.efficiency_ratio * 100.0
        );
    }

    println!(
        "summary,,,,,{:.4},{:.4},,,{:.4}",
        result.avg_is_return,
        result.avg_oos_return,
        result.walk_forward_efficiency * 100.0
    );
}

#[allow(clippy::too_many_arguments)]
fn build_asset_config(
    symbol: &str,
    asset_class: AssetClassArg,
    multiplier: f64,
    tick_size: f64,
    margin_requirement: f64,
    base_precision: u8,
    quote_precision: u8,
    pip_size: f64,
    lot_size: f64,
    underlying: Option<String>,
) -> AssetConfig {
    match asset_class {
        AssetClassArg::Equity => AssetConfig::equity(symbol),
        AssetClassArg::Future => AssetConfig::new(
            symbol,
            AssetClass::Future {
                multiplier,
                tick_size,
                margin_requirement,
            },
        ),
        AssetClassArg::Crypto => AssetConfig::new(
            symbol,
            AssetClass::Crypto {
                base_precision,
                quote_precision,
            },
        ),
        AssetClassArg::Forex => AssetConfig::new(symbol, AssetClass::Forex { pip_size, lot_size }),
        AssetClassArg::Option => {
            let underlying_symbol = underlying.unwrap_or_else(|| symbol.to_string());
            AssetConfig::new(
                symbol,
                AssetClass::Option {
                    underlying: underlying_symbol,
                    multiplier,
                },
            )
        }
    }
}

fn default_param_grid(strategy: StrategyType) -> Option<Vec<StrategyParam>> {
    match strategy {
        StrategyType::SmaCrossover => {
            let mut params = Vec::new();
            for fast in (5..=20).step_by(5) {
                for slow in (20..=60).step_by(10) {
                    if fast < slow {
                        params.push(StrategyParam::Sma { fast, slow });
                    }
                }
            }
            Some(params)
        }
        StrategyType::Momentum => {
            let mut params = Vec::new();
            for lookback in (5..=30).step_by(5) {
                params.push(StrategyParam::Momentum {
                    lookback,
                    threshold: 0.0,
                });
            }
            Some(params)
        }
        StrategyType::MeanReversion => {
            let mut params = Vec::new();
            let periods = [10, 15, 20, 30];
            let entry_thresholds = [1.5, 2.0, 2.5];
            let exit_thresholds = [0.5, 1.0];
            for &period in &periods {
                for &entry_std in &entry_thresholds {
                    for &exit_std in &exit_thresholds {
                        if entry_std > exit_std {
                            params.push(StrategyParam::MeanReversion {
                                period,
                                num_std: entry_std,
                                entry_std,
                                exit_std,
                            });
                        }
                    }
                }
            }
            Some(params)
        }
        StrategyType::Rsi => {
            let mut params = Vec::new();
            let periods = [7, 14, 21];
            let oversold_levels = [25.0, 30.0, 35.0];
            let overbought_levels = [65.0, 70.0, 75.0];
            for &period in &periods {
                for &oversold in &oversold_levels {
                    for &overbought in &overbought_levels {
                        if oversold < overbought {
                            params.push(StrategyParam::Rsi {
                                period,
                                oversold,
                                overbought,
                            });
                        }
                    }
                }
            }
            Some(params)
        }
        StrategyType::Breakout => {
            let mut params = Vec::new();
            let entry_periods = [20, 30, 40, 50];
            let exit_periods = [10, 15, 20];
            for &entry_period in &entry_periods {
                for &exit_period in &exit_periods {
                    if exit_period < entry_period {
                        params.push(StrategyParam::Breakout {
                            entry_period,
                            exit_period,
                        });
                    }
                }
            }
            Some(params)
        }
        StrategyType::Macd => {
            let mut params = Vec::new();
            for fast in (8..=16).step_by(4) {
                for slow in (20..=30).step_by(5) {
                    if fast < slow {
                        params.push(StrategyParam::Macd {
                            fast,
                            slow,
                            signal: 9,
                        });
                    }
                }
            }
            Some(params)
        }
    }
}

fn strategy_from_param(param: &StrategyParam) -> Box<dyn Strategy> {
    match param {
        StrategyParam::Sma { fast, slow } => Box::new(SmaCrossover::new(*fast, *slow)),
        StrategyParam::Momentum {
            lookback,
            threshold,
        } => Box::new(MomentumStrategy::new(*lookback, *threshold)),
        StrategyParam::MeanReversion {
            period,
            num_std,
            entry_std,
            exit_std,
        } => Box::new(MeanReversion::new(*period, *num_std, *entry_std, *exit_std)),
        StrategyParam::Rsi {
            period,
            oversold,
            overbought,
        } => Box::new(RsiStrategy::new(*period, *oversold, *overbought)),
        StrategyParam::Breakout {
            entry_period,
            exit_period,
        } => Box::new(BreakoutStrategy::new(*entry_period, *exit_period)),
        StrategyParam::Macd { fast, slow, signal } => {
            Box::new(MacdStrategy::new(*fast, *slow, *signal))
        }
    }
}

impl From<ExecutionPriceArg> for ExecutionPrice {
    fn from(arg: ExecutionPriceArg) -> Self {
        match arg {
            ExecutionPriceArg::Open => ExecutionPrice::Open,
            ExecutionPriceArg::Close => ExecutionPrice::Close,
            ExecutionPriceArg::Vwap => ExecutionPrice::Vwap,
            ExecutionPriceArg::Twap => ExecutionPrice::Twap,
            ExecutionPriceArg::RandomInRange => ExecutionPrice::RandomInRange,
            ExecutionPriceArg::Midpoint => ExecutionPrice::Midpoint,
        }
    }
}

impl From<LotSelectionArg> for LotSelectionMethod {
    fn from(arg: LotSelectionArg) -> Self {
        match arg {
            LotSelectionArg::Fifo => LotSelectionMethod::FIFO,
            LotSelectionArg::Lifo => LotSelectionMethod::LIFO,
            LotSelectionArg::HighestCost => LotSelectionMethod::HighestCost,
            LotSelectionArg::LowestCost => LotSelectionMethod::LowestCost,
        }
    }
}

impl From<WalkForwardMetricArg> for WalkForwardMetric {
    fn from(arg: WalkForwardMetricArg) -> Self {
        match arg {
            WalkForwardMetricArg::Sharpe => WalkForwardMetric::Sharpe,
            WalkForwardMetricArg::Sortino => WalkForwardMetric::Sortino,
            WalkForwardMetricArg::Return => WalkForwardMetric::Return,
            WalkForwardMetricArg::Calmar => WalkForwardMetric::Calmar,
            WalkForwardMetricArg::ProfitFactor => WalkForwardMetric::ProfitFactor,
        }
    }
}

fn run_optimization(
    data_path: &PathBuf,
    symbol: &str,
    strategy_type: StrategyType,
    capital: f64,
    metric: OptimizeMetric,
    output: OutputFormat,
) -> Result<()> {
    info!("Loading data from: {}", data_path.display());
    let bars = load_csv(data_path, &DataConfig::default())?;

    let config = BacktestConfig {
        initial_capital: capital,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data(symbol.to_string(), bars);

    info!("Running parameter optimization...");

    let Some(params) = default_param_grid(strategy_type) else {
        println!("Optimization not implemented for this strategy");
        return Ok(());
    };

    let n_trials = params.len();
    info!(
        "Testing {} parameter combinations (n_trials for deflated Sharpe)",
        n_trials
    );

    // Extract just BacktestResults, discarding the parameter info
    let mut results: Vec<mantis::BacktestResult> = engine
        .optimize(symbol, params, |param| strategy_from_param(param))?
        .into_iter()
        .map(|(_, r)| r)
        .collect();

    if results.is_empty() {
        println!("No optimization results produced for the selected strategy");
        return Ok(());
    }

    // Sort by metric
    results.sort_by(|a, b| {
        let metric_a = match metric {
            OptimizeMetric::Sharpe => a.sharpe_ratio,
            OptimizeMetric::Sortino => a.sortino_ratio,
            OptimizeMetric::Return => a.total_return_pct,
            OptimizeMetric::Calmar => a.calmar_ratio,
        };
        let metric_b = match metric {
            OptimizeMetric::Sharpe => b.sharpe_ratio,
            OptimizeMetric::Sortino => b.sortino_ratio,
            OptimizeMetric::Return => b.total_return_pct,
            OptimizeMetric::Calmar => b.calmar_ratio,
        };
        metric_b.partial_cmp(&metric_a).unwrap()
    });

    println!("\nOptimization Results (sorted by {:?}):\n", metric);
    println!(
        "Note: {} parameter combinations tested. Deflated Sharpe ratio accounts for multiple testing.\n",
        n_trials
    );

    match output {
        OutputFormat::Text => {
            ResultFormatter::print_table(&results);

            // Show deflated Sharpe for top result
            if let Some(best) = results.first() {
                use mantis::analytics::PerformanceMetrics;
                let metrics = PerformanceMetrics::from_result_with_trials(best, n_trials);
                println!(
                    "\n--- Overfitting Detection Metrics (Best Result) ---\n\
                     Sharpe Ratio: {:.3}\n\
                     Deflated Sharpe Ratio: {:.3} (adjusted for {} trials)\n\
                     Probabilistic Sharpe Ratio: {:.3}\n\
                     Note: Deflated SR < 0 suggests performance may be due to luck/overfitting",
                    best.sharpe_ratio,
                    metrics.deflated_sharpe_ratio,
                    n_trials,
                    metrics.probabilistic_sharpe_ratio
                );
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&results).unwrap();
            println!("{}", json);
        }
        OutputFormat::Csv => {
            println!("{}", ResultFormatter::csv_header());
            for result in &results {
                println!("{}", ResultFormatter::to_csv_line(result));
            }
        }
    }

    Ok(())
}

fn print_strategies() {
    println!("\nAvailable Strategies:\n");

    println!("Built-in Strategies:");
    println!("  sma-crossover");
    println!("    Simple Moving Average Crossover strategy.");
    println!("    Parameters: --fast-period (default: 10), --slow-period (default: 30)");
    println!();

    println!("  momentum");
    println!("    Momentum/Rate of Change strategy.");
    println!("    Parameters: --lookback (default: 20)");
    println!();

    println!("  mean-reversion");
    println!("    Mean reversion using Bollinger Bands.");
    println!("    Uses default parameters: period=20, num_std=2.0");
    println!();

    println!("  rsi");
    println!("    Relative Strength Index strategy.");
    println!("    Parameters: --rsi-period (default: 14)");
    println!();

    println!("  breakout");
    println!("    Donchian Channel Breakout strategy.");
    println!("    Uses default parameters: entry=20, exit=10");
    println!();

    println!("  macd");
    println!("    MACD (Moving Average Convergence Divergence) strategy.");
    println!("    Parameters: --fast-period (default: 12), --slow-period (default: 26)");
    println!("    Signal period is fixed at 9.");
    println!();

    println!("ML/DL Integration (use via library API):");
    println!("  ExternalSignalStrategy");
    println!("    Accept continuous signals from ML model predictions.");
    println!();

    println!("  ClassificationStrategy");
    println!("    Use discrete class predictions (1=long, 0=hold, -1=short).");
    println!();

    println!("  ConfidenceWeightedStrategy");
    println!("    Position sizing based on model confidence scores.");
    println!();

    println!("  EnsembleSignalStrategy");
    println!("    Combine predictions from multiple ML models.");
    println!();

    println!("Feature Extraction:");
    println!("  Use 'mantis features' command to export features for ML training.");
    println!("  Example: mantis features -d data.csv -o ml_data --feature-config comprehensive");
    println!();
}

fn init_config(output: &PathBuf) -> Result<()> {
    use std::fs;

    let example = BacktestFileConfig::example();
    fs::write(output, example)?;
    println!("Created example configuration file: {}", output.display());
    println!("\nEdit this file to customize your backtest, then run:");
    println!("  mantis run-config -c {}", output.display());
    Ok(())
}

fn run_from_config(config_path: &PathBuf, output: OutputFormat) -> Result<()> {
    info!("Loading configuration from: {}", config_path.display());

    let file_config = BacktestFileConfig::load(config_path)?;
    let backtest_config = file_config.to_backtest_config()?;

    // Get data path
    let data_path = file_config.data.path.ok_or_else(|| {
        mantis::BacktestError::ConfigError("No data path specified in config".to_string())
    })?;

    info!("Loading data from: {}", data_path);
    let bars = load_csv(&data_path, &DataConfig::default())?;

    let mut engine = Engine::new(backtest_config);
    engine.add_data(file_config.data.symbol.clone(), bars);

    // Create strategy based on config
    let params = &file_config.strategy.params;
    let mut strategy: Box<dyn Strategy> = match file_config.strategy.name.to_lowercase().as_str() {
        "sma-crossover" | "sma" => {
            let fast = params.fast_period.unwrap_or(10);
            let slow = params.slow_period.unwrap_or(30);
            Box::new(SmaCrossover::new(fast, slow))
        }
        "momentum" => {
            let lookback = params.lookback.unwrap_or(20);
            let threshold = params.momentum_threshold.unwrap_or(0.0);
            Box::new(MomentumStrategy::new(lookback, threshold))
        }
        "mean-reversion" | "mean_reversion" => Box::new(MeanReversion::default_params()),
        "rsi" => {
            let period = params.rsi_period.unwrap_or(14);
            let oversold = params.rsi_oversold.unwrap_or(30.0);
            let overbought = params.rsi_overbought.unwrap_or(70.0);
            Box::new(RsiStrategy::new(period, oversold, overbought))
        }
        "breakout" => {
            let entry = params.entry_period.unwrap_or(20);
            let exit = params.exit_period.unwrap_or(10);
            Box::new(BreakoutStrategy::new(entry, exit))
        }
        "macd" => {
            let fast = params.fast_period.unwrap_or(12);
            let slow = params.slow_period.unwrap_or(26);
            let signal = params.signal_period.unwrap_or(9);
            Box::new(MacdStrategy::new(fast, slow, signal))
        }
        other => {
            return Err(mantis::BacktestError::ConfigError(format!(
                "Unknown strategy: {}",
                other
            )));
        }
    };

    let result = engine.run(strategy.as_mut(), &file_config.data.symbol)?;

    match output {
        OutputFormat::Text => ResultFormatter::print_report(&result),
        OutputFormat::Json => println!("{}", ResultFormatter::to_json(&result)),
        OutputFormat::Csv => {
            println!("{}", ResultFormatter::csv_header());
            println!("{}", ResultFormatter::to_csv_line(&result));
        }
    }

    Ok(())
}

fn extract_features(
    data_path: &PathBuf,
    output_dir: &PathBuf,
    target_horizon: usize,
    feature_config: FeatureConfigType,
    train_ratio: f64,
    validation_ratio: f64,
) -> Result<()> {
    info!("Loading data from: {}", data_path.display());
    let bars = load_csv(data_path, &DataConfig::default())?;

    println!("Loaded {} bars", bars.len());

    // Create output directory
    fs::create_dir_all(output_dir)?;

    // Configure feature extraction
    let config = match feature_config {
        FeatureConfigType::Minimal => FeatureConfig::minimal(),
        FeatureConfigType::Default => FeatureConfig::default(),
        FeatureConfigType::Comprehensive => FeatureConfig::comprehensive(),
    };

    let extractor = FeatureExtractor::new(config);
    println!("Warmup period: {} bars", extractor.warmup_period());

    // Split data
    let splitter = TimeSeriesSplitter::new(train_ratio, validation_ratio).with_gap(5);
    let (train_bars, val_bars, test_bars) = splitter.split(&bars);

    println!("\nData Split:");
    println!("  Train: {} bars", train_bars.len());
    println!("  Validation: {} bars", val_bars.len());
    println!("  Test: {} bars", test_bars.len());

    // Export training data
    println!("\nExporting features...");

    let train_csv = extractor.to_csv(&train_bars, Some(target_horizon));
    let train_path = output_dir.join("train.csv");
    fs::write(&train_path, &train_csv)?;
    println!(
        "  Saved: {} ({} bytes)",
        train_path.display(),
        train_csv.len()
    );

    let val_csv = extractor.to_csv(&val_bars, Some(target_horizon));
    let val_path = output_dir.join("validation.csv");
    fs::write(&val_path, &val_csv)?;
    println!("  Saved: {} ({} bytes)", val_path.display(), val_csv.len());

    let test_csv = extractor.to_csv(&test_bars, Some(target_horizon));
    let test_path = output_dir.join("test.csv");
    fs::write(&test_path, &test_csv)?;
    println!(
        "  Saved: {} ({} bytes)",
        test_path.display(),
        test_csv.len()
    );

    // Export metadata
    let (_, feature_names) = extractor.extract_matrix(&train_bars);
    let metadata = serde_json::json!({
        "feature_names": feature_names,
        "num_features": feature_names.len(),
        "target_horizon": target_horizon,
        "warmup_period": extractor.warmup_period(),
        "splits": {
            "train_bars": train_bars.len(),
            "validation_bars": val_bars.len(),
            "test_bars": test_bars.len(),
        }
    });

    let meta_path = output_dir.join("metadata.json");
    fs::write(&meta_path, serde_json::to_string_pretty(&metadata).unwrap())?;
    println!("  Saved: {}", meta_path.display());

    println!("\nFeature extraction complete!");
    println!("Output directory: {}", output_dir.display());
    println!("\nTo use with Python:");
    println!("  import pandas as pd");
    println!(
        "  train_df = pd.read_csv('{}/train.csv')",
        output_dir.display()
    );

    Ok(())
}

fn resample_data(
    input_path: &PathBuf,
    output_path: &PathBuf,
    interval: ResampleIntervalArg,
    format: DataFormatArg,
) -> Result<()> {
    info!("Loading data from: {}", input_path.display());
    let bars = load_data_with_format(input_path, &DataConfig::default(), format)?;

    println!("Loaded {} bars from input file", bars.len());

    let target_interval = interval.to_interval();
    let resampled = resample(&bars, target_interval);

    println!(
        "Resampled to {} bars at {:?} interval",
        resampled.len(),
        target_interval
    );

    // Write output based on file extension
    let ext = output_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("csv")
        .to_lowercase();

    match ext.as_str() {
        "csv" => {
            // Create a simple CSV writer
            use std::io::Write;
            let mut file = fs::File::create(output_path)?;
            writeln!(file, "timestamp,open,high,low,close,volume")?;
            for bar in &resampled {
                writeln!(
                    file,
                    "{},{},{},{},{},{}",
                    bar.timestamp.format("%Y-%m-%d %H:%M:%S"),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume
                )?;
            }
            println!("Saved to: {}", output_path.display());
        }
        "parquet" | "pq" => {
            // Parquet OHLCV export is not directly supported yet
            // Users should use CSV format for resampled data
            return Err(mantis::BacktestError::DataError(
                "Parquet output for resampled OHLCV data is not yet supported. Use .csv extension instead.".to_string()
            ));
        }
        _ => {
            return Err(mantis::BacktestError::DataError(format!(
                "Unsupported output format: {}. Use .csv or .parquet",
                ext
            )));
        }
    }

    Ok(())
}

fn run_quality_report(
    data_path: &PathBuf,
    interval_seconds: i64,
    format: DataFormatArg,
) -> Result<()> {
    info!("Loading data from: {}", data_path.display());
    let bars = load_data_with_format(data_path, &DataConfig::default(), format)?;

    println!("Analyzing data quality...\n");

    let report = data_quality_report(&bars, interval_seconds);

    println!("Data Quality Report");
    println!("===================");
    println!("Total bars: {}", report.total_bars);
    println!(
        "Expected interval: {} seconds ({:.1} minutes / {:.2} hours)",
        report.expected_interval_seconds,
        report.expected_interval_seconds as f64 / 60.0,
        report.expected_interval_seconds as f64 / 3600.0
    );
    println!();

    if !bars.is_empty() {
        println!(
            "Date range: {} to {}",
            bars.first().unwrap().timestamp,
            bars.last().unwrap().timestamp
        );
        println!();
    }

    println!("Gaps: {} detected", report.gaps.len());
    if !report.gaps.is_empty() {
        println!("Gap percentage: {:.2}%", report.gap_percentage);
        println!();
        println!("Gap Details:");
        for (i, gap) in report.gaps.iter().enumerate().take(10) {
            println!(
                "  {}. {} to {} ({} bars missing)",
                i + 1,
                gap.start.format("%Y-%m-%d %H:%M:%S"),
                gap.end.format("%Y-%m-%d %H:%M:%S"),
                gap.expected_bars
            );
        }
        if report.gaps.len() > 10 {
            println!("  ... and {} more gaps", report.gaps.len() - 10);
        }
        println!();
    }

    println!("Invalid bars: {}", report.invalid_bars);
    println!(
        "Duplicate timestamps: {} (removed during loading)",
        report.duplicate_timestamps
    );
    println!();

    if report.is_acceptable() {
        println!("Status: GOOD - Data quality is acceptable");
    } else {
        println!("Status: ISSUES DETECTED - Review gaps and invalid bars");
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_monte_carlo(
    data_path: &PathBuf,
    symbol: &str,
    strategy_type: StrategyType,
    capital: f64,
    position_size: f64,
    commission: f64,
    slippage: f64,
    simulations: usize,
    confidence: f64,
    seed: Option<u64>,
    resample_trades: bool,
    shuffle_returns: bool,
    fast_period: usize,
    slow_period: usize,
    rsi_period: usize,
    lookback: usize,
    format: DataFormatArg,
    output: OutputFormat,
) -> Result<()> {
    info!("Loading data from: {}", data_path.display());
    let bars = load_data_with_format(data_path, &DataConfig::default(), format)?;

    println!("Running backtest before Monte Carlo simulation...\n");

    let cost_model = CostModel {
        commission_pct: commission / 100.0,
        slippage_pct: slippage / 100.0,
        ..Default::default()
    };

    let config = BacktestConfig {
        initial_capital: capital,
        cost_model,
        position_size,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data(symbol.to_string(), bars);

    let mut strategy: Box<dyn Strategy> = match strategy_type {
        StrategyType::SmaCrossover => Box::new(SmaCrossover::new(fast_period, slow_period)),
        StrategyType::Momentum => Box::new(MomentumStrategy::new(lookback, 0.0)),
        StrategyType::MeanReversion => Box::new(MeanReversion::default_params()),
        StrategyType::Rsi => Box::new(RsiStrategy::new(rsi_period, 30.0, 70.0)),
        StrategyType::Breakout => Box::new(BreakoutStrategy::default_params()),
        StrategyType::Macd => Box::new(MacdStrategy::new(fast_period, slow_period, 9)),
    };

    let result = engine.run(strategy.as_mut(), symbol)?;

    let closed_trades: Vec<_> = result.trades.iter().filter(|t| t.is_closed()).collect();
    if closed_trades.is_empty() {
        println!("No closed trades to analyze. Cannot run Monte Carlo simulation.");
        return Ok(());
    }

    println!(
        "Backtest complete: {} closed trades, {:.2}% return\n",
        closed_trades.len(),
        result.total_return_pct
    );

    println!("Running Monte Carlo simulation...");
    println!(
        "  Simulations: {}\n  Confidence: {:.0}%\n  Resample trades: {}\n  Shuffle returns: {}\n",
        simulations,
        confidence * 100.0,
        resample_trades,
        shuffle_returns
    );

    let mc_config = MonteCarloConfig {
        num_simulations: simulations,
        confidence_level: confidence,
        seed,
        resample_trades,
        shuffle_returns,
    };

    let mut simulator = MonteCarloSimulator::new(mc_config);
    let mc_result = simulator.simulate_from_result(&result);

    match output {
        OutputFormat::Text => print_monte_carlo_text(&mc_result),
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&mc_result)?;
            println!("{}", json);
        }
        OutputFormat::Csv => print_monte_carlo_csv(&mc_result),
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_cost_sensitivity(
    data_path: &PathBuf,
    symbol: &str,
    strategy_type: StrategyType,
    capital: f64,
    position_size: f64,
    commission: f64,
    slippage: f64,
    multipliers_str: &str,
    include_zero_cost: bool,
    robustness_threshold: f64,
    fast_period: usize,
    slow_period: usize,
    rsi_period: usize,
    lookback: usize,
    format: DataFormatArg,
    output: OutputFormat,
) -> Result<()> {
    info!("Loading data from: {}", data_path.display());
    let bars = load_data_with_format(data_path, &DataConfig::default(), format)?;

    // Parse multipliers from comma-separated string
    let mut multipliers: Vec<f64> = multipliers_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if multipliers.is_empty() {
        multipliers = vec![1.0, 2.0, 5.0, 10.0];
    }

    // Add zero-cost baseline if requested
    if include_zero_cost && !multipliers.contains(&0.0) {
        multipliers.insert(0, 0.0);
    }

    multipliers.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("Running cost sensitivity analysis...");
    println!("  Multipliers: {:?}", multipliers);
    println!(
        "  Robustness threshold (Sharpe at 5x): {:.2}\n",
        robustness_threshold
    );

    let cost_model = CostModel {
        commission_pct: commission / 100.0,
        slippage_pct: slippage / 100.0,
        ..Default::default()
    };

    let config = BacktestConfig {
        initial_capital: capital,
        cost_model,
        position_size,
        show_progress: false,
        ..Default::default()
    };

    let sensitivity_config = CostSensitivityConfig {
        multipliers,
        robustness_threshold_5x: Some(robustness_threshold),
        include_zero_cost,
    };

    let mut strategy: Box<dyn Strategy> = match strategy_type {
        StrategyType::SmaCrossover => Box::new(SmaCrossover::new(fast_period, slow_period)),
        StrategyType::Momentum => Box::new(MomentumStrategy::new(lookback, 0.0)),
        StrategyType::MeanReversion => Box::new(MeanReversion::default_params()),
        StrategyType::Rsi => Box::new(RsiStrategy::new(rsi_period, 30.0, 70.0)),
        StrategyType::Breakout => Box::new(BreakoutStrategy::default_params()),
        StrategyType::Macd => Box::new(MacdStrategy::new(fast_period, slow_period, 9)),
    };

    let analysis = run_cost_sensitivity_analysis(
        &config,
        &sensitivity_config,
        &bars,
        strategy.as_mut(),
        symbol,
    )
    .map_err(|e| {
        mantis::BacktestError::ConfigError(format!("Cost sensitivity analysis failed: {}", e))
    })?;

    match output {
        OutputFormat::Text => print_cost_sensitivity_text(&analysis),
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&analysis)?;
            println!("{}", json);
        }
        OutputFormat::Csv => print_cost_sensitivity_csv(&analysis),
    }

    Ok(())
}

fn print_cost_sensitivity_text(analysis: &CostSensitivityAnalysis) {
    println!("\n{}", "Cost Sensitivity Analysis".bold().underline());
    println!(
        "Symbol: {}  |  Strategy: {}",
        analysis.symbol.bright_cyan(),
        analysis.strategy_name.bright_cyan()
    );
    println!("{}", "=".repeat(70));
    println!();

    // Scenario table
    println!("{}", "Cost Scenarios".bold());
    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>10}",
        "Multiplier", "Return", "Sharpe", "Max DD", "Trades"
    );
    println!("{}", "-".repeat(60));

    for scenario in &analysis.scenarios {
        let return_str = format!("{:.2}%", scenario.total_return_pct());
        let sharpe_str = format!("{:.3}", scenario.sharpe_ratio());
        let dd_str = format!("{:.2}%", scenario.result.max_drawdown_pct);

        let return_colored = if scenario.total_return_pct() > 0.0 {
            return_str.green()
        } else {
            return_str.red()
        };

        let sharpe_colored = if scenario.sharpe_ratio() > 0.5 {
            sharpe_str.green()
        } else if scenario.sharpe_ratio() > 0.0 {
            sharpe_str.yellow()
        } else {
            sharpe_str.red()
        };

        println!(
            "{:<12} {:>12} {:>12} {:>12} {:>10}",
            format!("{}x", scenario.multiplier),
            return_colored,
            sharpe_colored,
            dd_str.red(),
            scenario.result.total_trades
        );
    }

    println!();
    println!("{}", "Degradation Analysis (vs 1x baseline)".bold());
    println!("{}", "-".repeat(60));

    if let Some(degrad_2x) = analysis.sharpe_degradation_at(2.0) {
        let degrad_str = format!("{:.1}%", degrad_2x);
        let colored = if degrad_2x < 25.0 {
            degrad_str.green()
        } else if degrad_2x < 50.0 {
            degrad_str.yellow()
        } else {
            degrad_str.red()
        };
        println!("  Sharpe degradation at 2x costs: {}", colored);
    }

    if let Some(degrad_5x) = analysis.sharpe_degradation_at(5.0) {
        let degrad_str = format!("{:.1}%", degrad_5x);
        let colored = if degrad_5x < 50.0 {
            degrad_str.green()
        } else if degrad_5x < 75.0 {
            degrad_str.yellow()
        } else {
            degrad_str.red()
        };
        println!("  Sharpe degradation at 5x costs: {}", colored);
    }

    if let Some(degrad_10x) = analysis.sharpe_degradation_at(10.0) {
        let degrad_str = format!("{:.1}%", degrad_10x);
        let colored = if degrad_10x < 75.0 {
            degrad_str.green()
        } else {
            degrad_str.red()
        };
        println!("  Sharpe degradation at 10x costs: {}", colored);
    }

    if let Some(elasticity) = analysis.cost_elasticity() {
        println!("\n  Cost elasticity: {:.3}", elasticity);
    }

    if let Some(breakeven) = analysis.breakeven_multiplier() {
        println!("  Breakeven cost multiplier: {:.2}x", breakeven);
    } else {
        println!("  Breakeven: Strategy profitable at all tested multipliers");
    }

    println!();
    println!("{}", "Robustness Assessment".bold());
    println!("{}", "-".repeat(60));

    if let Some(scenario_5x) = analysis.scenario_at(5.0) {
        let sharpe_5x = scenario_5x.sharpe_ratio();
        let threshold = 0.5;
        let passes = sharpe_5x >= threshold;
        let status = if passes {
            "PASS".green().bold()
        } else {
            "FAIL".red().bold()
        };
        println!(
            "  5x cost test: {} (Sharpe = {:.3}, threshold = {:.3})",
            status, sharpe_5x, threshold
        );
    }

    let overall_robust = analysis.is_robust(0.5);
    let verdict = if overall_robust {
        "ROBUST".green().bold()
    } else {
        "NOT ROBUST".red().bold()
    };
    println!("\n  Overall verdict: {}", verdict);

    if overall_robust {
        println!(
            "{}",
            "\n  Strategy maintains acceptable performance under stressed cost conditions."
                .green()
                .dimmed()
        );
    } else {
        println!(
            "{}",
            "\n  Strategy performance degrades significantly under higher costs."
                .yellow()
                .dimmed()
        );
        println!(
            "{}",
            "  Consider: Reducing trading frequency, widening entry criteria, or accepting lower returns."
                .yellow()
                .dimmed()
        );
    }
}

fn print_cost_sensitivity_csv(analysis: &CostSensitivityAnalysis) {
    println!("multiplier,return_pct,sharpe,max_dd_pct,total_trades,total_costs,avg_cost_per_trade");
    for scenario in &analysis.scenarios {
        println!(
            "{},{:.4},{:.4},{:.4},{},{:.4},{:.4}",
            scenario.multiplier,
            scenario.total_return_pct(),
            scenario.sharpe_ratio(),
            scenario.result.max_drawdown_pct,
            scenario.result.total_trades,
            scenario.total_costs,
            scenario.avg_cost_per_trade
        );
    }

    // Summary row
    if let Some(degrad_5x) = analysis.sharpe_degradation_at(5.0) {
        println!("# sharpe_degradation_5x,{:.4}", degrad_5x);
    }
    if let Some(elasticity) = analysis.cost_elasticity() {
        println!("# cost_elasticity,{:.4}", elasticity);
    }
    if let Some(breakeven) = analysis.breakeven_multiplier() {
        println!("# breakeven_multiplier,{:.4}", breakeven);
    }
    println!("# is_robust,{}", analysis.is_robust(0.5));
}

fn print_monte_carlo_text(result: &MonteCarloResult) {
    println!("\n{}", "Monte Carlo Simulation Results".bold().underline());
    println!("{}", "=".repeat(50));
    println!();

    println!("{}", "Configuration".bold());
    println!("  Simulations:      {}", result.num_simulations);
    println!("  Trades analyzed:  {}", result.num_trades);
    println!(
        "  Confidence level: {:.0}%",
        result.config.confidence_level * 100.0
    );
    println!();

    println!("{}", "Return Statistics".bold());
    let mean_colored = if result.mean_return > 0.0 {
        format!("{:.2}%", result.mean_return).green()
    } else {
        format!("{:.2}%", result.mean_return).red()
    };
    let median_colored = if result.median_return > 0.0 {
        format!("{:.2}%", result.median_return).green()
    } else {
        format!("{:.2}%", result.median_return).red()
    };
    println!("  Mean return:      {}", mean_colored);
    println!("  Median return:    {}", median_colored);
    println!("  Std deviation:    {:.2}%", result.return_std);
    println!(
        "  {:.0}% CI:          [{:.2}%, {:.2}%]",
        result.config.confidence_level * 100.0,
        result.return_ci.0,
        result.return_ci.1
    );
    let prob_colored = if result.prob_positive_return > 0.6 {
        format!("{:.1}%", result.prob_positive_return * 100.0).green()
    } else if result.prob_positive_return > 0.5 {
        format!("{:.1}%", result.prob_positive_return * 100.0).yellow()
    } else {
        format!("{:.1}%", result.prob_positive_return * 100.0).red()
    };
    println!("  P(Return > 0):    {}", prob_colored);
    println!();

    println!("{}", "Risk Metrics".bold());
    println!(
        "  Mean max DD:      {}",
        format!("{:.2}%", result.mean_max_drawdown).red()
    );
    println!(
        "  Median max DD:    {}",
        format!("{:.2}%", result.median_max_drawdown).red()
    );
    println!(
        "  95th %ile DD:     {}",
        format!("{:.2}%", result.max_drawdown_95th).red()
    );
    println!(
        "  VaR ({:.0}%):        {:.2}%",
        result.config.confidence_level * 100.0,
        result.var
    );
    println!("  CVaR (ES):        {:.2}%", result.cvar);
    println!();

    println!("{}", "Sharpe Ratio".bold());
    let sharpe_mean_colored = if result.mean_sharpe > 1.0 {
        format!("{:.3}", result.mean_sharpe).green()
    } else if result.mean_sharpe > 0.0 {
        format!("{:.3}", result.mean_sharpe).yellow()
    } else {
        format!("{:.3}", result.mean_sharpe).red()
    };
    println!("  Mean Sharpe:      {}", sharpe_mean_colored);
    println!("  Median Sharpe:    {:.3}", result.median_sharpe);
    println!(
        "  {:.0}% CI:          [{:.3}, {:.3}]",
        result.config.confidence_level * 100.0,
        result.sharpe_ci.0,
        result.sharpe_ci.1
    );
    let sharpe_prob_colored = if result.prob_positive_sharpe > 0.6 {
        format!("{:.1}%", result.prob_positive_sharpe * 100.0).green()
    } else if result.prob_positive_sharpe > 0.5 {
        format!("{:.1}%", result.prob_positive_sharpe * 100.0).yellow()
    } else {
        format!("{:.1}%", result.prob_positive_sharpe * 100.0).red()
    };
    println!("  P(Sharpe > 0):    {}", sharpe_prob_colored);
    println!();

    println!("{}", "Return Distribution Percentiles".bold());
    let percentile_order = ["5th", "10th", "25th", "50th", "75th", "90th", "95th"];
    for p in &percentile_order {
        if let Some(&val) = result.return_percentiles.get(*p) {
            let val_colored = if val > 0.0 {
                format!("{:.2}%", val).green()
            } else {
                format!("{:.2}%", val).red()
            };
            println!("  {}:             {}", p, val_colored);
        }
    }
    println!();

    println!("{}", "Robustness Assessment".bold());
    let score = result.robustness_score();
    let score_colored = if score > 70.0 {
        format!("{:.1}/100", score).green()
    } else if score > 50.0 {
        format!("{:.1}/100", score).yellow()
    } else {
        format!("{:.1}/100", score).red()
    };
    println!("  Robustness score: {}", score_colored);

    let verdict = if result.is_robust() {
        "ROBUST".green().bold()
    } else {
        "NOT ROBUST".red().bold()
    };
    println!("  Verdict:          {}", verdict);
    println!();

    // Explanation of verdict
    if result.is_robust() {
        println!(
            "{}",
            "Strategy shows robust performance across simulated scenarios."
                .green()
                .dimmed()
        );
    } else {
        println!(
            "{}",
            "Strategy may not be robust. Consider:".yellow().dimmed()
        );
        if result.prob_positive_return <= 0.6 {
            println!(
                "{}",
                "  - Low probability of positive returns".yellow().dimmed()
            );
        }
        if result.prob_positive_sharpe <= 0.5 {
            println!(
                "{}",
                "  - Low probability of positive Sharpe ratio"
                    .yellow()
                    .dimmed()
            );
        }
        if result.median_return <= 0.0 {
            println!("{}", "  - Negative median return".yellow().dimmed());
        }
    }
}

fn print_monte_carlo_csv(result: &MonteCarloResult) {
    println!("metric,value");
    println!("num_simulations,{}", result.num_simulations);
    println!("num_trades,{}", result.num_trades);
    println!("mean_return,{:.4}", result.mean_return);
    println!("median_return,{:.4}", result.median_return);
    println!("return_std,{:.4}", result.return_std);
    println!("return_ci_lower,{:.4}", result.return_ci.0);
    println!("return_ci_upper,{:.4}", result.return_ci.1);
    println!("prob_positive_return,{:.4}", result.prob_positive_return);
    println!("mean_max_drawdown,{:.4}", result.mean_max_drawdown);
    println!("median_max_drawdown,{:.4}", result.median_max_drawdown);
    println!("max_drawdown_95th,{:.4}", result.max_drawdown_95th);
    println!("var,{:.4}", result.var);
    println!("cvar,{:.4}", result.cvar);
    println!("mean_sharpe,{:.4}", result.mean_sharpe);
    println!("median_sharpe,{:.4}", result.median_sharpe);
    println!("sharpe_ci_lower,{:.4}", result.sharpe_ci.0);
    println!("sharpe_ci_upper,{:.4}", result.sharpe_ci.1);
    println!("prob_positive_sharpe,{:.4}", result.prob_positive_sharpe);
    println!("robustness_score,{:.4}", result.robustness_score());
    println!("is_robust,{}", result.is_robust());
}

fn validate_data(data_path: &PathBuf) -> Result<()> {
    println!("Validating data file: {}", data_path.display());

    let bars = load_csv(data_path, &DataConfig::default())?;

    println!("\nData Summary:");
    println!("  Rows: {}", bars.len());

    if !bars.is_empty() {
        println!("  Start: {}", bars.first().unwrap().timestamp);
        println!("  End: {}", bars.last().unwrap().timestamp);

        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        let min_price = closes.iter().fold(f64::INFINITY, |a: f64, &b| a.min(b));
        let max_price = closes.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));
        let avg_price: f64 = closes.iter().sum::<f64>() / closes.len() as f64;

        println!("  Price Range: {:.2} - {:.2}", min_price, max_price);
        println!("  Average Price: {:.2}", avg_price);

        let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
        let avg_volume: f64 = volumes.iter().sum::<f64>() / volumes.len() as f64;
        println!("  Average Volume: {:.0}", avg_volume);
    }

    println!("\nValidation: PASSED");
    Ok(())
}

// Experiment management functions

fn log_experiment_to_store(result: &mantis::BacktestResult, duration_ms: i64) -> Result<()> {
    ensure_store_directory()?;
    let store = ExperimentStore::new(default_store_path())?;
    let record = ExperimentRecord::from_backtest_result(result, Some(duration_ms))?;
    store.save(&record)?;
    Ok(())
}

fn experiments_list(
    strategy: &Option<String>,
    min_sharpe: Option<f64>,
    max_drawdown: Option<f64>,
    limit: Option<usize>,
    sort_by: &str,
    desc: bool,
) -> Result<()> {
    let store = ExperimentStore::new(default_store_path())?;

    let filter = ExperimentFilter {
        strategy_name: strategy.clone(),
        min_sharpe,
        max_drawdown,
        limit,
        sort_by: Some(sort_by.to_string()),
        sort_desc: desc,
        ..Default::default()
    };

    let experiments = store.list(&filter)?;

    if experiments.is_empty() {
        println!("No experiments found matching filters.");
        return Ok(());
    }

    println!(
        "\n{}",
        format!("Found {} experiments", experiments.len()).bold()
    );
    println!("Store: {}\n", default_store_path().dimmed());

    let mut builder = Builder::default();
    builder.push_record(vec![
        "ID".bold().to_string(),
        "Timestamp".bold().to_string(),
        "Strategy".bold().to_string(),
        "Symbols".bold().to_string(),
        "Return %".bold().to_string(),
        "Sharpe".bold().to_string(),
        "Max DD %".bold().to_string(),
        "Trades".bold().to_string(),
        "Win Rate %".bold().to_string(),
    ]);

    for exp in &experiments {
        let id_short = exp.experiment_id.chars().take(8).collect::<String>();
        let return_colored = if exp.total_return > 0.0 {
            format!("{:>8.2}", exp.total_return).green()
        } else {
            format!("{:>8.2}", exp.total_return).red()
        };

        let sharpe_colored = if exp.sharpe_ratio > 1.0 {
            format!("{:>6.2}", exp.sharpe_ratio).green()
        } else if exp.sharpe_ratio > 0.0 {
            format!("{:>6.2}", exp.sharpe_ratio).yellow()
        } else {
            format!("{:>6.2}", exp.sharpe_ratio).red()
        };

        let dd_colored = format!("{:>8.2}", exp.max_drawdown).red();

        builder.push_record(vec![
            id_short.bright_cyan().to_string(),
            exp.timestamp.format("%Y-%m-%d %H:%M").to_string(),
            exp.strategy_name.clone(),
            exp.symbols.clone(),
            return_colored.to_string(),
            sharpe_colored.to_string(),
            dd_colored.to_string(),
            exp.num_trades.to_string(),
            format!("{:.1}", exp.win_rate),
        ]);
    }

    let mut table = builder.build();
    table.with(Style::rounded());
    println!("{}", table);

    println!("\nUse 'mantis experiments show <id>' to view details");

    Ok(())
}

fn experiments_show(id: &str) -> Result<()> {
    let store = ExperimentStore::new(default_store_path())?;

    let exp = store.get(id)?.ok_or_else(|| {
        mantis::BacktestError::ConfigError(format!("Experiment not found: {}", id))
    })?;

    println!("\n{}", "Experiment Details".bold().underline());
    println!();

    println!("{}: {}", "ID".bold(), exp.experiment_id.bright_cyan());
    println!(
        "{}: {}",
        "Timestamp".bold(),
        exp.timestamp.format("%Y-%m-%d %H:%M:%S")
    );
    if let Some(duration) = exp.duration_ms {
        println!("{}: {}ms", "Duration".bold(), duration);
    }
    println!(
        "{}: {}",
        "Strategy".bold(),
        exp.strategy_name.bright_yellow()
    );
    println!("{}: {}", "Symbols".bold(), exp.symbols);
    println!();

    println!("{}", "Git Information".bold().underline());
    if let Some(commit) = &exp.git_commit {
        println!("{}: {}", "Commit".bold(), commit.bright_blue());
    }
    if let Some(branch) = &exp.git_branch {
        println!("{}: {}", "Branch".bold(), branch);
    }
    println!(
        "{}: {}",
        "Dirty".bold(),
        if exp.git_dirty {
            "Yes".red()
        } else {
            "No".green()
        }
    );
    println!();

    println!("{}", "Performance Metrics".bold().underline());

    let return_colored = if exp.total_return > 0.0 {
        format!("{:.2}%", exp.total_return).green()
    } else {
        format!("{:.2}%", exp.total_return).red()
    };
    println!("{}: {}", "Total Return".bold(), return_colored);

    let sharpe_colored = if exp.sharpe_ratio > 1.0 {
        format!("{:.3}", exp.sharpe_ratio).green()
    } else if exp.sharpe_ratio > 0.0 {
        format!("{:.3}", exp.sharpe_ratio).yellow()
    } else {
        format!("{:.3}", exp.sharpe_ratio).red()
    };
    println!("{}: {}", "Sharpe Ratio".bold(), sharpe_colored);
    println!("{}: {:.3}", "Sortino Ratio".bold(), exp.sortino_ratio);
    println!("{}: {:.3}", "Calmar Ratio".bold(), exp.calmar_ratio);
    println!(
        "{}: {}",
        "Max Drawdown".bold(),
        format!("{:.2}%", exp.max_drawdown).red()
    );
    println!();

    println!("{}", "Trade Statistics".bold().underline());
    println!("{}: {}", "Number of Trades".bold(), exp.num_trades);
    println!("{}: {:.1}%", "Win Rate".bold(), exp.win_rate);
    println!("{}: {:.2}", "Profit Factor".bold(), exp.profit_factor);
    println!();

    if let Some(tags) = &exp.tags {
        println!("{}: {}", "Tags".bold(), tags.bright_magenta());
    }

    if let Some(notes) = &exp.notes {
        println!("{}: {}", "Notes".bold(), notes);
        println!();
    }

    println!("{}: {}", "Config Hash".bold(), exp.config_hash.dimmed());

    Ok(())
}

fn experiments_compare(id1: &str, id2: &str) -> Result<()> {
    let store = ExperimentStore::new(default_store_path())?;

    let exp1 = store.get(id1)?.ok_or_else(|| {
        mantis::BacktestError::ConfigError(format!("Experiment not found: {}", id1))
    })?;
    let exp2 = store.get(id2)?.ok_or_else(|| {
        mantis::BacktestError::ConfigError(format!("Experiment not found: {}", id2))
    })?;

    println!("\n{}", "Experiment Comparison".bold().underline());
    println!();

    let mut builder = Builder::default();
    builder.push_record(vec![
        "Metric".bold().to_string(),
        format!("Experiment 1 ({})", id1.chars().take(8).collect::<String>())
            .bold()
            .to_string(),
        format!("Experiment 2 ({})", id2.chars().take(8).collect::<String>())
            .bold()
            .to_string(),
        "Difference".bold().to_string(),
    ]);

    // Strategy
    builder.push_record(vec![
        "Strategy".to_string(),
        exp1.strategy_name.clone(),
        exp2.strategy_name.clone(),
        "-".to_string(),
    ]);

    // Symbols
    builder.push_record(vec![
        "Symbols".to_string(),
        exp1.symbols.clone(),
        exp2.symbols.clone(),
        "-".to_string(),
    ]);

    // Total Return
    let return_diff = exp1.total_return - exp2.total_return;
    let diff_colored = if return_diff > 0.0 {
        format!("+{:.2}%", return_diff).green()
    } else {
        format!("{:.2}%", return_diff).red()
    };
    builder.push_record(vec![
        "Total Return %".to_string(),
        format!("{:.2}%", exp1.total_return),
        format!("{:.2}%", exp2.total_return),
        diff_colored.to_string(),
    ]);

    // Sharpe
    let sharpe_diff = exp1.sharpe_ratio - exp2.sharpe_ratio;
    let sharpe_diff_colored = if sharpe_diff > 0.0 {
        format!("+{:.3}", sharpe_diff).green()
    } else {
        format!("{:.3}", sharpe_diff).red()
    };
    builder.push_record(vec![
        "Sharpe Ratio".to_string(),
        format!("{:.3}", exp1.sharpe_ratio),
        format!("{:.3}", exp2.sharpe_ratio),
        sharpe_diff_colored.to_string(),
    ]);

    // Sortino
    let sortino_diff = exp1.sortino_ratio - exp2.sortino_ratio;
    builder.push_record(vec![
        "Sortino Ratio".to_string(),
        format!("{:.3}", exp1.sortino_ratio),
        format!("{:.3}", exp2.sortino_ratio),
        format!("{:+.3}", sortino_diff),
    ]);

    // Calmar
    let calmar_diff = exp1.calmar_ratio - exp2.calmar_ratio;
    builder.push_record(vec![
        "Calmar Ratio".to_string(),
        format!("{:.3}", exp1.calmar_ratio),
        format!("{:.3}", exp2.calmar_ratio),
        format!("{:+.3}", calmar_diff),
    ]);

    // Max Drawdown
    let dd_diff = exp1.max_drawdown - exp2.max_drawdown;
    let dd_diff_colored = if dd_diff < 0.0 {
        format!("{:.2}%", dd_diff).green()
    } else {
        format!("+{:.2}%", dd_diff).red()
    };
    builder.push_record(vec![
        "Max Drawdown %".to_string(),
        format!("{:.2}%", exp1.max_drawdown),
        format!("{:.2}%", exp2.max_drawdown),
        dd_diff_colored.to_string(),
    ]);

    // Trades
    let trades_diff = exp1.num_trades - exp2.num_trades;
    builder.push_record(vec![
        "Number of Trades".to_string(),
        exp1.num_trades.to_string(),
        exp2.num_trades.to_string(),
        format!("{:+}", trades_diff),
    ]);

    // Win Rate
    let win_rate_diff = exp1.win_rate - exp2.win_rate;
    builder.push_record(vec![
        "Win Rate %".to_string(),
        format!("{:.1}%", exp1.win_rate),
        format!("{:.1}%", exp2.win_rate),
        format!("{:+.1}%", win_rate_diff),
    ]);

    // Profit Factor
    let pf_diff = exp1.profit_factor - exp2.profit_factor;
    builder.push_record(vec![
        "Profit Factor".to_string(),
        format!("{:.2}", exp1.profit_factor),
        format!("{:.2}", exp2.profit_factor),
        format!("{:+.2}", pf_diff),
    ]);

    let mut table = builder.build();
    table.with(Style::rounded());
    println!("{}", table);
    println!();

    Ok(())
}

fn experiments_tag(id: &str, tags: &[String]) -> Result<()> {
    let store = ExperimentStore::new(default_store_path())?;

    // Verify experiment exists and get full ID
    let exp = store.get(id)?.ok_or_else(|| {
        mantis::BacktestError::ConfigError(format!("Experiment not found: {}", id))
    })?;

    store.add_tags(&exp.experiment_id, tags)?;

    println!(
        "{} Added tags to experiment {}: {}",
        "Success:".green().bold(),
        id.bright_cyan(),
        tags.join(", ").bright_magenta()
    );

    Ok(())
}

fn experiments_note(id: &str, note: &str) -> Result<()> {
    let store = ExperimentStore::new(default_store_path())?;

    // Verify experiment exists and get full ID
    let exp = store.get(id)?.ok_or_else(|| {
        mantis::BacktestError::ConfigError(format!("Experiment not found: {}", id))
    })?;

    store.add_notes(&exp.experiment_id, note)?;

    println!(
        "{} Added note to experiment {}",
        "Success:".green().bold(),
        id.bright_cyan()
    );

    Ok(())
}

fn experiments_delete(id: &str) -> Result<()> {
    let store = ExperimentStore::new(default_store_path())?;

    // Verify experiment exists
    let exp = store.get(id)?.ok_or_else(|| {
        mantis::BacktestError::ConfigError(format!("Experiment not found: {}", id))
    })?;

    println!("Are you sure you want to delete this experiment?");
    println!("  ID: {}", exp.experiment_id.bright_cyan());
    println!("  Strategy: {}", exp.strategy_name);
    println!("  Timestamp: {}", exp.timestamp.format("%Y-%m-%d %H:%M:%S"));
    println!();
    println!("Type 'yes' to confirm deletion:");

    use std::io::{self, BufRead};
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;

    if line.trim().to_lowercase() == "yes" {
        store.delete(&exp.experiment_id)?;
        println!(
            "{} Deleted experiment {}",
            "Success:".green().bold(),
            id.bright_cyan()
        );
    } else {
        println!("Deletion cancelled.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse() {
        let cli = Cli::try_parse_from([
            "mantis",
            "run",
            "-d",
            "test.csv",
            "-s",
            "TEST",
            "-S",
            "sma-crossover",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_strategies_command() {
        let cli = Cli::try_parse_from(["mantis", "strategies"]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_walk_forward_parse() {
        let cli = Cli::try_parse_from([
            "mantis",
            "walk-forward",
            "-d",
            "test.csv",
            "-s",
            "TEST",
            "--folds",
            "3",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_default_param_grid_for_sma() {
        let params = default_param_grid(StrategyType::SmaCrossover);
        assert!(params.is_some());
        assert!(!params.unwrap().is_empty());
    }

    #[test]
    fn test_walk_forward_metric_arg_conversion() {
        let metric: WalkForwardMetric = WalkForwardMetricArg::ProfitFactor.into();
        assert!(matches!(metric, WalkForwardMetric::ProfitFactor));
    }
}

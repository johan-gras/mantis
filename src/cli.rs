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
use mantis::monte_carlo::{MonteCarloConfig, MonteCarloResult, MonteCarloSimulator};
use mantis::portfolio::{CostModel, MarginConfig};
use mantis::sensitivity::{
    ParameterRange, SensitivityAnalysis, SensitivityConfig, SensitivityMetric,
};
use mantis::strategies::{
    BreakoutStrategy, MacdStrategy, MeanReversion, MomentumStrategy, RsiStrategy, SmaCrossover,
};
use mantis::strategy::Strategy;
use mantis::types::{AssetClass, AssetConfig, ExecutionPrice, LotSelectionMethod, Verdict};
use mantis::walkforward::{
    WalkForwardAnalyzer, WalkForwardConfig, WalkForwardMetric, WalkForwardResult,
};

use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;
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

        /// Annual borrow cost for short positions (e.g., 3.0 for 3%)
        #[arg(long, default_value = "3.0")]
        borrow_cost: f64,

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
        #[arg(long, default_value = "1")]
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

        /// Annual borrow cost for short positions (e.g., 3.0 for 3%)
        #[arg(long, default_value = "3.0")]
        borrow_cost: f64,

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
        #[arg(long, default_value = "1")]
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

    /// Run parameter sensitivity analysis to detect fragile strategies
    Sensitivity {
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

        /// Metric to analyze (sharpe, sortino, return, calmar, profit-factor, win-rate, max-drawdown)
        #[arg(long, value_enum, default_value = "sharpe")]
        metric: SensitivityMetricArg,

        /// Number of steps for parameter ranges
        #[arg(long, default_value = "5")]
        steps: usize,

        /// Export heatmap to CSV file
        #[arg(long)]
        heatmap_output: Option<PathBuf>,

        /// Data file format (auto-detects from extension if not specified)
        #[arg(short, long, value_enum, default_value = "auto")]
        format: DataFormatArg,
    },
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
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
#[value(rename_all = "kebab-case")]
pub enum SensitivityMetricArg {
    Sharpe,
    Sortino,
    Return,
    Calmar,
    ProfitFactor,
    WinRate,
    MaxDrawdown,
}

impl From<SensitivityMetricArg> for SensitivityMetric {
    fn from(arg: SensitivityMetricArg) -> Self {
        match arg {
            SensitivityMetricArg::Sharpe => SensitivityMetric::Sharpe,
            SensitivityMetricArg::Sortino => SensitivityMetric::Sortino,
            SensitivityMetricArg::Return => SensitivityMetric::Return,
            SensitivityMetricArg::Calmar => SensitivityMetric::Calmar,
            SensitivityMetricArg::ProfitFactor => SensitivityMetric::ProfitFactor,
            SensitivityMetricArg::WinRate => SensitivityMetric::WinRate,
            SensitivityMetricArg::MaxDrawdown => SensitivityMetric::MaxDrawdown,
        }
    }
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
            borrow_cost,
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
            *borrow_cost,
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
            borrow_cost,
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
            *borrow_cost,
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

        Commands::Sensitivity {
            data,
            symbol,
            strategy,
            capital,
            position_size,
            commission,
            slippage,
            metric,
            steps,
            heatmap_output,
            format,
        } => run_sensitivity(
            data,
            symbol,
            *strategy,
            *capital,
            *position_size,
            *commission,
            *slippage,
            *metric,
            *steps,
            heatmap_output.clone(),
            *format,
            cli.output,
        ),
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
    borrow_cost: f64,
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
        borrow_cost_rate: borrow_cost / 100.0,
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

    let result = engine.run(strategy.as_mut(), symbol)?;

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
    borrow_cost: f64,
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
        borrow_cost_rate: borrow_cost / 100.0,
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

    // Use the Verdict enum for three-way classification
    let verdict = result.verdict();
    let verdict_colored = match verdict {
        Verdict::Robust => verdict.label().to_uppercase().green().bold(),
        Verdict::Borderline => verdict.label().to_uppercase().yellow().bold(),
        Verdict::LikelyOverfit => verdict.label().to_uppercase().red().bold(),
    };
    println!("\nOverall Verdict: {}", verdict_colored);
    println!("  {}", verdict.description().dimmed());
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
        block_bootstrap: true, // Default to block bootstrap per spec
        block_size: None,      // Auto-calculate as floor(sqrt(n))
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

/// Run parameter sensitivity analysis
#[allow(clippy::too_many_arguments)]
fn run_sensitivity(
    data_path: &PathBuf,
    symbol: &str,
    strategy_type: StrategyType,
    capital: f64,
    position_size: f64,
    commission: f64,
    slippage: f64,
    metric_arg: SensitivityMetricArg,
    steps: usize,
    heatmap_output: Option<PathBuf>,
    format: DataFormatArg,
    output: OutputFormat,
) -> Result<()> {
    info!("Loading data from: {}", data_path.display());
    let bars = load_data_with_format(data_path, &DataConfig::default(), format)?;

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

    let metric: SensitivityMetric = metric_arg.into();

    // Build sensitivity config based on strategy type
    let sensitivity_config = build_sensitivity_config(strategy_type, steps, metric);

    println!("{}", "Parameter Sensitivity Analysis".bold().underline());
    println!("Strategy: {:?}", strategy_type);
    println!("Symbol: {}", symbol);
    println!("Metric: {}", metric.display_name());
    println!(
        "Parameters: {} combinations\n",
        sensitivity_config.num_combinations()
    );

    // Run analysis based on strategy type
    let analysis =
        run_strategy_sensitivity(&config, &sensitivity_config, &bars, symbol, strategy_type)?;

    // Output results
    match output {
        OutputFormat::Text => print_sensitivity_text(&analysis),
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&analysis)?;
            println!("{}", json);
        }
        OutputFormat::Csv => {
            println!("{}", analysis.to_csv());
        }
    }

    // Export heatmap if requested
    if let Some(ref heatmap_path) = heatmap_output {
        export_heatmap(&analysis, strategy_type, heatmap_path)?;
    }

    Ok(())
}

/// Build sensitivity config based on strategy type
fn build_sensitivity_config(
    strategy_type: StrategyType,
    steps: usize,
    metric: SensitivityMetric,
) -> SensitivityConfig {
    match strategy_type {
        StrategyType::SmaCrossover => SensitivityConfig::new()
            .add_parameter("fast_period", ParameterRange::linear_int(5, 20, steps))
            .add_parameter("slow_period", ParameterRange::linear_int(20, 60, steps))
            .metric(metric)
            .with_constraint(|params| {
                params.get("fast_period").unwrap() < params.get("slow_period").unwrap()
            }),
        StrategyType::Momentum => SensitivityConfig::new()
            .add_parameter("lookback", ParameterRange::linear_int(5, 30, steps))
            .add_parameter("threshold", ParameterRange::linear(0.0, 0.05, steps))
            .metric(metric),
        StrategyType::MeanReversion => SensitivityConfig::new()
            .add_parameter("period", ParameterRange::linear_int(10, 30, steps))
            .add_parameter("entry_std", ParameterRange::linear(1.5, 3.0, steps))
            .metric(metric),
        StrategyType::Rsi => SensitivityConfig::new()
            .add_parameter("period", ParameterRange::linear_int(7, 21, steps))
            .add_parameter("oversold", ParameterRange::linear(20.0, 35.0, steps))
            .add_parameter("overbought", ParameterRange::linear(65.0, 80.0, steps))
            .metric(metric),
        StrategyType::Breakout => SensitivityConfig::new()
            .add_parameter("entry_period", ParameterRange::linear_int(20, 50, steps))
            .add_parameter("exit_period", ParameterRange::linear_int(10, 25, steps))
            .metric(metric)
            .with_constraint(|params| {
                params.get("exit_period").unwrap() < params.get("entry_period").unwrap()
            }),
        StrategyType::Macd => SensitivityConfig::new()
            .add_parameter("fast_period", ParameterRange::linear_int(8, 16, steps))
            .add_parameter("slow_period", ParameterRange::linear_int(20, 30, steps))
            .metric(metric)
            .with_constraint(|params| {
                params.get("fast_period").unwrap() < params.get("slow_period").unwrap()
            }),
    }
}

/// Run sensitivity analysis for a specific strategy type
fn run_strategy_sensitivity(
    config: &BacktestConfig,
    sensitivity_config: &SensitivityConfig,
    bars: &[mantis::types::Bar],
    symbol: &str,
    strategy_type: StrategyType,
) -> Result<SensitivityAnalysis> {
    let analysis = match strategy_type {
        StrategyType::SmaCrossover => {
            SensitivityAnalysis::run(config, sensitivity_config, bars, symbol, |params| {
                SmaCrossover::new(
                    *params.get("fast_period").unwrap() as usize,
                    *params.get("slow_period").unwrap() as usize,
                )
            })
        }
        StrategyType::Momentum => {
            SensitivityAnalysis::run(config, sensitivity_config, bars, symbol, |params| {
                MomentumStrategy::new(
                    *params.get("lookback").unwrap() as usize,
                    *params.get("threshold").unwrap(),
                )
            })
        }
        StrategyType::MeanReversion => {
            SensitivityAnalysis::run(config, sensitivity_config, bars, symbol, |params| {
                MeanReversion::new(
                    *params.get("period").unwrap() as usize,
                    2.0, // num_std (fixed)
                    *params.get("entry_std").unwrap(),
                    0.5, // Fixed exit_std
                )
            })
        }
        StrategyType::Rsi => {
            SensitivityAnalysis::run(config, sensitivity_config, bars, symbol, |params| {
                RsiStrategy::new(
                    *params.get("period").unwrap() as usize,
                    *params.get("oversold").unwrap(),
                    *params.get("overbought").unwrap(),
                )
            })
        }
        StrategyType::Breakout => {
            SensitivityAnalysis::run(config, sensitivity_config, bars, symbol, |params| {
                BreakoutStrategy::new(
                    *params.get("entry_period").unwrap() as usize,
                    *params.get("exit_period").unwrap() as usize,
                )
            })
        }
        StrategyType::Macd => {
            SensitivityAnalysis::run(config, sensitivity_config, bars, symbol, |params| {
                MacdStrategy::new(
                    *params.get("fast_period").unwrap() as usize,
                    *params.get("slow_period").unwrap() as usize,
                    9, // Fixed signal period
                )
            })
        }
    };

    analysis.map_err(|e| {
        mantis::BacktestError::ConfigError(format!("Sensitivity analysis failed: {}", e))
    })
}

/// Print sensitivity analysis results in text format
fn print_sensitivity_text(analysis: &SensitivityAnalysis) {
    let summary = analysis.summary();
    println!("{}", summary);

    // Print best parameters
    if let Some(best) = analysis.best_result() {
        println!("{}", "Best Parameters".bold());
        println!("{}", "-".repeat(40));
        for (name, value) in &best.params {
            println!("  {}: {:.2}", name, value);
        }
        println!(
            "\n  {} = {:.4}",
            analysis.config.metric.display_name(),
            best.metric_value
        );
        println!("  Sharpe: {:.3}", best.result.sharpe_ratio);
        println!("  Return: {:.2}%", best.result.total_return_pct);
        println!("  Max DD: {:.2}%", best.result.max_drawdown_pct);
    }

    // Print cliffs (sharp drops)
    if !analysis.cliffs.is_empty() {
        println!(
            "\n{}",
            "⚠️  Detected Cliffs (Sharp Performance Drops)"
                .yellow()
                .bold()
        );
        println!("{}", "-".repeat(50));
        for cliff in &analysis.cliffs {
            println!(
                "  {}: {} → {} causes {:.1}% drop",
                cliff.parameter, cliff.value_before, cliff.value_after, cliff.drop_pct
            );
        }
    }

    // Print plateaus (stable regions)
    if !analysis.plateaus.is_empty() {
        println!(
            "\n{}",
            "✓ Detected Plateaus (Stable Regions)".green().bold()
        );
        println!("{}", "-".repeat(50));
        for plateau in &analysis.plateaus {
            println!(
                "  {}: {:.2} to {:.2} (avg metric: {:.3})",
                plateau.parameter, plateau.start_value, plateau.end_value, plateau.avg_metric
            );
        }
    }

    // Print parameter importance
    let importance = analysis.parameter_importance();
    if !importance.is_empty() {
        println!("\n{}", "Parameter Importance".bold());
        println!("{}", "-".repeat(40));
        for (name, imp) in importance {
            let bar_len = (imp * 20.0).round() as usize;
            let bar = "█".repeat(bar_len);
            println!("  {:15} {:5.1}% {}", name, imp * 100.0, bar);
        }
    }

    // Print verdict
    println!("\n{}", "Verdict".bold());
    println!("{}", "-".repeat(40));
    if analysis.is_fragile(0.5) {
        println!(
            "{}",
            "  ⚠️  Strategy appears FRAGILE - performance varies significantly with parameter changes"
                .yellow()
        );
        println!(
            "{}",
            "  Consider: Using parameters from stable plateau regions, or simplifying strategy"
                .yellow()
                .dimmed()
        );
    } else {
        println!(
            "{}",
            "  ✓ Strategy appears ROBUST - consistent performance across parameter space".green()
        );
    }
}

/// Export heatmap to CSV file
fn export_heatmap(
    analysis: &SensitivityAnalysis,
    _strategy_type: StrategyType,
    path: &PathBuf,
) -> Result<()> {
    // Get the first two parameters for heatmap
    let params: Vec<&String> = analysis.config.parameters.keys().collect();
    if params.len() < 2 {
        println!("Note: Heatmap requires at least 2 parameters. Exporting full results instead.");
        fs::write(path, analysis.to_csv())?;
        return Ok(());
    }

    let x_param = params[0];
    let y_param = params[1];

    if let Some(heatmap) = analysis.heatmap(x_param, y_param) {
        fs::write(path, heatmap.to_csv())?;
        println!(
            "\nHeatmap exported to: {} ({} vs {})",
            path.display(),
            x_param,
            y_param
        );
    } else {
        println!("Could not generate heatmap for {} vs {}", x_param, y_param);
        fs::write(path, analysis.to_csv())?;
    }

    Ok(())
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

    // Use the Verdict enum for three-way classification
    let verdict = result.verdict();
    let verdict_colored = match verdict {
        Verdict::Robust => verdict.label().to_uppercase().green().bold(),
        Verdict::Borderline => verdict.label().to_uppercase().yellow().bold(),
        Verdict::LikelyOverfit => verdict.label().to_uppercase().red().bold(),
    };
    println!("  Verdict:          {}", verdict_colored);
    println!();

    // Explanation based on verdict
    match verdict {
        Verdict::Robust => {
            println!(
                "{}",
                "Strategy shows robust performance across simulated scenarios."
                    .green()
                    .dimmed()
            );
        }
        Verdict::Borderline => {
            println!(
                "{}",
                "Strategy shows borderline performance. Proceed with caution."
                    .yellow()
                    .dimmed()
            );
            print_monte_carlo_warnings(result);
        }
        Verdict::LikelyOverfit => {
            println!("{}", "Strategy may not be robust. Consider:".red().dimmed());
            print_monte_carlo_warnings(result);
        }
    }
}

fn print_monte_carlo_warnings(result: &MonteCarloResult) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};
    use mantis::engine::BacktestResult;
    use mantis::sensitivity::{Cliff, ParameterResult, Plateau};
    use mantis::walkforward::{WalkForwardWindow, WindowResult};
    use mantis::CostScenario;
    use std::collections::HashMap;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use uuid::Uuid;

    fn sample_data_path() -> PathBuf {
        let path = PathBuf::from("data/sample.csv");
        assert!(path.exists(), "sample data missing: {:?}", path);
        path
    }

    fn make_test_result(
        strategy: &str,
        total_return_pct: f64,
        sharpe_ratio: f64,
        max_drawdown_pct: f64,
    ) -> BacktestResult {
        let start_time = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end_time = start_time + Duration::days(10);
        BacktestResult {
            strategy_name: strategy.to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100_000.0,
            final_equity: 100_000.0 * (1.0 + total_return_pct / 100.0),
            total_return_pct,
            annual_return_pct: total_return_pct,
            trading_days: 10,
            total_trades: 4,
            winning_trades: 2,
            losing_trades: 2,
            win_rate: 50.0,
            avg_win: 100.0,
            avg_loss: -100.0,
            profit_factor: 1.0,
            max_drawdown_pct,
            sharpe_ratio,
            sortino_ratio: sharpe_ratio * 1.2,
            calmar_ratio: if max_drawdown_pct.abs() > 0.0 {
                total_return_pct / max_drawdown_pct.abs()
            } else {
                0.0
            },
            trades: Vec::new(),
            equity_curve: Vec::new(),
            start_time,
            end_time,
            experiment_id: Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: HashMap::new(),
            seed: Some(42),
        }
    }

    fn make_walk_forward_result() -> WalkForwardResult {
        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let window = WalkForwardWindow {
            index: 0,
            is_start: start,
            is_end: start + Duration::days(4),
            oos_start: start + Duration::days(5),
            oos_end: start + Duration::days(9),
            is_bars: 5,
            oos_bars: 5,
        };

        WalkForwardResult {
            config: WalkForwardConfig {
                num_windows: 1,
                in_sample_ratio: 0.7,
                anchored: true,
                min_bars_per_window: 5,
            },
            windows: vec![WindowResult {
                window,
                in_sample_result: make_test_result("IS", 5.0, 1.2, -5.0),
                out_of_sample_result: make_test_result("OOS", 2.5, 0.6, -6.0),
                efficiency_ratio: 0.5,
                parameter_hash: 123,
            }],
            combined_oos_return: 2.5,
            avg_is_return: 5.0,
            avg_oos_return: 2.5,
            avg_efficiency_ratio: 0.5,
            walk_forward_efficiency: 0.5,
            avg_is_sharpe: 1.2,
            avg_oos_sharpe: 0.6,
            oos_sharpe_threshold_met: true,
            parameter_stability: 0.8,
        }
    }

    fn make_cost_sensitivity_analysis() -> CostSensitivityAnalysis {
        let base = make_test_result("Base", 4.0, 1.0, -8.0);
        let scenario = |multiplier: f64, ret: f64, sharpe: f64| CostScenario {
            multiplier,
            result: make_test_result("Test", ret, sharpe, -10.0),
            total_costs: 10.0 * multiplier,
            avg_cost_per_trade: 1.0 * multiplier,
        };

        CostSensitivityAnalysis {
            scenarios: vec![
                CostScenario {
                    multiplier: 1.0,
                    result: base,
                    total_costs: 10.0,
                    avg_cost_per_trade: 1.0,
                },
                scenario(2.0, 2.0, 0.7),
                scenario(5.0, 1.0, 0.3),
                scenario(10.0, -1.0, -0.2),
            ],
            symbol: "TEST".to_string(),
            strategy_name: "TestStrategy".to_string(),
        }
    }

    fn make_sensitivity_analysis() -> SensitivityAnalysis {
        let config = SensitivityConfig::new()
            .add_parameter("fast", ParameterRange::discrete(vec![5.0, 10.0]))
            .add_parameter("slow", ParameterRange::discrete(vec![20.0, 30.0]))
            .metric(SensitivityMetric::Sharpe);

        let mut params_a = HashMap::new();
        params_a.insert("fast".to_string(), 5.0);
        params_a.insert("slow".to_string(), 20.0);

        let mut params_b = HashMap::new();
        params_b.insert("fast".to_string(), 10.0);
        params_b.insert("slow".to_string(), 30.0);

        SensitivityAnalysis {
            results: vec![
                ParameterResult {
                    params: params_a,
                    metric_value: 1.2,
                    result: make_test_result("SensA", 3.0, 1.2, -4.0),
                },
                ParameterResult {
                    params: params_b,
                    metric_value: 0.4,
                    result: make_test_result("SensB", 1.0, 0.4, -6.0),
                },
            ],
            strategy_name: "TestStrategy".to_string(),
            symbol: "TEST".to_string(),
            config,
            cliffs: vec![Cliff {
                parameter: "fast".to_string(),
                value_before: 5.0,
                value_after: 10.0,
                metric_before: 1.2,
                metric_after: 0.4,
                drop_pct: 66.0,
            }],
            plateaus: vec![Plateau {
                parameter: "slow".to_string(),
                start_value: 20.0,
                end_value: 30.0,
                avg_metric: 0.8,
                std_metric: 0.05,
            }],
        }
    }

    fn make_monte_carlo_result() -> MonteCarloResult {
        let mut percentiles = HashMap::new();
        percentiles.insert("5th".to_string(), -2.0);
        percentiles.insert("10th".to_string(), -1.0);
        percentiles.insert("25th".to_string(), 0.0);
        percentiles.insert("50th".to_string(), 1.0);
        percentiles.insert("75th".to_string(), 2.0);
        percentiles.insert("90th".to_string(), 3.0);
        percentiles.insert("95th".to_string(), 4.0);

        MonteCarloResult {
            config: MonteCarloConfig {
                num_simulations: 100,
                confidence_level: 0.9,
                seed: Some(42),
                resample_trades: true,
                shuffle_returns: false,
                block_bootstrap: true,
                block_size: None,
            },
            num_simulations: 100,
            num_trades: 25,
            mean_return: -1.0,
            median_return: -0.5,
            return_std: 2.0,
            return_ci: (-3.0, 1.0),
            prob_positive_return: 0.4,
            mean_max_drawdown: -12.0,
            median_max_drawdown: -10.0,
            max_drawdown_ci: (-20.0, -5.0),
            max_drawdown_95th: -22.0,
            mean_sharpe: -0.2,
            median_sharpe: -0.3,
            sharpe_ci: (-0.8, 0.1),
            prob_positive_sharpe: 0.4,
            var: -4.0,
            cvar: -6.0,
            return_distribution: vec![-2.0, -1.0, 0.0, 1.0],
            sharpe_distribution: vec![-0.5, -0.2, 0.0, 0.1],
            drawdown_distribution: vec![-10.0, -12.0, -15.0],
            return_percentiles: percentiles,
        }
    }

    fn run_backtest_with(
        sizing_method: PositionSizingMethodArg,
        fixed_dollar: Option<f64>,
        target_vol: Option<f64>,
        risk_per_trade: Option<f64>,
        output: OutputFormat,
    ) -> Result<()> {
        let data_path = sample_data_path();
        run_backtest(
            &data_path,
            "TEST",
            StrategyType::SmaCrossover,
            10_000.0,
            0.02,
            sizing_method,
            fixed_dollar,
            target_vol,
            20,
            risk_per_trade,
            2.0,
            14,
            0.1,
            0.1,
            3.0,
            None,
            ExecutionPriceArg::Open,
            1.0,
            5,
            LotSelectionArg::Fifo,
            true,
            10.0,
            0.5,
            1.5,
            0.25,
            0.30,
            0.03,
            false,
            0.15,
            true,
            3,
            5,
            14,
            5,
            DataFormatArg::Csv,
            AssetClassArg::Equity,
            1.0,
            0.01,
            0.5,
            8,
            8,
            0.0001,
            1.0,
            None,
            Some(42),
            output,
        )
    }

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

    #[test]
    fn test_enum_conversions() {
        assert_eq!(
            ExecutionPrice::from(ExecutionPriceArg::Midpoint),
            ExecutionPrice::Midpoint
        );
        match LotSelectionMethod::from(LotSelectionArg::HighestCost) {
            LotSelectionMethod::HighestCost => {}
            other => panic!("unexpected lot selection: {:?}", other),
        }
        assert_eq!(
            SensitivityMetric::from(SensitivityMetricArg::WinRate),
            SensitivityMetric::WinRate
        );
    }

    #[test]
    fn test_build_asset_config_variants() {
        let equity = build_asset_config(
            "EQ",
            AssetClassArg::Equity,
            1.0,
            0.01,
            0.5,
            8,
            8,
            0.0001,
            1.0,
            None,
        );
        assert!(matches!(equity.asset_class, AssetClass::Equity));

        let future = build_asset_config(
            "FUT",
            AssetClassArg::Future,
            50.0,
            0.25,
            0.1,
            8,
            8,
            0.0001,
            1.0,
            None,
        );
        if let AssetClass::Future { multiplier, .. } = future.asset_class {
            assert_eq!(multiplier, 50.0);
        } else {
            panic!("expected future asset class");
        }

        let crypto = build_asset_config(
            "BTC",
            AssetClassArg::Crypto,
            1.0,
            0.01,
            0.5,
            8,
            8,
            0.0001,
            1.0,
            None,
        );
        assert!(matches!(crypto.asset_class, AssetClass::Crypto { .. }));

        let forex = build_asset_config(
            "FX",
            AssetClassArg::Forex,
            1.0,
            0.01,
            0.5,
            8,
            8,
            0.0001,
            1.0,
            None,
        );
        assert!(matches!(forex.asset_class, AssetClass::Forex { .. }));

        let option = build_asset_config(
            "OPT",
            AssetClassArg::Option,
            100.0,
            0.01,
            0.5,
            8,
            8,
            0.0001,
            1.0,
            Some("UNDER".to_string()),
        );
        if let AssetClass::Option { underlying, .. } = option.asset_class {
            assert_eq!(underlying, "UNDER");
        } else {
            panic!("expected option asset class");
        }
    }

    #[test]
    fn test_run_backtest_sizing_methods() {
        run_backtest_with(
            PositionSizingMethodArg::Percent,
            None,
            None,
            None,
            OutputFormat::Json,
        )
        .unwrap();
        run_backtest_with(
            PositionSizingMethodArg::Fixed,
            Some(1000.0),
            None,
            None,
            OutputFormat::Csv,
        )
        .unwrap();
        run_backtest_with(
            PositionSizingMethodArg::Volatility,
            None,
            Some(0.15),
            None,
            OutputFormat::Text,
        )
        .unwrap();
        run_backtest_with(
            PositionSizingMethodArg::Signal,
            None,
            None,
            None,
            OutputFormat::Text,
        )
        .unwrap();
        run_backtest_with(
            PositionSizingMethodArg::Risk,
            None,
            None,
            Some(0.001),
            OutputFormat::Text,
        )
        .unwrap();
    }

    #[test]
    fn test_run_walk_forward_and_errors() {
        let data_path = sample_data_path();

        let ok = run_walk_forward(
            &data_path,
            "TEST",
            StrategyType::SmaCrossover,
            10_000.0,
            0.1,
            0.1,
            0.1,
            3.0,
            None,
            ExecutionPriceArg::Open,
            1.0,
            5,
            LotSelectionArg::Fifo,
            true,
            2.0,
            0.5,
            1.5,
            0.25,
            0.30,
            0.03,
            false,
            0.15,
            false,
            2,
            0.7,
            true,
            5,
            WalkForwardMetricArg::Sharpe,
            DataFormatArg::Csv,
            OutputFormat::Text,
        );
        assert!(ok.is_ok());

        let err = run_walk_forward(
            &data_path,
            "TEST",
            StrategyType::SmaCrossover,
            10_000.0,
            0.1,
            0.1,
            0.1,
            3.0,
            None,
            ExecutionPriceArg::Open,
            1.0,
            5,
            LotSelectionArg::Fifo,
            true,
            2.0,
            0.5,
            1.5,
            0.25,
            0.30,
            0.03,
            false,
            0.15,
            false,
            0,
            0.7,
            true,
            5,
            WalkForwardMetricArg::Sharpe,
            DataFormatArg::Csv,
            OutputFormat::Text,
        )
        .unwrap_err();
        assert!(format!("{}", err).contains("folds"));

        let err = run_walk_forward(
            &data_path,
            "TEST",
            StrategyType::SmaCrossover,
            10_000.0,
            0.1,
            0.1,
            0.1,
            3.0,
            None,
            ExecutionPriceArg::Open,
            1.0,
            5,
            LotSelectionArg::Fifo,
            true,
            2.0,
            0.5,
            1.5,
            0.25,
            0.30,
            0.03,
            false,
            0.15,
            false,
            2,
            1.2,
            true,
            5,
            WalkForwardMetricArg::Sharpe,
            DataFormatArg::Csv,
            OutputFormat::Text,
        )
        .unwrap_err();
        assert!(format!("{}", err).contains("In-sample"));
    }

    #[test]
    fn test_run_optimization_and_prints() {
        let data_path = sample_data_path();
        let ok = run_optimization(
            &data_path,
            "TEST",
            StrategyType::SmaCrossover,
            10_000.0,
            OptimizeMetric::Return,
            OutputFormat::Csv,
        );
        assert!(ok.is_ok());

        print_strategies();

        let wf = make_walk_forward_result();
        print_walk_forward_text(&wf);
        print_walk_forward_csv(&wf);
    }

    #[test]
    fn test_run_monte_carlo_and_helpers() {
        let data_path = sample_data_path();
        let ok = run_monte_carlo(
            &data_path,
            "TEST",
            StrategyType::SmaCrossover,
            10_000.0,
            0.1,
            0.1,
            0.1,
            10,
            0.9,
            Some(42),
            true,
            false,
            3,
            5,
            14,
            5,
            DataFormatArg::Csv,
            OutputFormat::Csv,
        );
        assert!(ok.is_ok());

        let mc = make_monte_carlo_result();
        print_monte_carlo_text(&mc);
        print_monte_carlo_warnings(&mc);
        print_monte_carlo_csv(&mc);
    }

    #[test]
    fn test_run_cost_sensitivity_and_prints() {
        let data_path = sample_data_path();
        let ok = run_cost_sensitivity(
            &data_path,
            "TEST",
            StrategyType::SmaCrossover,
            10_000.0,
            0.1,
            0.1,
            0.1,
            "1,2,5,10",
            true,
            0.5,
            3,
            5,
            14,
            5,
            DataFormatArg::Csv,
            OutputFormat::Json,
        );
        assert!(ok.is_ok());

        let analysis = make_cost_sensitivity_analysis();
        print_cost_sensitivity_text(&analysis);
        print_cost_sensitivity_csv(&analysis);
    }

    #[test]
    fn test_run_sensitivity_and_export_heatmap() {
        let data_path = sample_data_path();
        let heatmap = NamedTempFile::new().unwrap();
        let ok = run_sensitivity(
            &data_path,
            "TEST",
            StrategyType::SmaCrossover,
            10_000.0,
            0.1,
            0.1,
            0.1,
            SensitivityMetricArg::Return,
            2,
            Some(heatmap.path().to_path_buf()),
            DataFormatArg::Csv,
            OutputFormat::Text,
        );
        assert!(ok.is_ok());

        let analysis = make_sensitivity_analysis();
        print_sensitivity_text(&analysis);
    }

    #[test]
    fn test_resample_and_quality_report() {
        let data_path = sample_data_path();
        let output = NamedTempFile::new().unwrap();
        let output_path = output.path().with_extension("csv");
        let ok = resample_data(
            &data_path,
            &output_path,
            ResampleIntervalArg::FiveMinutes,
            DataFormatArg::Csv,
        );
        assert!(ok.is_ok());

        let bad_output = output.path().with_extension("parquet");
        let err = resample_data(
            &data_path,
            &bad_output,
            ResampleIntervalArg::FiveMinutes,
            DataFormatArg::Csv,
        )
        .unwrap_err();
        assert!(format!("{}", err).contains("Parquet output"));

        let ok = run_quality_report(&data_path, 60 * 60 * 24, DataFormatArg::Csv);
        assert!(ok.is_ok());
    }

    #[test]
    fn test_init_run_config_and_validate() {
        let mut config_file = NamedTempFile::new().unwrap();
        let config_path = config_file.path().to_path_buf();

        let ok = init_config(&config_path);
        assert!(ok.is_ok());

        let ok = run_from_config(&config_path, OutputFormat::Json);
        assert!(ok.is_ok());

        let ok = validate_data(&sample_data_path());
        assert!(ok.is_ok());

        writeln!(config_file, "\n# updated").unwrap();
    }
}

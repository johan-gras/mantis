//! Command-line interface for the backtest engine.

use mantis::analytics::ResultFormatter;
use mantis::config::BacktestFileConfig;
use mantis::data::{load_csv, load_data, load_parquet, DataConfig};
use mantis::engine::{BacktestConfig, Engine};
use mantis::error::Result;
use mantis::features::{FeatureConfig, FeatureExtractor, TimeSeriesSplitter};
use mantis::portfolio::CostModel;
use mantis::strategies::{
    BreakoutStrategy, MacdStrategy, MeanReversion, MomentumStrategy, RsiStrategy, SmaCrossover,
};
use mantis::strategy::Strategy;

use clap::{Parser, Subcommand, ValueEnum};
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
        #[arg(short, long, default_value = "1.0")]
        position_size: f64,

        /// Commission percentage (e.g., 0.1 for 0.1%)
        #[arg(long, default_value = "0.1")]
        commission: f64,

        /// Slippage percentage (e.g., 0.05 for 0.05%)
        #[arg(long, default_value = "0.05")]
        slippage: f64,

        /// Allow short selling
        #[arg(long)]
        allow_short: bool,

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
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Csv,
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
            commission,
            slippage,
            allow_short,
            fast_period,
            slow_period,
            rsi_period,
            lookback,
            format,
        } => run_backtest(
            data,
            symbol,
            *strategy,
            *capital,
            *position_size,
            *commission,
            *slippage,
            *allow_short,
            *fast_period,
            *slow_period,
            *rsi_period,
            *lookback,
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
    }
}

#[allow(clippy::too_many_arguments)]
fn run_backtest(
    data_path: &PathBuf,
    symbol: &str,
    strategy_type: StrategyType,
    capital: f64,
    position_size: f64,
    commission: f64,
    slippage: f64,
    allow_short: bool,
    fast_period: usize,
    slow_period: usize,
    rsi_period: usize,
    lookback: usize,
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
        allow_short,
        show_progress: true,
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

    // Extract just BacktestResults, discarding the parameter info
    let mut results: Vec<mantis::BacktestResult> = match strategy_type {
        StrategyType::SmaCrossover => {
            // Generate parameter combinations
            let mut params = Vec::new();
            for fast in (5..=20).step_by(5) {
                for slow in (20..=60).step_by(10) {
                    if fast < slow {
                        params.push((fast, slow));
                    }
                }
            }

            engine
                .optimize(symbol, params, |&(fast, slow)| {
                    Box::new(SmaCrossover::new(fast, slow))
                })?
                .into_iter()
                .map(|(_, r)| r)
                .collect()
        }
        StrategyType::Momentum => {
            let params: Vec<usize> = (5..=30).step_by(5).collect();
            engine
                .optimize(symbol, params, |&lookback| {
                    Box::new(MomentumStrategy::new(lookback, 0.0))
                })?
                .into_iter()
                .map(|(_, r)| r)
                .collect()
        }
        StrategyType::Rsi => {
            let params: Vec<usize> = (7..=21).step_by(7).collect();
            engine
                .optimize(symbol, params, |&period| {
                    Box::new(RsiStrategy::new(period, 30.0, 70.0))
                })?
                .into_iter()
                .map(|(_, r)| r)
                .collect()
        }
        StrategyType::Macd => {
            // Optimize MACD with different fast/slow period combinations
            let mut params = Vec::new();
            for fast in (8..=16).step_by(4) {
                for slow in (20..=30).step_by(5) {
                    if fast < slow {
                        params.push((fast, slow, 9usize));
                    }
                }
            }
            engine
                .optimize(symbol, params, |&(fast, slow, signal)| {
                    Box::new(MacdStrategy::new(fast, slow, signal))
                })?
                .into_iter()
                .map(|(_, r)| r)
                .collect()
        }
        _ => {
            println!("Optimization not implemented for this strategy");
            return Ok(());
        }
    };

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

    match output {
        OutputFormat::Text => {
            ResultFormatter::print_table(&results);
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
}

//! Mantis - A high-performance backtesting engine for quantitative trading.
//!
//! # Overview
//!
//! Mantis is a high-performance, event-driven backtesting engine written in Rust.
//! It provides everything you need to develop, test, and analyze trading strategies:
//!
//! - **Fast execution**: Native Rust performance with parallel optimization
//! - **Realistic simulation**: Configurable slippage, commissions, and order types
//! - **Comprehensive analytics**: Sharpe, Sortino, drawdown, and many more metrics
//! - **Flexible strategies**: Easy-to-implement Strategy trait
//! - **Risk management**: Stop-loss, take-profit, and trailing stops
//! - **Walk-forward analysis**: Robust optimization to prevent overfitting
//! - **Configuration files**: TOML-based configuration for reproducible backtests
//! - **Built-in strategies**: SMA Crossover, Momentum, Mean Reversion, RSI, MACD, Breakout
//! - **ML/DL integration**: Feature extraction, external signals, train/test splitting
//!
//! # Quick Start
//!
//! ```no_run
//! use mantis::{
//!     engine::{Engine, BacktestConfig},
//!     strategies::SmaCrossover,
//!     data::load_csv,
//! };
//!
//! // Create engine with configuration
//! let config = BacktestConfig {
//!     initial_capital: 100_000.0,
//!     ..Default::default()
//! };
//! let mut engine = Engine::new(config);
//!
//! // Load data
//! let bars = load_csv("data/AAPL.csv", &Default::default()).unwrap();
//! engine.add_data("AAPL", bars);
//!
//! // Run backtest
//! let mut strategy = SmaCrossover::new(10, 30);
//! let result = engine.run(&mut strategy, "AAPL").unwrap();
//!
//! println!("Return: {:.2}%", result.total_return_pct);
//! println!("Sharpe: {:.2}", result.sharpe_ratio);
//! ```
//!
//! # Creating Custom Strategies
//!
//! Implement the `Strategy` trait to create your own trading strategies:
//!
//! ```
//! use mantis::strategy::{Strategy, StrategyContext};
//! use mantis::types::Signal;
//!
//! struct MyStrategy {
//!     threshold: f64,
//! }
//!
//! impl Strategy for MyStrategy {
//!     fn name(&self) -> &str {
//!         "My Custom Strategy"
//!     }
//!
//!     fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
//!         let price = ctx.current_bar().close;
//!         // Your logic here
//!         Signal::Hold
//!     }
//! }
//! ```
//!
//! # Technical Indicators
//!
//! The [`data`] module provides many technical indicators:
//! - SMA, EMA (Simple/Exponential Moving Averages)
//! - RSI (Relative Strength Index)
//! - MACD (Moving Average Convergence Divergence)
//! - Bollinger Bands
//! - ATR (Average True Range)
//! - Stochastic Oscillator
//! - Williams %R
//! - CCI (Commodity Channel Index)
//! - OBV (On-Balance Volume)
//! - VWAP (Volume Weighted Average Price)
//!
//! # Modules
//!
//! - [`types`]: Core data types (Bar, Order, Trade, Position)
//! - [`data`]: Data loading and technical indicators
//! - [`strategy`]: Strategy trait and context
//! - [`portfolio`]: Portfolio and position management
//! - [`engine`]: Backtest execution engine
//! - [`analytics`]: Performance metrics and reporting
//! - [`strategies`]: Built-in example strategies
//! - [`risk`]: Risk management (stop-loss, take-profit, position sizing)
//! - [`walkforward`]: Walk-forward optimization analysis
//! - [`cost_sensitivity`]: Transaction cost sensitivity analysis for robustness testing
//! - [`sensitivity`]: Parameter sensitivity analysis with heatmaps and stability detection
//! - [`options`]: Options pricing (Black-Scholes), Greeks, and derivatives support
//! - [`config`]: TOML configuration file support
//! - [`features`]: Feature extraction for ML/DL workflows

pub mod analytics;
pub mod config;
pub mod cpcv;
pub mod data;
pub mod engine;
pub mod error;
pub mod experiments;
pub mod export;
pub mod features;
pub mod metadata;
pub mod monte_carlo;
pub mod multi_asset;
// TODO: ONNX module awaiting ort crate stabilization (v2.0 API in flux, v1.x yanked)
// Infrastructure code complete in src/onnx.rs, needs: stable ort version + integration testing
// pub mod onnx;
pub mod cost_sensitivity;
pub mod options;
pub mod portfolio;
pub mod regime;
pub mod risk;
pub mod sensitivity;
pub mod strategies;
pub mod strategy;
pub mod streaming;
pub mod timeframe;
pub mod types;
pub mod validation;
pub mod viz;
pub mod walkforward;

// Python bindings (only compiled with --features python)
#[cfg(feature = "python")]
pub mod python;

// Re-exports for convenience
pub use analytics::{
    rolling_drawdown, rolling_drawdown_windowed, rolling_max_drawdown, rolling_sharpe,
    rolling_volatility, BenchmarkMetrics, DrawdownAnalysis, DrawdownPeriod, FactorAttribution,
    FactorLoadings, FactorModelType, FactorReturns, PerformanceMetrics, ResultFormatter,
    StatisticalTestResult, StatisticalTests,
};
pub use engine::{BacktestConfig, BacktestResult, Engine};
pub use error::{BacktestError, ErrorHelp, Result};
pub use options::{
    black_scholes, calculate_greeks, validate_put_call_parity, ExerciseStyle, Greeks,
    OptionContract, OptionType, SettlementType,
};
pub use strategy::Strategy;
pub use types::{
    AssetClass, AssetConfig, Bar, CorporateAction, CorporateActionType, DividendAdjustMethod,
    DividendType, ExecutionPrice, LotSelectionMethod, Order, Side, Signal, TaxLot, Trade, Verdict,
    VolumeProfile,
};

// Data handling re-exports
pub use data::{
    adjust_for_dividends, adjust_for_splits, align_series, apply_adjustment_factor,
    cumulative_adjustment_factor, data_quality_report, detect_gaps, fill_gaps,
    filter_actions_for_symbol, list_samples, load_corporate_actions, load_dir, load_multi,
    load_sample, resample, unalign_series, AlignMode, AlignedBars, DataGap, DataQualityReport,
    FillMethod, ResampleInterval,
};

// Multi-timeframe support
pub use timeframe::TimeframeManager;

// Signal validation
pub use validation::{
    validate_signal, validate_signal_quick, validate_signals, SignalStats, SignalValidationConfig,
    SignalValidationResult,
};

// Risk management and position sizing
pub use risk::{
    PositionSizer, PositionSizingMethod, RiskConfig, RiskManager, StopLoss, TakeProfit,
};

// Parameter sensitivity analysis
pub use sensitivity::{
    Cliff, HeatmapData, ParameterRange, ParameterResult, Plateau, SensitivityAnalysis,
    SensitivityConfig, SensitivityMetric, SensitivitySummary,
};

// Cost sensitivity analysis
pub use cost_sensitivity::{
    run_cost_sensitivity_analysis, CostScenario, CostSensitivityAnalysis, CostSensitivityConfig,
};

// Export utilities
pub use export::export_walkforward_html;

// Visualization utilities
pub use viz::{
    compare_strategies, equity_sparkline, export_heatmap_svg, heatmap_to_ascii, heatmap_to_svg,
    result_summary, result_with_verdict, sparkline, sparkline_with_config, walkforward_fold_chart,
    walkforward_summary, HeatmapSvgConfig, SparklineConfig, StrategyComparison,
};

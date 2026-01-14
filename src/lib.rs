//! Ralph Backtest - A production-quality backtesting engine for quantitative trading.
//!
//! # Overview
//!
//! Ralph is a high-performance, event-driven backtesting engine written in Rust.
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
//! use ralph_backtest::{
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
//! use ralph_backtest::strategy::{Strategy, StrategyContext};
//! use ralph_backtest::types::Signal;
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
//! - [`config`]: TOML configuration file support
//! - [`features`]: Feature extraction for ML/DL workflows

pub mod analytics;
pub mod config;
pub mod data;
pub mod engine;
pub mod error;
pub mod export;
pub mod features;
pub mod monte_carlo;
pub mod multi_asset;
pub mod portfolio;
pub mod regime;
pub mod risk;
pub mod strategy;
pub mod strategies;
pub mod streaming;
pub mod types;
pub mod walkforward;

// Re-exports for convenience
pub use engine::{BacktestConfig, BacktestResult, Engine};
pub use error::{BacktestError, Result};
pub use strategy::Strategy;
pub use types::{Bar, Order, Signal, Side, Trade};

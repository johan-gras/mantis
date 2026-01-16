# Changelog

All notable changes to Mantis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI workflow for automated testing across Linux, macOS, and Windows
- Python bindings build verification in CI for Python 3.9-3.12
- Benchmark compilation checks in CI
- Documentation build checks in CI

### Fixed
- Benchmark compilation errors from references to non-existent modules
- Unused import warning in `src/export.rs`
- Code formatting issues in `src/cli.rs`

## [1.0.0] - 2026-01-16

### Added

#### Core Engine
- High-performance backtest execution engine in Rust
- Signal interpretation (1/-1/0 and magnitude-based scaling)
- Order execution at next bar OPEN with configurable slippage
- Multi-symbol portfolio support with automatic aggregation
- Frequency auto-detection (1s to 1W) with proper annualization
- Stop-loss, take-profit, and trailing stop triggers
- Limit order support with TTL and gap-through handling

#### Position Sizing
- Percent of equity (default 10%)
- Fixed dollar amount
- Volatility-targeted sizing
- Signal-scaled sizing
- Risk-based/ATR sizing
- Maximum position and leverage constraints

#### Execution Realism
- Commission models: percentage, fixed, per-share
- Slippage models: fixed percentage, sqrt market impact
- Volume participation limits
- Short selling with daily borrow costs
- Margin enforcement (Reg T + portfolio margin)

#### Performance Metrics
- Core metrics: total_return, cagr, sharpe, sortino, calmar, max_drawdown, volatility
- Advanced metrics: Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR)
- Benchmark comparison: alpha, beta, tracking error, information ratio
- Rolling metrics: rolling_sharpe, rolling_drawdown, rolling_volatility
- Factor attribution: FF3, FF5, Carhart 4-factor, 6-factor models
- Statistical tests: Jarque-Bera, Durbin-Watson, ADF, Ljung-Box

#### Validation & Robustness
- Walk-forward analysis (anchored/rolling windows, configurable folds)
- Monte Carlo simulation with bootstrap resampling
- Overfitting detection via DSR and PSR
- Parameter sensitivity analysis
- Cost sensitivity analysis
- Auto-warnings for suspicious results
- CPCV (Combinatorial Purged Cross-Validation)

#### Data Handling
- CSV and Parquet loading with auto-detection
- Column auto-detection (33+ naming variants)
- Date parsing (20+ formats including Unix timestamps)
- Data validation (OHLC constraints, duplicates, gaps)
- Multi-symbol support with timestamp alignment
- Split/dividend adjustments
- 12 technical indicators (SMA, EMA, RSI, MACD, ATR, Bollinger, etc.)
- Resampling (minute to month)

#### Visualization
- Equity curve plots (SVG + ASCII sparklines)
- Drawdown charts
- Returns heatmaps
- Walk-forward fold charts
- Strategy comparison tables
- HTML report export
- Dark/light theme support

#### Python API
- Core functions: `mt.load()`, `mt.backtest()`, `mt.sweep()`, `mt.compare()`
- Results methods: `validate()`, `plot()`, `monte_carlo()`, `report()`, `save()`
- Sensitivity analysis: `mt.sensitivity()`, `mt.cost_sensitivity()`
- Parameter ranges: `linear_range`, `log_range`, `discrete_range`, `centered_range`
- Fluent API: `mt.Backtest()` class with chainable methods
- ONNX support: `mt.load_model()`, `mt.generate_signals()` (optional feature)
- Complete type stubs for IDE support

#### Documentation
- Quick start guide
- Cookbook with 7 recipes
- API reference (auto-generated)
- Concept documentation
- MkDocs with Material theme
- Bundled sample data (AAPL, SPY, BTC)
- Interactive playground links (Colab/Binder)

### Test Coverage
- 403 Rust unit tests
- 20 integration tests
- 12 doc tests (24 intentionally ignored)
- All tests passing

[Unreleased]: https://github.com/johan/mantis/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/johan/mantis/releases/tag/v1.0.0

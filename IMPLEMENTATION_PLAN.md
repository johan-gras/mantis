# Implementation Plan

Prioritized roadmap for closing gaps between current implementation and specifications.

**Last Updated:** 2026-01-16

---

## Executive Summary

The Mantis backtesting framework has a solid core implementation with excellent coverage of:
- Core backtest engine with comprehensive cost modeling
- 40+ technical indicators
- Monte Carlo simulation with confidence intervals
- Walk-forward analysis with OOS degradation metrics
- CPCV (Combinatorial Purged Cross-Validation)
- Multi-asset portfolio optimization (Black-Litterman, Mean-Variance, HRP, Risk Parity)
- Options pricing (Black-Scholes, Binomial, Greeks)
- Regime detection (trend, volatility, volume)
- Streaming indicators for real-time applications
- Feature extraction for ML (40+ features, sequence building)
- Experiment tracking with SQLite
- Comprehensive CLI with 11 commands
- Export formats: CSV, JSON, Parquet, NPY, Markdown
- Performance benchmarks (9 benchmark groups in benches/backtest_bench.rs)

**Critical Issues Discovered (Verified 2026-01-16):**
- **LOOKAHEAD BUG**: FIXED - Orders now buffer and fill at bar[i+1].open
- Python bindings (PyO3) completely missing

**Items Verified as CORRECT:**
- Config defaults: slippage (0.1%), commission (0.1%), position_size (10%)
- Limit order fill logic in portfolio.rs
- DSR/PSR implementation in analytics.rs
- Black-Litterman & Mean-Variance optimization fully implemented
- Walk-forward OOS degradation calculation correct
- Statistical tests (ADF, autocorrelation, Ljung-Box) - FIXED in commit 0b67bff
- ALL TESTS NOW PASSING

---

## Test Status Summary

**Last Run:** 2026-01-16
- **Total Tests:** 571 passed (lib) + 5 (integration) + 28 (benchmarks) + 14 (doc tests)
- **Failed:** 0 tests
- **Status:** ALL TESTS PASSING

---

## P0 - Critical (Blocking Core Functionality)

### 1. Statistical Tests [COMPLETE]
**Status:** FIXED in commit 0b67bff (2026-01-16)

All 3 previously failing tests now pass:
- ADF (Augmented Dickey-Fuller) test
- Autocorrelation test
- Ljung-Box test

**No action required.**

---

### 2. Python Bindings (PyO3) [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- PyO3 0.22 with ABI3 stable Python ABI (supports Python 3.8+)
- Maturin build system configured in pyproject.toml
- Package name: `mantis-bt` (installable via `pip install mantis-bt`)

**Files created:**
- `src/python/mod.rs` - PyO3 module wiring (_mantis native module)
- `src/python/types.rs` - PyBar, PyTrade Python-exposed types
- `src/python/data.rs` - load(), load_multi(), load_dir() functions with numpy arrays
- `src/python/backtest.rs` - backtest(), signal_check() functions, PyBacktestConfig
- `src/python/results.rs` - PyBacktestResult, PyValidationResult with metrics
- `pyproject.toml` - Maturin build configuration
- `python/mantis/__init__.py` - Python wrapper with compare(), sweep() helpers
- `python/mantis/__init__.pyi` - Type stubs for IDE autocomplete (note: `date_column` parameter was removed from load/load_multi/load_dir since it was documented but not implemented)
- `python/mantis/py.typed` - PEP 561 marker for typed package

**Features:**
- `mantis.load("file.csv")` - Returns dict with numpy arrays (timestamp, open, high, low, close, volume) and Bar objects
- `mantis.backtest(data, signal)` - Run backtest with pre-computed signal array
- `mantis.backtest(data, strategy="sma-crossover")` - Run with built-in strategy
- All 6 built-in strategies accessible: sma-crossover, momentum, mean-reversion, rsi, macd, breakout
- `signal_check()` - Validate signals before backtesting
- `compare()` - Compare multiple backtest results
- `sweep()` - Parameter sweep with callable signal generator
- BacktestResult with: total_return, sharpe, sortino, calmar, max_drawdown, win_rate, profit_factor, trades, equity_curve, warnings()
- `validate(data, signal, folds, train_ratio)` - Walk-forward validation from Python
- ValidationResult with: is_sharpe, oos_sharpe, efficiency, verdict, fold_count, is_robust()
- `fold_details()` method on ValidationResult - Returns List[FoldDetail] with per-fold metrics
- FoldDetail class with: is_sharpe, oos_sharpe, is_return, oos_return, efficiency, is_bars, oos_bars
- `results.plot(width=40)` - ASCII sparkline visualization of equity curve
- `results.save("file.json")` - Export metrics to JSON file
- `results.report("file.html")` - Generate self-contained HTML report with SVG charts
- `validation.plot(width=20)` - ASCII visualization of fold-by-fold performance
- `validation.report("file.html")` - Generate self-contained HTML report for validation results
- `results.deflated_sharpe` - Deflated Sharpe Ratio property
- `results.psr` - Probabilistic Sharpe Ratio property
- `results.metrics()` now includes deflated_sharpe and psr

**Dependencies:** None
**Effort:** Large (completed)

---

### 3. ONNX Module Re-enablement [COMPLETE]
**Status:** COMPLETE - Re-enabled with ort 2.0 API

**Implementation details:**
- ONNX module enabled as optional feature (`--features onnx`)
- Uses ort 2.0.0-rc.11 (production-ready)
- Full implementation with batch inference support (524 lines in src/onnx.rs)
- CUDA support is disabled pending additional configuration
- All unit tests passing

**Files modified:**
- `Cargo.toml` - ort 2.0.0-rc.11 and ndarray dependencies enabled under `[features]`
- `src/lib.rs` - ONNX module conditionally compiled with `#[cfg(feature = "onnx")]`
- `src/onnx.rs` - Updated to ort 2.0 API

**Dependencies:** None (ort crate now stable enough)
**Effort:** Small (completed)

---

### 3a. ONNX Python Bindings [COMPLETE]
**Status:** COMPLETE - Full Python API for ONNX inference in backtests (2026-01-16)

**Implementation details:**
- Python bindings for ONNX model loading and inference
- `OnnxModel` class for loading and running ONNX models
- `ModelConfig` class for configuring model parameters
- `InferenceStats` class for tracking inference performance
- `load_model()` convenience function
- `generate_signals()` function for batch inference
- `backtest()` now accepts `model="path.onnx"` and `features=feature_df` parameters
- Supports signal threshold for converting predictions to discrete signals
- Auto-detects feature dimensions from numpy arrays and DataFrames
- Helpful error messages with common causes and quick fixes

**Python API:**
```python
# Method 1: Use model path directly in backtest
results = mt.backtest(data, model="model.onnx", features=feature_df)

# Method 2: With signal threshold
results = mt.backtest(data, model="model.onnx", features=features, signal_threshold=0.5)

# Method 3: Load model separately
model = mt.OnnxModel("model.onnx", input_size=10)
signals = mt.generate_signals(model, features, threshold=0.5)
results = mt.backtest(data, signals)

# Method 4: Using load_model convenience function
model = mt.load_model("model.onnx", input_size=10)
prediction = model.predict([0.1, 0.2, 0.3, ...])
predictions = model.predict_batch([[...], [...], ...])
```

**Files created:**
- `src/python/onnx.rs` - PyO3 bindings for OnnxModel, ModelConfig, InferenceStats, load_model, generate_signals

**Files modified:**
- `src/python/mod.rs` - Added ONNX module registration with feature gate
- `src/python/backtest.rs` - Added model, features, model_input_size, signal_threshold parameters to backtest()
- `python/mantis/__init__.py` - Added ONNX imports with fallback for when feature not enabled
- `python/mantis/__init__.pyi` - Added type stubs for all ONNX classes and functions

**Dependencies:** Requires `--features python,onnx` to build
**Effort:** Medium (completed)

---

### 4. Helpful Error Messages [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- New error types in `src/error.rs`:
  - `SignalShapeMismatch` - Detects signal/data length mismatches with helpful guidance
  - `InvalidSignal` - Catches NaN, infinity, and out-of-range values
  - `LookaheadBias` - Warns when signals reference future data
  - `SuspiciousSignal` - Flags statistically unusual patterns
- `ErrorHelp` struct with `common_causes` and `quick_fixes` fields for user guidance
- New validation module `src/validation.rs` with:
  - `validate_signal()` - Full validation with configurable checks
  - `validate_signal_quick()` - Fast validation for hot paths
  - `validate_signals()` - Batch validation for multiple signals
  - `SignalValidationConfig` - Configuration for validation behavior
  - `SignalStats` - Statistics about signal quality
- 11 new tests for validation (all passing)

**Dependencies:** None
**Effort:** Medium (3-5 days)

---

## P1 - High Priority (Significant Feature Gaps)

### 5. Rolling Metrics [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- `rolling_sharpe(returns, window, annualization_factor)` - Rolling Sharpe ratio over sliding window
- `rolling_drawdown(equity)` - Running drawdown from peak equity
- `rolling_drawdown_windowed(equity, window)` - Drawdown with windowed peak lookback
- `rolling_max_drawdown(equity, window)` - Worst drawdown in each rolling window
- `rolling_volatility(returns, window, annualization_factor)` - Rolling annualized volatility
- All functions exported from lib.rs
- 8 unit tests added

**Dependencies:** None
**Effort:** Medium (2-3 days)

---

### 6. Short Borrow Costs [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- `borrow_cost_rate` field added to `CostModel` in portfolio.rs (default 3% annual)
- `accrue_borrow_costs()` method charges daily: position_value × (borrow_rate / 252)
- Only charges for short equity positions (futures have their own cost model)
- Called from `record_equity()` before margin interest accrual
- `borrow_cost` field added to `CostSettings` in config.rs
- CLI flags `--borrow-cost` added to Run and WalkForward commands
- `scale_cost_model()` updated to scale borrow costs in cost sensitivity analysis
- 4 unit tests added (test_borrow_cost_for_short_position, test_borrow_cost_not_charged_for_long_position, test_borrow_cost_zero_rate_no_charge, test_borrow_cost_multiple_days)

**Dependencies:** None
**Effort:** Medium (2-3 days)

---

### 7. load_multi / load_dir Functions [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- `load_multi()` function that loads multiple symbols from a HashMap<symbol, path>
- `load_dir()` function that loads all files matching a glob pattern from a directory
- Both functions support custom DataConfig
- Corresponding methods added to DataManager: `load_multi()`, `load_dir()`, `load_multi_with_config()`, `load_dir_with_config()`
- Added `glob` crate dependency for pattern matching
- 5 unit tests added for the new functionality

**Dependencies:** None
**Effort:** Small (1-2 days)

---

### 8. Cost Sensitivity CLI Command [COMPLETE]
**Status:** COMPLETE - CLI command implemented

**Implementation details:**
- Command: `mantis cost-sensitivity`
- Supports all standard backtest options (data, symbol, strategy, capital, position_size, commission, slippage)
- Custom cost multipliers via `--multipliers` (comma-separated list like "1.0,2.0,5.0,10.0")
- `--include-zero-cost` flag for theoretical upper bound analysis
- `--robustness-threshold` for Sharpe threshold at 5x costs
- Output formats: Text (default), JSON (`-o json`), CSV (`-o csv`)

**Dependencies:** None
**Effort:** Small (1 day)

---

### 9. Position Sizing Integration [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- Created `PositionSizingMethod` enum with 5 methods:
  - `PercentOfEquity` - Size based on percentage of current equity (default)
  - `FixedDollar` - Fixed dollar amount per trade
  - `VolatilityTargeted` - Target a specific annualized volatility
  - `SignalScaled` - Scale position size by signal strength (0.0-1.0)
  - `RiskBased` - Risk a fixed percentage with ATR-based stop loss
- Added new `PositionSizer` methods in `src/risk.rs`:
  - `size_fixed_dollar()` - Calculate shares for fixed dollar amount
  - `size_by_signal()` - Scale position by signal strength with max position limit
  - `size_by_volatility_target()` - Target volatility using rolling std dev
  - `size_percent_of_equity()` - Wrapper for percent-of-equity sizing
- Integrated `PositionSizingMethod` into `BacktestConfig` (`src/config.rs`)
- Updated `signal_to_order()` in `src/engine.rs` to use selected sizing method with volatility and ATR calculations
- Added CLI options in `src/cli.rs`:
  - `--sizing-method` - Select sizing method (percent, fixed, volatility, signal, risk)
  - `--fixed-dollar` - Dollar amount for fixed sizing
  - `--target-vol` - Target annualized volatility (e.g., 0.15 for 15%)
  - `--vol-lookback` - Lookback period for volatility calculation
  - `--risk-per-trade` - Risk percentage for risk-based sizing
  - `--stop-atr` - ATR multiplier for stop loss distance
  - `--atr-period` - ATR calculation period
- Added 7 new unit tests for position sizing methods (all passing)

**Dependencies:** None
**Effort:** Medium (2-3 days)

---

### 10. Multi-Symbol CLI Command [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- Added `mantis portfolio` CLI command for multi-asset backtesting
- Supports 9 portfolio allocation strategies:
  - `equal-weight` - Equal weight allocation across all assets
  - `momentum` - Momentum-based allocation (overweight recent winners)
  - `inverse-vol` - Inverse volatility allocation
  - `risk-parity` - Risk parity allocation (equal risk contribution)
  - `min-variance` - Mean-variance optimization (minimum variance)
  - `max-sharpe` - Mean-variance optimization (maximum Sharpe)
  - `hrp` - Hierarchical Risk Parity
  - `drift-equal` - Equal weight with drift-based rebalancing
  - `drift-momentum` - Momentum with drift-based rebalancing
- Loads multiple data files from directory using glob patterns
- Supports portfolio constraints: max-position, max-leverage, max-turnover, min/max-holdings
- Configurable rebalancing frequency and lookback periods
- Output formats: text (default), JSON, CSV
- Uses existing MultiAssetEngine infrastructure

**CLI usage:**
```bash
mantis portfolio -d ./data/stocks/ -p "*.csv" --strategy risk-parity --rebalance-freq 21
```

**Dependencies:** None
**Effort:** Medium (completed)

---

## P2 - Medium Priority (Enhanced Functionality)

### 11. Parameter Sensitivity Analysis [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- Created `src/sensitivity.rs` module with ~1100 lines
- `ParameterRange` enum with Linear, Logarithmic, Discrete, Centered variants
- `SensitivityConfig` for configuring parameter grids with constraints
- `SensitivityAnalysis` struct with parallel execution via rayon
- `HeatmapData` for 2D visualization-ready exports
- Cliff detection (sharp performance drops between adjacent values)
- Plateau detection (stable parameter regions)
- Parameter importance ranking
- Stability score calculation
- CLI command `mantis sensitivity` with options for metric, steps, heatmap output
- Supports all 6 built-in strategies with predefined parameter ranges
- 10 unit tests added

**Dependencies:** None
**Effort:** Medium (3-4 days)

---

### 12. Visualization Module [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- Created `src/viz.rs` module (~845 lines)
- ASCII sparklines for terminal equity curve visualization:
  - `sparkline()` - Convert any numeric slice to ASCII sparkline
  - `sparkline_with_config()` - Configurable sparkline with custom characters
  - `equity_sparkline()` - Specialized sparkline for equity curves
- Strategy comparison table:
  - `compare_strategies()` - Format multiple backtest results for comparison
  - `StrategyComparison` struct with formatted output
- Walk-forward fold visualization:
  - `walkforward_fold_chart()` - ASCII chart showing IS vs OOS performance per fold
  - `walkforward_summary()` - Summary table for walk-forward results
- SVG heatmap generation for parameter sensitivity:
  - `heatmap_to_svg()` - Generate SVG heatmap from 2D data
  - `export_heatmap_svg()` - Export heatmap to file
- ASCII heatmap for terminals:
  - `heatmap_to_ascii()` - Terminal-friendly heatmap with configurable characters
- Result summary with sparkline:
  - `result_summary()` - Backtest result summary with embedded sparkline
  - `result_with_verdict()` - Result summary including verdict classification
- All functions exported from lib.rs
- 13 unit tests added and passing

**Dependencies:** None
**Effort:** Medium (3-4 days)

---

### 13. HTML Report Generation [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- `export_report_html()` method added to `Exporter` struct
- `export_report_html()` method added to `MultiAssetExporter` struct
- Self-contained HTML report with embedded CSS (no external dependencies)
- Dark/light theme support via CSS prefers-color-scheme media query
- Report sections:
  - Summary (period, symbols, trading days)
  - Performance metrics grid (color-coded: green=positive, red=negative)
  - Trade statistics grid
  - Equity curve SVG chart (auto-downsampled for large datasets)
  - Drawdown SVG chart
  - Trade list table with P&L highlighting
- SVG charts with:
  - Y-axis labels and grid lines
  - X-axis date labels
  - Responsive viewBox for any screen size
  - Automatic color coding (green for gains, red for losses)
- 6 new tests added for HTML export functionality

**Dependencies:** None
**Effort:** Small (1 day)

---

### 14. Verdict System [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- `Verdict` enum with three variants: `Robust`, `Borderline`, `LikelyOverfit`
- Classification based on OOS/IS degradation ratio thresholds:
  - `Robust`: > 0.80 OOS/IS ratio with positive OOS returns and good efficiency
  - `Borderline`: 0.60-0.80 OOS/IS ratio or moderate efficiency
  - `LikelyOverfit`: < 0.60 OOS/IS ratio or negative OOS returns
- `verdict()` method added to `WalkForwardResult` using `from_criteria()` with degradation ratio, OOS positivity, and efficiency
- `verdict()` method added to `MonteCarloResult` using robustness score and probability thresholds
- CLI updated to display verdict with color-coded output (green=robust, yellow=borderline, red=likely_overfit)
- Helper methods: `is_acceptable()`, `description()`, `label()`, `Display` trait impl
- `from_degradation_ratio()` for simple classification, `from_criteria()` for nuanced multi-factor classification
- Exported from `lib.rs`
- 12 unit tests added for Verdict classification

**Dependencies:** None
**Effort:** Small (1 day)

---

## P3 - Lower Priority (Nice to Have)

### 15. Polars Backend Support [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- pandas DataFrame input support added to `backtest()` and `validate()` functions
- polars DataFrame input support added to `backtest()` and `validate()` functions
- Auto-detection of DataFrame type via `__class__.__module__`
- Case-insensitive OHLCV column detection (open/Open/o, high/High/h, etc.)
- Timestamp extraction from DatetimeIndex, date columns, or polars datetime columns
- Fallback to sequential timestamps if no date info available
- Type stubs updated to include DataFrame types

**Files modified:**
- `src/python/backtest.rs` - Added `extract_bars_from_pandas()`, `extract_bars_from_polars()`, `extract_pandas_timestamps()`, `extract_polars_timestamps()` functions
- `python/mantis/__init__.pyi` - Updated type hints to include DataFrame types

**Dependencies:** None
**Effort:** Small (completed)

---

### 16. Sample Data Bundling [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- `load_sample(name)` function loads bundled sample data by name (case-insensitive)
- `list_samples()` function returns available sample names
- Sample data files in `data/samples/`: AAPL.csv, SPY.csv, BTC.csv
- ~10 years of daily OHLCV data (2014-2024)
- AAPL/SPY: ~2609 bars (trading days only)
- BTC: ~3653 bars (includes weekends)
- Data embedded in binary via `include_str!()` macro
- Python bindings: `mt.load_sample("AAPL")`, `mt.list_samples()`
- Type stubs included for IDE autocomplete
- 6 unit tests added (all passing)

**Files created/modified:**
- `data/samples/AAPL.csv` - Apple stock sample data
- `data/samples/SPY.csv` - S&P 500 ETF sample data
- `data/samples/BTC.csv` - Bitcoin sample data
- `src/data.rs` - Added `load_sample()`, `list_samples()`, `load_csv_from_string()`
- `src/lib.rs` - Added re-exports
- `src/python/data.rs` - Added Python bindings
- `src/python/mod.rs` - Registered functions
- `python/mantis/__init__.py` - Added exports
- `python/mantis/__init__.pyi` - Added type stubs

**Dependencies:** None
**Effort:** Small (1 day)

---

### 17. Documentation Site [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- mkdocs-material theme with dark/light mode toggle
- Search with fuzzy matching enabled
- Code copy buttons on all examples
- Mobile-friendly responsive design

**Documentation structure:**
- `docs/index.md` - Homepage with hero section and quick links
- `docs/quickstart.md` - 5-minute getting started guide
- `docs/cookbook/` - Copy-paste recipes:
  - `index.md` - Cookbook overview
  - `loading-data.md` - Data loading examples
  - `backtests.md` - Running backtests
  - `validation.md` - Walk-forward validation
  - `multi-symbol.md` - Multi-asset backtesting
  - `position-sizing.md` - Position sizing methods
  - `visualization.md` - Charts and reports
- `docs/concepts/` - How it works:
  - `how-it-works.md` - Architecture overview
  - `execution-model.md` - Order execution details
  - `cost-model.md` - Trading costs explained
  - `validation.md` - Validation philosophy
- `docs/api/` - API reference:
  - `index.md` - API overview
  - `data.md` - Data loading functions
  - `backtest.md` - Backtest functions
  - `results.md` - Results classes
  - `validation.md` - Validation classes
- `docs/playground.md` - Google Colab/Binder links
- `docs/stylesheets/extra.css` - Custom styling

**Files created:**
- `mkdocs.yml` - Site configuration with material theme
- `docs/` directory with all documentation pages

**Dependencies:** None (mkdocs-material is external for building)
**Effort:** Medium (completed)

---

### 18. Python API Spec Gaps [COMPLETE]
**Status:** All sub-items complete (2026-01-16)

**Missing per python-api.md specification:**

#### 18a. `results.validate()` Method [COMPLETE]
- Added validate() method to PyBacktestResult in src/python/results.rs
- Stores bars, signal, and config when creating result from backtest()
- Performs walk-forward validation using stored data
- Parameters: folds (default 12), train_ratio (default 0.75), anchored (default true)
- Returns ValidationResult with IS/OOS metrics and verdict
- Type stubs updated in python/mantis/__init__.pyi
- **Priority:** P1 (core UX feature)
- **Effort:** Small (completed)

#### 18b. `mt.load_results()` Function [COMPLETE]
- Added load_results() function to src/python/data.rs
- Loads PerformanceSummary from JSON file
- Creates PyBacktestResult from loaded data
- Added from_summary() constructor to PyBacktestResult
- Proper error handling for missing/invalid files
- Type stubs updated in python/mantis/__init__.pyi
- **Priority:** P2
- **Effort:** Small (completed)

#### 18c. `mt.Backtest` Fluent API Class [COMPLETE]
- Added Backtest class to python/mantis/__init__.py
- Fluent builder pattern with method chaining
- Methods: commission(), slippage(), size(), cash(), stop_loss(), take_profit(), allow_short(), borrow_cost(), run()
- Type stubs updated in python/mantis/__init__.pyi
- Compatible with both signal arrays and built-in strategies
- **Priority:** P3 (nice-to-have, functional API works)
- **Effort:** Medium (completed)

#### 18d. Interactive Plotly Charts [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- Added `plotly>=5.0.0` as optional dependency (`pip install mantis-bt[jupyter]`)
- `results.plot()` now auto-detects Jupyter environment via IPython shell detection
- In Jupyter with plotly: Returns interactive Plotly Figure with equity curve and drawdown subplots
- In terminal or without plotly: Falls back to ASCII sparkline (backward compatible)
- `validation.plot()` also supports Plotly in Jupyter with IS vs OOS bar charts
- Added `_repr_html_()` method for rich display in Jupyter notebooks
- New parameter `show_drawdown=True` to optionally hide drawdown subplot in Plotly
- Type stubs updated in `__init__.pyi`

**Files modified:**
- `pyproject.toml` - Added `jupyter = ["plotly>=5.0.0"]` optional dependency
- `python/mantis/__init__.py` - Added wrapper classes with Plotly support
- `python/mantis/__init__.pyi` - Updated type stubs

**Dependencies:** None (plotly is optional)
**Effort:** Medium (completed)

#### 18e. Python API max_position and fill_price Parameters [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- Added `max_position` parameter to Python backtest API (default 1.0 = 100% of equity)
- Added `fill_price` parameter to Python backtest API (default "next_open")
- fill_price options: "next_open", "close", "vwap", "twap", "midpoint", "random"
- Parameters work in both functional API and fluent Backtest class
- Type stubs updated in __init__.pyi

**Files modified:**
- src/python/backtest.rs - Added parameters to PyBacktestConfig and backtest()
- python/mantis/__init__.py - Added to backtest() function and Backtest fluent API class
- python/mantis/__init__.pyi - Type stubs updated

**Dependencies:** None
**Effort:** Small (completed)

#### 18f. Python Sensitivity Analysis Bindings [COMPLETE]
**Status:** COMPLETE

**Implementation details:**
- Created `src/python/sensitivity.rs` module (~1050 lines) with PyO3 bindings
- Parameter range functions: `linear_range()`, `log_range()`, `discrete_range()`, `centered_range()`
- `mt.sensitivity()` - Run parameter sensitivity analysis on built-in strategies
  - Supports all 6 built-in strategies: sma-crossover, momentum, mean-reversion, rsi, macd, breakout
  - Configurable metric: sharpe, sortino, return, calmar, profit_factor, win_rate, max_drawdown
  - Returns `SensitivityResult` with stability scores, cliffs, plateaus, heatmaps
- `mt.cost_sensitivity()` - Test strategy robustness to transaction costs
  - Accepts signal arrays or built-in strategies
  - Configurable cost multipliers (default: 0x, 1x, 2x, 5x, 10x)
  - Returns `CostSensitivityResult` with degradation metrics, robustness check
- Python wrapper classes with Plotly visualization support in Jupyter
- Type stubs in `python/mantis/__init__.pyi` for IDE autocomplete
- All functions exported from `python/mantis/__init__.py`

**Python API:**
```python
# Parameter sensitivity
result = mt.sensitivity(
    data,
    strategy="sma-crossover",
    params={
        "fast_period": mt.linear_range(5, 20, 4),
        "slow_period": mt.linear_range(20, 60, 5),
    },
    metric="sharpe"
)
print(result.stability_score())  # 0.72
print(result.best_params())  # {'fast_period': 10.0, 'slow_period': 40.0}
result.plot_heatmap("fast_period", "slow_period")  # Interactive Plotly in Jupyter

# Cost sensitivity
result = mt.cost_sensitivity(data, signal)
print(result.sharpe_degradation_at(5.0))  # 45.2%
print(result.is_robust())  # True
result.plot()  # Interactive Plotly in Jupyter
```

**Dependencies:** None
**Effort:** Medium (completed)

---

## Newly Discovered Gaps (2026-01-16 Audit)

### 19. Frequency Auto-Detection and Annualization [COMPLETE]
**Status:** COMPLETE (2026-01-16)
**Priority:** P1 (was HIGH IMPACT - now fixed)

**Implementation details:**
- Added `DataFrequency` enum to `src/types.rs` with 14 frequency variants (Second1 through Month)
- `DataFrequency::detect(bars)` - Auto-detects frequency by analyzing timestamp gaps (uses median for robustness)
- `DataFrequency::annualization_factor(trading_hours_24)` - Returns correct factor for both traditional markets (252 days) and 24/7 markets (365 days)
- `DataFrequency::is_likely_crypto(bars)` - Heuristic to detect 24/7 markets by checking for weekend bars
- Added `data_frequency: Option<DataFrequency>` to `BacktestConfig` for explicit override
- Added `trading_hours_24: Option<bool>` to `BacktestConfig` for market type override
- Engine now auto-detects frequency and uses proper annualization factor for Sharpe/Sortino calculations
- 13 new unit tests for DataFrequency (all passing)

**Annualization factors:**
- Traditional markets: Day=252, Hour1=1638, Minute1=98280, etc.
- 24/7 markets: Day=365, Hour1=8760, Minute1=525600, etc.

**Files modified:**
- `src/types.rs` - Added DataFrequency enum with all methods
- `src/engine.rs` - Updated BacktestConfig, updated result calculation to use auto-detected frequency
- `src/config.rs` - Updated BacktestConfig initialization
- `src/lib.rs` - Added DataFrequency to public exports

**Python API exposure:** Implemented in Item #25 below.

**Effort:** Medium (completed)
**Dependencies:** None

---

### 20. Python Benchmark Comparison [COMPLETE]
**Status:** COMPLETE
**Priority:** P2

**Implementation details:**
- Added `benchmark` parameter to `mt.backtest()` function
- Added benchmark properties to `BacktestResult`:
  - `alpha` - Jensen's alpha (risk-adjusted excess return)
  - `beta` - Portfolio beta (sensitivity to benchmark)
  - `benchmark_return` - Benchmark total return for period
  - `excess_return` - Strategy return minus benchmark return
  - `tracking_error` - Annualized tracking error
  - `information_ratio` - Alpha per unit of active risk
  - `benchmark_correlation` - Correlation with benchmark (-1 to 1)
  - `up_capture` - Up-capture ratio
  - `down_capture` - Down-capture ratio
  - `has_benchmark` - Boolean indicating if benchmark metrics available
- `metrics()` dict includes all benchmark metrics when available
- Fluent API `Backtest` class supports `.benchmark(data)` method
- Type stubs updated in `__init__.pyi` for IDE autocomplete

**Files modified:**
- `src/python/backtest.rs` - Added benchmark parameter, BenchmarkMetrics calculation
- `src/python/results.rs` - Added benchmark properties and metrics, updated constructors
- `python/mantis/__init__.py` - Added benchmark wrapper properties and method
- `python/mantis/__init__.pyi` - Updated type stubs

**Effort:** Small (completed)
**Dependencies:** None

---

### 21. Python Split/Dividend Adjustment [COMPLETE]
**Status:** COMPLETE
**Priority:** P2

**Implementation details:**
- Added `mt.adjust()` function to Python API
- Parameters:
  - `data` - Data dictionary from load() containing OHLCV arrays
  - `splits` - Optional list of split dictionaries with keys: date, ratio, reverse (optional)
  - `dividends` - Optional list of dividend dictionaries with keys: date, amount, type (optional)
  - `method` - Dividend adjustment method: "proportional" (default), "absolute", "none"
- Returns new data dictionary with adjusted OHLCV arrays
- Uses existing Rust functions: adjust_for_splits(), adjust_for_dividends()
- Type stubs added for IDE autocomplete

**Files modified:**
- `src/python/data.rs` - Added adjust() function with parse_splits(), parse_dividends(), parse_date_from_dict() helpers
- `src/python/mod.rs` - Registered adjust function
- `python/mantis/__init__.py` - Added adjust to imports
- `python/mantis/__init__.pyi` - Added type stubs with docstrings

**Example:**
```python
>>> data = mt.load("AAPL.csv")
>>> # Adjust for a 4:1 split on 2020-08-31
>>> adjusted = mt.adjust(data, splits=[{"date": "2020-08-31", "ratio": 4.0}])
>>> # Adjust for dividends with proportional method
>>> adjusted = mt.adjust(data, dividends=[{"date": "2024-02-09", "amount": 0.24}])
```

**Effort:** Small (completed)
**Dependencies:** None

---

### 22. ATR-Based Stop-Loss/Take-Profit in Python [COMPLETE]
**Status:** COMPLETE
**Priority:** P2

**Implementation details:**
- `stop_loss` and `take_profit` parameters now accept both floats and strings
- ATR-based syntax: `"2atr"` for 2x ATR stop/target
- Trailing stop syntax: `"5trail"` for 5% trailing stop
- Risk-reward syntax: `"2rr"` for 2:1 risk-reward ratio take profit
- ATR is automatically calculated (14-period) from input data
- Works with both `mt.backtest()` function and fluent `Backtest` API

**Parameter formats:**
- Float: percentage (e.g., `0.05` for 5%)
- String ATR: `"2atr"`, `"1.5atr"` (ATR multiplier)
- String trailing: `"5trail"` (trailing stop percentage)
- String risk-reward: `"2rr"`, `"3:1"` (risk-reward ratio)

**Files modified:**
- `src/python/backtest.rs` - Added `StopSpec`, `TakeProfitSpec` enums, parsing logic, ATR integration
- `python/mantis/__init__.py` - Updated type hints and docstrings
- `python/mantis/__init__.pyi` - Updated type stubs

**Example:**
```python
>>> # ATR-based stop loss
>>> results = mt.backtest(data, signal, stop_loss="2atr")
>>> # Percentage stop with ATR take profit
>>> results = mt.backtest(data, signal, stop_loss=0.05, take_profit="3atr")
>>> # Risk-reward ratio
>>> results = mt.backtest(data, signal, stop_loss="2atr", take_profit="2rr")
>>> # Fluent API
>>> mt.Backtest(data, signal).stop_loss("1.5atr").take_profit("3atr").run()
```

**Effort:** Small (completed)
**Dependencies:** None

---

### 23. Parallel Parameter Sweep [COMPLETE]
**Status:** COMPLETE
**Priority:** P3

**Implementation details:**
- Created `src/python/sweep.rs` module (~400 lines) with rayon-based parallel execution
- `PySweepResult` class with `best()`, `sorted_by()`, `top()`, `summary()`, `plot()` methods
- `PySweepResultItem` class for individual param/result pairs
- Signals are pre-computed in Python (since signal_fn can't be called from Rust)
- Rust parallel loop runs backtests using rayon's `par_iter()` with GIL released
- Python `SweepResult` wrapper class with Plotly visualization support in Jupyter
- Updated type stubs in `__init__.pyi`
- Function signature updated: `parallel=True` parameter (`n_jobs` kept for backwards compat but deprecated)

**Files created/modified:**
- `src/python/sweep.rs` (new)
- `src/python/mod.rs` (added sweep module and functions)
- `python/mantis/__init__.py` (updated sweep function, added SweepResult class)
- `python/mantis/__init__.pyi` (added type stubs)

**Effort:** Medium (completed)
**Dependencies:** None

---

### 24. Advanced Plot Features [COMPLETE]
**Status:** COMPLETE (2026-01-16)
**Priority:** P2

**Implementation details:**
- Enhanced `BacktestResult.plot()` method with new parameters:
  - `trades=True` - Show trade entry/exit markers (triangles) on the equity curve
  - `benchmark=True` - Show benchmark comparison overlay if benchmark data available
  - `save="file.html/png/pdf"` - Save plot to file (HTML, PNG, PDF, SVG supported)
  - `title="Custom Title"` - Custom plot title
  - `height=600` - Custom height in pixels
  - `theme="dark"` or `"light"` - Color theme support
- Trade markers show:
  - Green triangle-up for BUY/COVER entries
  - Red triangle-down for SELL/SHORT entries
  - Hover text with quantity, price, and P&L
- Dark theme colors: teal equity curve, coral drawdown, dark background
- Light theme colors: blue equity curve, red drawdown, white background
- Image export requires kaleido (`pip install kaleido`)
- Fallback for ASCII-only mode: save as .txt file

**Files modified:**
- `python/mantis/__init__.py` - Added new parameters and helper methods
- `python/mantis/__init__.pyi` - Updated type stubs

**Effort:** Small (completed)
**Dependencies:** None

---

### 25. Python Frequency Override Parameters [COMPLETE]
**Status:** COMPLETE (2026-01-16)
**Priority:** P1

**Implementation details:**
- Added `freq` parameter to Python `backtest()` function
  - Accepts strings: "1s", "5s", "1min", "5min", "15min", "30min", "1h", "4h", "1d", "1w", "1mo"
  - Auto-detected from bar timestamps if None (default)
- Added `trading_hours_24` parameter to Python `backtest()` function
  - When True, uses 24/7 market annualization (365 days/year)
  - When False, uses traditional market annualization (252 days/year)
  - Auto-detected from weekend bars if None (default)
- Added `parse_freq()` helper function in backtest.rs
- Updated `PyBacktestConfig` class with new fields
- Added `freq()` and `trading_hours_24()` methods to fluent `Backtest` API class
- Updated type stubs in `__init__.pyi` for IDE autocomplete

**Files modified:**
- `src/python/backtest.rs` - Added parameters, parse_freq(), updated PyBacktestConfig
- `python/mantis/__init__.py` - Added fluent API methods
- `python/mantis/__init__.pyi` - Updated type stubs and docstrings

**Effort:** Small (completed)
**Dependencies:** None

---

## Remaining Spec Gaps (Discovered 2026-01-16)

These features are implemented in the Rust core but NOT exposed in the Python API:

### 26. Rolling Metrics Python API [COMPLETE]
**Status:** COMPLETE
**Priority:** P2

**Implementation details:**
- Added `rolling_sharpe(window=252, annualization_factor=252.0)` method to PyBacktestResult
- Added `rolling_drawdown(window=None)` method to PyBacktestResult
- Added `rolling_max_drawdown(window=252)` method to PyBacktestResult
- Added `rolling_volatility(window=21, annualization_factor=252.0)` method to PyBacktestResult
- Added `calculate_returns()` helper method
- Updated Python type stubs in `python/mantis/__init__.pyi`
- Updated Python wrapper in `python/mantis/__init__.py`

**Effort:** Small
**Dependencies:** None

---

### 27. Limit Order Python API [COMPLETE]
**Status:** COMPLETE
**Priority:** P3

**Implementation details:**
- Added `order_type` parameter to Python `backtest()` function - accepts "market" (default) or "limit"
- Added `limit_offset` parameter - offset as fraction of close price (e.g., 0.01 = 1%)
- For buys: limit_price = close * (1 - limit_offset) (below close)
- For sells: limit_price = close * (1 + limit_offset) (above close)
- Added `use_limit_orders` and `limit_offset` fields to Rust BacktestConfig
- Updated `signal_to_order()` in engine.rs to create limit orders via `create_order()` helper
- Added `order_type()` and `limit_offset()` methods to fluent `Backtest` API class
- Updated PyBacktestConfig with order_type and limit_offset fields
- Updated Python type stubs (__init__.pyi) with new parameters and methods
- All 571 tests pass, clippy passes

**Effort:** Medium
**Dependencies:** None

---

### 28. Volume Participation Python API [COMPLETE]
**Status:** COMPLETE
**Priority:** P3

**Implementation details:**
- Added `max_volume_participation` parameter to `PyBacktestConfig` (default None = no limit)
- Added `max_volume_participation` parameter to `backtest()` function
- Integrated with `CostModel.max_volume_participation`
- Added `max_volume_participation()` method to fluent `Backtest` API class
- Updated Python type stubs and wrapper

**Effort:** Small
**Dependencies:** None

---

### 29. Python API Additional Metrics and Plot Methods [COMPLETE]
**Status:** COMPLETE
**Priority:** P2

**Implementation details:**
- Added `volatility` property to PyBacktestResult - annualized volatility as decimal
- Added `max_drawdown_duration` property - duration in days
- Added `avg_trade_duration` property - average holding period in days
- Added `plot_drawdown()` method - dedicated drawdown visualization with Plotly
- Added `plot_returns()` method - monthly returns heatmap or daily returns histogram
- Added `plot_trades()` method - equity curve with trade markers
- Updated `metrics()` dict to include all new metrics
- Updated Python type stubs (__init__.pyi) with all new properties and methods

**Files modified:**
- `src/python/results.rs`
- `python/mantis/__init__.py`
- `python/mantis/__init__.pyi`

**Effort:** Small
**Dependencies:** None

---

## Summary Table

| ID | Item | Status | Priority | Effort | Dependencies |
|----|------|--------|----------|--------|--------------|
| 1 | Statistical Tests | **COMPLETE** | P0 | - | - |
| 2 | Python Bindings (PyO3) | **COMPLETE** | P0 | Large | None |
| 3 | ONNX Module | **COMPLETE** | P0 | Small | None |
| 3a | ONNX Python Bindings | **COMPLETE** | P1 | Medium | None |
| 4 | Helpful Error Messages | **COMPLETE** | P0 | Medium | None |
| 5 | Rolling Metrics | **COMPLETE** | P1 | Medium | None |
| 6 | Short Borrow Costs | **COMPLETE** | P1 | Medium | None |
| 7 | load_multi/load_dir | **COMPLETE** | P1 | Small | None |
| 8 | Cost Sensitivity CLI | **COMPLETE** | P1 | Small | None |
| 9 | Position Sizing Integration | **COMPLETE** | P1 | Medium | None |
| 10 | Multi-Symbol CLI Command | **COMPLETE** | P1 | Medium | None |
| 11 | Parameter Sensitivity | **COMPLETE** | P2 | Medium | None |
| 12 | Visualization Module | **COMPLETE** | P2 | Medium | None |
| 13 | HTML Reports | **COMPLETE** | P2 | Small | None |
| 14 | Verdict System | **COMPLETE** | P2 | Small | None |
| 15 | Polars Backend | **COMPLETE** | P3 | Small | None |
| 16 | Sample Data Bundling | **COMPLETE** | P3 | Small | None |
| 17 | Documentation Site | **COMPLETE** | P3 | Medium | None |
| 18a | results.validate() method | **COMPLETE** | P1 | Small | None |
| 18b | mt.load_results() | **COMPLETE** | P2 | Small | None |
| 18c | mt.Backtest fluent API | **COMPLETE** | P3 | Medium | None |
| 18d | Interactive Plotly charts | **COMPLETE** | P2 | Medium | None |
| 18e | max_position/fill_price params | **COMPLETE** | P3 | Small | None |
| 18f | Sensitivity analysis bindings | **COMPLETE** | P3 | Medium | None |
| 19 | Frequency Auto-Detection | **COMPLETE** | P1 | Medium | None |
| 20 | Python Benchmark Comparison | **COMPLETE** | P2 | Small | None |
| 21 | Python Split/Dividend Adjustment | **COMPLETE** | P2 | Small | None |
| 22 | ATR-Based Stop-Loss in Python | **COMPLETE** | P2 | Small | None |
| 23 | Parallel Parameter Sweep | **COMPLETE** | P3 | Medium | None |
| 24 | Advanced Plot Features | **COMPLETE** | P2 | Small | None |
| 25 | Python Frequency Override Params | **COMPLETE** | P1 | Small | None |
| 26 | Rolling Metrics Python API | **COMPLETE** | P2 | Small | None |
| 27 | Limit Order Python API | **COMPLETE** | P3 | Medium | None |
| 28 | Volume Participation Python API | **COMPLETE** | P3 | Small | None |
| 29 | Python API Additional Metrics and Plot Methods | **COMPLETE** | P2 | Small | None |

---

## Items Verified as COMPLETE

| Item | Verification |
|------|-------------|
| **Order Buffering/Lookahead** | WORKING: engine.rs:598-623, PendingOrder struct, buffer_order_for_next_bar() |
| **Monthly Returns from Equity** | IMPLEMENTED: analytics.rs:941-967, groups by (year, month), calculates actual returns |
| **Auto-Warnings (check_suspicious_metrics)** | IMPLEMENTED: analytics.rs:979-1038, 5 warning types (Sharpe>3, WinRate>80%, DD<5%, Trades<30, PF>5) |
| **OOS Degradation Check** | IMPLEMENTED: analytics.rs:1043-1075, check_oos_degradation() standalone function |
| **Walk-Forward Default Parameters** | CORRECT: walkforward.rs:31-40 (12 folds, 75/25 ratio, anchored=true) |
| **Walk-Forward OOS/IS Degradation** | IMPLEMENTED: walkforward.rs:173-182, 354-359, 414-419 |
| **Parameter Stability** | IMPLEMENTED: walkforward.rs:361-364, 424-430 |
| **DSR (Deflated Sharpe Ratio)** | IMPLEMENTED: analytics.rs:863-885 |
| **PSR (Probabilistic Sharpe Ratio)** | IMPLEMENTED: analytics.rs:904-939 |
| **Column Auto-Detection** | IMPLEMENTED: data.rs:14-36 (CSV), data.rs:244-276 (Parquet) |
| **Square-Root Market Impact** | IMPLEMENTED: portfolio.rs:122-154, MarketImpactModel::SquareRoot |
| **Volume Participation Limits** | IMPLEMENTED: portfolio.rs:177-180, 304-316, applied at 969-972 |
| **Limit Order Fill Logic** | IMPLEMENTED: portfolio.rs:1581-1639 |
| **Black-Litterman** | FULLY IMPLEMENTED: multi_asset.rs:2124-2635 |
| **Mean-Variance Optimization** | FULLY IMPLEMENTED: multi_asset.rs:1515-2082 |
| **HRP (Hierarchical Risk Parity)** | FULLY IMPLEMENTED: multi_asset.rs:2683-3141 |
| **Cost Sensitivity Module & CLI** | **COMPLETE**: cost_sensitivity.rs (725 lines), CLI command `mantis cost-sensitivity` with custom multipliers, zero-cost flag, robustness threshold, text/JSON/CSV output |
| **ONNX Module** | **COMPLETE**: Re-enabled with ort 2.0.0-rc.11; optional feature (`--features onnx`); full batch inference support (524 lines in src/onnx.rs); CUDA disabled pending configuration; all unit tests passing |
| Config Defaults | CORRECT: slippage=0.1%, commission=0.1%, position_size=10% |
| Performance Benchmarks | EXISTS in benches/backtest_bench.rs (9 benchmark groups) |
| CPCV | IMPLEMENTED in cpcv.rs with purging/embargo (663 lines) |
| Monte Carlo | IMPLEMENTED in monte_carlo.rs with CI, bootstrap (726 lines) |
| Multi-Asset Strategies | 8 strategies: EqualWeight, Momentum, InverseVol, RiskParity, MVO, BL, HRP, Drift |
| Options Pricing | IMPLEMENTED: Black-Scholes, Binomial, Greeks, put-call parity |
| Streaming Indicators | IMPLEMENTED in streaming.rs (SMA, EMA, RSI, MACD, BB, ATR, StdDev) |
| Statistical Tests | **FIXED** in commit 0b67bff: ADF, autocorrelation, Ljung-Box all passing |
| **Helpful Error Messages** | **COMPLETE**: New error types (SignalShapeMismatch, InvalidSignal, LookaheadBias, SuspiciousSignal), ErrorHelp struct, validation.rs module with validate_signal(), validate_signal_quick(), validate_signals(), SignalValidationConfig, SignalStats - 11 tests |
| **load_multi/load_dir** | **COMPLETE**: load_multi() and load_dir() functions, DataManager methods, glob pattern support, 5 unit tests |
| **Rolling Metrics** | **COMPLETE**: rolling_sharpe(), rolling_drawdown(), rolling_drawdown_windowed(), rolling_max_drawdown(), rolling_volatility() - all exported from lib.rs, 8 unit tests |
| **Short Borrow Costs** | **COMPLETE**: borrow_cost_rate field in CostModel (3% default), accrue_borrow_costs() method, charges daily for short equity positions, CostSettings integration, CLI --borrow-cost flag, scale_cost_model() support, 4 unit tests |
| **Position Sizing Integration** | **COMPLETE**: PositionSizingMethod enum (PercentOfEquity, FixedDollar, VolatilityTargeted, SignalScaled, RiskBased), new PositionSizer methods (size_fixed_dollar, size_by_signal, size_by_volatility_target, size_percent_of_equity), BacktestConfig integration, signal_to_order() updated with ATR/volatility calculations, CLI options (--sizing-method, --fixed-dollar, --target-vol, --vol-lookback, --risk-per-trade, --stop-atr, --atr-period), 7 unit tests |
| **Verdict System** | **COMPLETE**: Verdict enum (Robust, Borderline, LikelyOverfit), verdict() methods in WalkForwardResult and MonteCarloResult, from_degradation_ratio() and from_criteria() classification, CLI color-coded verdict display, is_acceptable(), description(), label() helpers, exported from lib.rs, 12 unit tests |
| **Parameter Sensitivity** | **COMPLETE**: src/sensitivity.rs (~1100 lines), ParameterRange enum (Linear, Logarithmic, Discrete, Centered), SensitivityConfig with constraints, SensitivityAnalysis with parallel rayon execution, HeatmapData for 2D visualization exports, cliff detection, plateau detection, parameter importance ranking, stability scores, CLI `mantis sensitivity` command with metric/steps/heatmap options, supports all 6 built-in strategies, 10 unit tests |
| **HTML Reports** | **COMPLETE**: export_report_html() method in Exporter and MultiAssetExporter, self-contained HTML with embedded CSS, dark/light theme support, SVG charts for equity curve and drawdown, performance metrics grid, trade statistics, trade list table, 6 unit tests |
| **Python Bindings (PyO3)** | **COMPLETE**: src/python/ module with mod.rs, types.rs, data.rs, backtest.rs, results.rs; pyproject.toml with maturin config; python/mantis/ wrapper with __init__.py, __init__.pyi type stubs, py.typed marker; load(), load_multi(), load_dir(), backtest(), signal_check(), validate() functions; 6 built-in strategies; compare(), sweep() helpers; BacktestResult with metrics, equity_curve, plot(), save(), report(); ValidationResult with is_robust(), fold_details(), plot(); FoldDetail class with is_sharpe, oos_sharpe, is_return, oos_return, efficiency, is_bars, oos_bars |
| **Visualization Module** | **COMPLETE**: src/viz.rs (~845 lines), ASCII sparklines (sparkline(), sparkline_with_config(), equity_sparkline()), strategy comparison (compare_strategies(), StrategyComparison), walk-forward visualization (walkforward_fold_chart(), walkforward_summary()), SVG heatmaps (heatmap_to_svg(), export_heatmap_svg()), ASCII heatmaps (heatmap_to_ascii()), result summaries (result_summary(), result_with_verdict()), all exported from lib.rs, 13 unit tests |
| **Sample Data Bundling** | **COMPLETE**: load_sample(), list_samples() functions; data/samples/AAPL.csv, SPY.csv, BTC.csv (~10 years daily OHLCV 2014-2024); embedded via include_str!(); Python bindings mt.load_sample(), mt.list_samples(); type stubs; 6 unit tests |
| **Polars Backend Support** | **COMPLETE**: pandas DataFrame input support in backtest() and validate(); polars DataFrame input support in backtest() and validate(); auto-detection of DataFrame type via __class__.__module__; case-insensitive OHLCV column detection (open/Open/o, high/High/h, etc.); timestamp extraction from DatetimeIndex, date columns, or polars datetime columns; fallback to sequential timestamps; type stubs updated with DataFrame types; Files: src/python/backtest.rs (extract_bars_from_pandas, extract_bars_from_polars, extract_pandas_timestamps, extract_polars_timestamps), python/mantis/__init__.pyi |
| **Multi-Symbol CLI Command** | **COMPLETE**: `mantis portfolio` CLI command with 9 portfolio strategies (equal-weight, momentum, inverse-vol, risk-parity, min-variance, max-sharpe, hrp, drift-equal, drift-momentum), glob pattern loading, portfolio constraints (max-position, max-leverage, max-turnover, min/max-holdings), text/JSON/CSV output |
| **Python Bindings Advanced Metrics** | **COMPLETE**: ValidationResult.report() method for HTML export; BacktestResult.deflated_sharpe and .psr properties; metrics() dict includes deflated_sharpe and psr |
| **Interactive Plotly Charts** | **COMPLETE**: plotly>=5.0.0 as optional dependency (`pip install mantis-bt[jupyter]`); results.plot() auto-detects Jupyter via IPython; returns interactive Plotly Figure with equity curve and drawdown subplots in Jupyter, ASCII sparkline fallback otherwise; validation.plot() also supports Plotly with IS vs OOS bar charts; _repr_html_() for rich Jupyter display; show_drawdown parameter; Files: pyproject.toml, python/mantis/__init__.py, python/mantis/__init__.pyi |
| **Documentation Site** | **COMPLETE**: mkdocs.yml with material theme, dark/light mode, search; docs/ with quickstart.md, cookbook/ (6 recipe pages), concepts/ (4 pages), api/ (5 reference pages), playground.md; Custom CSS for styling; Full coverage of Quick Start, Cookbook, API Reference, Concepts |
| **Python Sensitivity Analysis Bindings** | **COMPLETE**: src/python/sensitivity.rs (~1050 lines); mt.sensitivity() for parameter sensitivity on built-in strategies with stability scores, cliffs, plateaus, heatmaps; mt.cost_sensitivity() for cost robustness testing; parameter range functions (linear_range, log_range, discrete_range, centered_range); Plotly visualization in Jupyter; type stubs for IDE autocomplete |
| **Frequency Auto-Detection** | **COMPLETE**: DataFrequency enum with 14 variants (Second1-Month); detect() analyzes timestamp gaps; annualization_factor(trading_hours_24) for traditional (252 days) or 24/7 (365 days) markets; is_likely_crypto() heuristic; BacktestConfig.data_frequency and .trading_hours_24 overrides; Engine auto-detects frequency for Sharpe/Sortino; 13 unit tests |
| **Python Split/Dividend Adjustment** | **COMPLETE**: mt.adjust() function with splits (date, ratio, reverse) and dividends (date, amount, type) parameters; method options (proportional, absolute, none); parse_splits(), parse_dividends(), parse_date_from_dict() helpers; type stubs; Files: src/python/data.rs, src/python/mod.rs, python/mantis/__init__.py, python/mantis/__init__.pyi |
| **ATR-Based Stop-Loss/Take-Profit** | **COMPLETE**: stop_loss and take_profit accept float or string; ATR syntax ("2atr"), trailing ("5trail"), risk-reward ("2rr"); StopSpec, TakeProfitSpec enums; parse_stop_loss(), parse_take_profit() helpers; 14-period ATR auto-calculated; works with backtest() and Backtest fluent API; type stubs updated; Files: src/python/backtest.rs, python/mantis/__init__.py, python/mantis/__init__.pyi |
| **Parallel Parameter Sweep** | **COMPLETE**: src/python/sweep.rs (~400 lines) with rayon-based parallel execution; PySweepResult class with best(), sorted_by(), top(), summary(), plot() methods; PySweepResultItem for individual param/result pairs; signals pre-computed in Python (signal_fn not callable from Rust); Rust parallel loop using rayon par_iter() with GIL released; Python SweepResult wrapper with Plotly visualization in Jupyter; parallel=True parameter (n_jobs deprecated); Files: src/python/sweep.rs (new), src/python/mod.rs, python/mantis/__init__.py, python/mantis/__init__.pyi |
| **Python Frequency Override** | **COMPLETE**: freq parameter accepts "1s", "5s", "1min", "5min", "15min", "30min", "1h", "4h", "1d", "1w", "1mo"; trading_hours_24 bool param; parse_freq() helper; PyBacktestConfig updated; Backtest fluent API .freq() and .trading_hours_24() methods; type stubs updated; Files: src/python/backtest.rs, python/mantis/__init__.py, python/mantis/__init__.pyi |
| Codebase Cleanliness | **VERIFIED**: No TODOs/FIXMEs in codebase |
| ALL TESTS | **PASSING**: 571 lib tests (0 failures) |
| Python Bindings Build | **VERIFIED**: `cargo check --features python` compiles successfully (API drift fix 2026-01-16) |
| **Limit Order Python API** | **COMPLETE**: `order_type` param ("market"/"limit"), `limit_offset` as fraction of close; Rust BacktestConfig fields; `signal_to_order()` limit order creation; Backtest fluent API methods; PyBacktestConfig; type stubs updated; 571 tests pass |
| **Python API Additional Metrics and Plot Methods** | **COMPLETE**: `volatility` property (annualized volatility as decimal); `max_drawdown_duration` property (duration in days); `avg_trade_duration` property (average holding period in days); `plot_drawdown()` method (dedicated drawdown visualization with Plotly); `plot_returns()` method (monthly returns heatmap or daily returns histogram); `plot_trades()` method (equity curve with trade markers); `metrics()` dict includes all new metrics; type stubs updated; Files: src/python/results.rs, python/mantis/__init__.py, python/mantis/__init__.pyi |
| **ONNX Python Bindings** | **COMPLETE**: OnnxModel, ModelConfig, InferenceStats classes; load_model(), generate_signals() functions; backtest() accepts model="path.onnx" and features=df parameters; signal_threshold for converting predictions to signals; auto-detect feature dimensions; helpful error messages; Files: src/python/onnx.rs (new), src/python/mod.rs, src/python/backtest.rs, python/mantis/__init__.py, python/mantis/__init__.pyi |

---

## Recommended Execution Order

**Phase 0 - Foundation (Week 1):**
1. Python Bindings (#2) - 2-3 weeks (parallel track)
2. ~~Helpful Error Messages (#4)~~ - **COMPLETE**

**Phase 1 - Core Features (Weeks 2-3):**
3. ~~Rolling Metrics (#5)~~ - **COMPLETE**
4. ~~Short Borrow Costs (#6)~~ - **COMPLETE**
5. ~~load_multi/load_dir (#7)~~ - **COMPLETE**
6. ~~Cost Sensitivity CLI (#8)~~ - **COMPLETE**

**Phase 2 - Integration (Week 4):**
7. ~~Position Sizing Integration (#9)~~ - **COMPLETE**
8. ~~Multi-Symbol CLI Command (#10)~~ - **COMPLETE**

**Phase 3 - Analysis (Week 5):**
9. Parameter Sensitivity (#11) - 3-4 days
10. Verdict System (#14) - 1 day

**Phase 4 - Polish (Weeks 6+):**
11. Visualization Module (#12) - 3-4 days
12. HTML Reports (#13) - 1-2 days
13. ~~ONNX re-enablement (when ort stabilizes)~~ - **COMPLETE** (including Python bindings)
14. Lower priority items as time permits

---

## Technical Notes

### Lookahead Bug Fix Details (FIXED)

The lookahead bias has been resolved using the **order buffering** approach:

**Implementation:**
- All orders are now buffered via `buffer_order_for_next_bar()` instead of executing immediately
- Orders generated from bar[i] data now fill at bar[i+1].open (not bar[i].open)
- Stop-loss/take-profit exits are also buffered for next bar execution
- Entry tracking (entry_prices, trailing_stops) is now handled in `handle_pending_orders()` when orders fill
- Added `pending_exits` HashSet to prevent double-buffering of stop exits
- `PendingOrder` struct now includes `signal: Option<Signal>` field for entry tracking

**Files modified:**
- `src/engine.rs` - Main execution loop, `handle_pending_orders()`, `buffer_order_for_next_bar()` (new function)

### Python Bindings API Drift Fix (2026-01-16)

The Python bindings had compilation errors due to API drift between the core library and PyO3 bindings. All issues have been resolved and verified.

**Issues Fixed:**

1. **Strategy trait signature change**: Updated from `on_bar(&mut self, bars: &[Bar])` to `on_bar(&mut self, ctx: &StrategyContext)`
2. **Strategy type renames**: Updated imports from `Breakout` → `BreakoutStrategy`, `Momentum` → `MomentumStrategy`
3. **MeanReversion constructor change**: Now takes 4 parameters (period, num_std, entry_std, exit_std) instead of 2
4. **WalkForwardResult struct additions**: Added 3 new fields: `combined_oos_return`, `oos_sharpe_threshold_met`, `walk_forward_efficiency`
5. **BacktestResult struct additions**: Added new fields: `config`, `config_hash`, `data_checksums`, `experiment_id`, `git_info`, `seed`
6. **PyBacktestConfig::new() expansion**: Added 2 new parameters: `max_position` and `fill_price`
7. **Signal conversion refactor**: Replaced `From` trait usage with conditional logic for f64 → Signal conversion

**Files Modified:**
- `src/python/sensitivity.rs` - Fixed strategy imports, on_bar signature, MeanReversion constructor, removed unused imports
- `src/python/results.rs` - Fixed on_bar signature, WalkForwardResult initialization, BacktestResult fields, added imports
- `src/python/backtest.rs` - Fixed PyBacktestConfig::new() call, removed unused import

**Verification:**
- All 558 tests pass
- Python bindings compile successfully with `cargo check --features python`
- Build verified successful

---

### ONNX Module Status

The ONNX module has been re-enabled with ort 2.0:
- Uses ort 2.0.0-rc.11 (production-ready despite RC label)
- Enabled as optional feature: `cargo build --features onnx`
- CUDA support is disabled pending additional configuration (requires CUDA toolkit installation and ort CUDA feature enablement)
- All unit tests passing
- Full batch inference support available

### Python Bindings Architecture

Two paths available:
1. **Full PyO3 with maturin (recommended)** - native speed, proper types, Jupyter integration
2. **Enhanced subprocess (current)** - simpler but limited, no in-process DataFrames

### What's Already Excellent (No Action Needed)

- Core backtest engine with comprehensive cost modeling
- 40+ technical indicators in analytics.rs
- Monte Carlo simulation with confidence intervals
- Walk-forward analysis with OOS degradation
- CPCV (Combinatorial Purged Cross-Validation) with purging/embargo
- Multi-asset portfolio optimization (Black-Litterman, Mean-Variance, HRP)
- Options pricing (Black-Scholes, Binomial, Greeks)
- Regime detection (trend, volatility, volume)
- Streaming indicators for real-time applications
- Feature extraction for ML (40+ features, sequence building)
- Experiment tracking with SQLite
- Comprehensive CLI with 11 commands
- Export formats: CSV, JSON, Parquet, NPY, Markdown
- Performance benchmarks (9 benchmark groups)
- Limit order fill logic
- Position sizing utilities in risk.rs (need integration)
- Deflated Sharpe Ratio and Probabilistic Sharpe Ratio
- Statistical tests (ADF, autocorrelation, Ljung-Box) - ALL PASSING

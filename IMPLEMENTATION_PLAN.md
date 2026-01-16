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
- **Total Tests:** 552+ passed
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
- `python/mantis/__init__.pyi` - Type stubs for IDE autocomplete
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

**Dependencies:** None
**Effort:** Large (completed)

---

### 3. ONNX Module Re-enablement [PARTIAL - VERIFIED]
**Status:** Complete infrastructure exists (524 lines), but disabled

**Current state:**
- `src/lib.rs:113-115` - Module commented out with TODO
- `Cargo.toml` - ort and ndarray dependencies commented out
- `src/onnx.rs` - Full implementation with CUDA support, batch inference

**Blocking issue:** `ort` crate instability (v2.0 API in flux, v1.x yanked)

**Action:** Monitor `ort` crate releases. Re-enable when v2.0 stabilizes.

**Dependencies:** External (ort crate stability)
**Effort:** Small once unblocked

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

### 10. Multi-Symbol Engine Documentation [VERIFIED - DOCUMENTATION GAP]
**Status:** MultiAssetEngine exists (5583 lines) but needs documentation

**Current state:**
- `src/engine.rs:213` - Engine.run() processes single symbol only
- `src/multi_asset.rs` - Complete MultiAssetEngine with:
  - Black-Litterman (lines 2181-2582)
  - Mean-Variance optimization (lines 1519-1958)
  - HRP, Risk Parity, Momentum, Drift strategies
  - Portfolio constraints (leverage, sector limits, turnover)
- NOT integrated: MultiAssetEngine uses Portfolio directly, not Engine
- Separate from main Engine - documentation gap only

**Resolution options:**
1. Document that MultiAssetEngine is the intended multi-symbol solution
2. OR add multi-symbol execution path to Engine

**Dependencies:** None
**Effort:** Small (documentation) or Medium (integration)

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

### 15. Polars Backend Support [PARTIAL]
**Status:** Arrow compatibility provides foundation

**Dependencies:** Python bindings (P0 item 2)
**Effort:** Small (1-2 days)

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

### 17. Documentation Site [MISSING]
**Status:** Not implemented

**Spec requirements:**
- mkdocs/sphinx with search
- Quick Start (5-minute guide)
- Cookbook/recipes
- Auto-generated API reference

**Files affected:**
- `docs/` (new) - mkdocs content
- `mkdocs.yml` (new) - Site configuration

**Dependencies:** None
**Effort:** Medium (ongoing)

---

## Summary Table

| ID | Item | Status | Priority | Effort | Dependencies |
|----|------|--------|----------|--------|--------------|
| 1 | Statistical Tests | **COMPLETE** | P0 | - | - |
| 2 | Python Bindings (PyO3) | **COMPLETE** | P0 | Large | None |
| 3 | ONNX Module | PARTIAL | P0 | Small | ort crate |
| 4 | Helpful Error Messages | **COMPLETE** | P0 | Medium | None |
| 5 | Rolling Metrics | **COMPLETE** | P1 | Medium | None |
| 6 | Short Borrow Costs | **COMPLETE** | P1 | Medium | None |
| 7 | load_multi/load_dir | **COMPLETE** | P1 | Small | None |
| 8 | Cost Sensitivity CLI | **COMPLETE** | P1 | Small | None |
| 9 | Position Sizing Integration | **COMPLETE** | P1 | Medium | None |
| 10 | Multi-Symbol Documentation | DOC GAP | P1 | Small | None |
| 11 | Parameter Sensitivity | **COMPLETE** | P2 | Medium | None |
| 12 | Visualization Module | **COMPLETE** | P2 | Medium | None |
| 13 | HTML Reports | **COMPLETE** | P2 | Small | None |
| 14 | Verdict System | **COMPLETE** | P2 | Small | None |
| 15 | Polars Backend | PARTIAL | P3 | Small | Item 2 |
| 16 | Sample Data Bundling | **COMPLETE** | P3 | Small | None |
| 17 | Documentation Site | MISSING | P3 | Medium | None |

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
| **ONNX Module Code** | COMPLETE but disabled: onnx.rs (524 lines, waiting for ort crate) |
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
| **Python Bindings (PyO3)** | **COMPLETE**: src/python/ module with mod.rs, types.rs, data.rs, backtest.rs, results.rs; pyproject.toml with maturin config; python/mantis/ wrapper with __init__.py, __init__.pyi type stubs, py.typed marker; load(), load_multi(), load_dir(), backtest(), signal_check() functions; 6 built-in strategies; compare(), sweep() helpers; BacktestResult with metrics and equity_curve |
| **Visualization Module** | **COMPLETE**: src/viz.rs (~845 lines), ASCII sparklines (sparkline(), sparkline_with_config(), equity_sparkline()), strategy comparison (compare_strategies(), StrategyComparison), walk-forward visualization (walkforward_fold_chart(), walkforward_summary()), SVG heatmaps (heatmap_to_svg(), export_heatmap_svg()), ASCII heatmaps (heatmap_to_ascii()), result summaries (result_summary(), result_with_verdict()), all exported from lib.rs, 13 unit tests |
| **Sample Data Bundling** | **COMPLETE**: load_sample(), list_samples() functions; data/samples/AAPL.csv, SPY.csv, BTC.csv (~10 years daily OHLCV 2014-2024); embedded via include_str!(); Python bindings mt.load_sample(), mt.list_samples(); type stubs; 6 unit tests |
| Codebase Cleanliness | **VERIFIED**: No TODOs/FIXMEs in codebase |
| ALL TESTS | **PASSING**: 558+ lib tests (0 failures) |

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
8. Multi-Symbol Documentation (#10) - 1 day

**Phase 3 - Analysis (Week 5):**
9. Parameter Sensitivity (#11) - 3-4 days
10. Verdict System (#14) - 1 day

**Phase 4 - Polish (Weeks 6+):**
11. Visualization Module (#12) - 3-4 days
12. HTML Reports (#13) - 1-2 days
13. ONNX re-enablement (when ort stabilizes)
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

### ONNX Blocking Issue

The `ort` crate situation requires monitoring:
- v1.x yanked from crates.io
- v2.0 API still in flux (RC stage)
- Check https://github.com/pykeio/ort/releases periodically
- Alternative consideration: `tract` crate as fallback

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

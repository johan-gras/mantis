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
- **Total Tests:** 532 passed
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

### 2. Python Bindings (PyO3) [MISSING - VERIFIED]
**Status:** Not implemented - CLI subprocess model used instead

**Current state:**
- No PyO3 dependency in Cargo.toml
- No src/python/ directory
- No maturin configuration
- No .pyi type stub files
- Only subprocess-based CLI integration (see examples/pytorch_integration.py)

**Spec requirements:**
- `pip install mantis` with native Python integration
- pandas/polars DataFrame support via Arrow
- Type stubs for IDE autocomplete
- Results display in Jupyter notebooks
- `results.plot()` works inline in Jupyter

**Files to create/modify:**
- `src/python/mod.rs` - PyO3 module wiring
- `src/python/data.rs` - Arrow-based DataFrame conversions
- `src/python/backtest.rs` - Python-facing backtest API
- `src/python/results.rs` - Results wrapper with `.to_pandas()`, `.to_polars()`
- `pyproject.toml` - Build configuration for maturin
- `python/mantis_bt/__init__.pyi` - Type stubs for autocomplete
- `Cargo.toml` - Add pyo3, arrow2 dependencies

**Dependencies:** None (foundational)
**Effort:** Large (2-3 weeks)

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

### 5. Rolling Metrics [MISSING - VERIFIED]
**Status:** Not implemented

**Spec requirements:**
- `rolling_sharpe(window=252)` - Rolling Sharpe ratio
- `rolling_drawdown()` - Rolling drawdown series

**Current state:** Only aggregate full-period metrics calculated.

**Files affected:**
- `src/analytics.rs` - Add rolling metric functions

**Dependencies:** None
**Effort:** Medium (2-3 days)

---

### 6. Short Borrow Costs [PARTIAL - VERIFIED]
**Status:** Only margin interest exists (no dedicated borrow fees)

**Current state:**
- Margin interest charged on negative cash balance (config.rs:127, 3% default)
- NO dedicated stock borrow fees
- NO locate fee fields
- NO hard-to-borrow list support

**Spec requirement:** Short positions should incur borrow costs (typically 5-50+ bps annually).

**Files affected:**
- `src/portfolio.rs` - Add `BorrowCost` struct to CostModel
- `src/config.rs` - Add borrow cost configuration

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

### 9. Position Sizing Integration [PARTIAL - VERIFIED]
**Status:** Utilities exist but not integrated into Engine

**Current state:**
- `src/engine.rs:451-522` - Only percent-of-equity sizing implemented
- `src/risk.rs:273-331` - PositionSizer utilities exist but NEVER CALLED:
  - `size_by_risk()` (line 273)
  - `size_by_volatility()` (line 284)
  - `size_by_kelly()` (line 302)
- These methods are NEVER CALLED by engine.rs
- engine.rs only uses percent-of-equity sizing
- NO FixedDollar sizing
- NO signal-scaled sizing

**Files affected:**
- `src/engine.rs` - Integrate PositionSizer into `signal_to_order()`
- Add `PositionSizingMethod` enum to BacktestConfig
- `src/risk.rs` - Add FixedDollar and SignalScaled methods

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

### 11. Parameter Sensitivity Analysis [PARTIAL]
**Status:** Walk-forward has parameter stability, no dedicated sweep module

**Current state:**
- `src/walkforward.rs` - `parameter_stability` field (line 426)
- CLI has Optimize command for grid search
- No dedicated sensitivity analysis/heatmap generation

**Files affected:**
- `src/sensitivity.rs` (new) - Parameter sweep analysis
- `src/cli.rs` - Add sensitivity command

**Dependencies:** None
**Effort:** Medium (3-4 days)

---

### 12. Visualization Module [MISSING - VERIFIED]
**Status:** Not implemented

**Spec requirements:**
- `results.plot()` - Interactive Plotly charts
- ASCII sparkline fallback in terminal
- `mt.compare()` - Multi-strategy comparison charts
- `validation.plot()` - Fold-by-fold performance

**Current state:**
- No viz.rs or visualization module
- Only terminal-based output using `colored` and `tabled` crates
- Export to Markdown reports only

**Files affected:**
- `src/viz.rs` (new) - Plotly JSON generation, ASCII sparklines
- `src/results.rs` - Add plot methods

**Dependencies:** Python bindings for full functionality
**Effort:** Medium (3-4 days)

---

### 13. HTML Report Generation [MISSING - VERIFIED]
**Status:** Not implemented

**Current state (src/export.rs):**
- Supports: CSV, JSON, Parquet, NPY, Markdown
- Documentation claims HTML (line 4) but NOT implemented
- NO `export_html()` method

**Files affected:**
- `src/export.rs` - Add `export_html()` function

**Dependencies:** Visualization module
**Effort:** Small (1-2 days)

---

### 14. Verdict System [PARTIAL - VERIFIED]
**Status:** Boolean methods exist, no explicit enum

**Current state:**
- `is_robust()` and `is_robust_with_sharpe()` methods exist in walkforward.rs (return booleans)
- NO explicit `Verdict` enum with "robust"/"borderline"/"likely_overfit" variants

**Spec requirement:**
```rust
enum Verdict { Robust, Borderline, LikelyOverfit }
```

**Files affected:**
- `src/walkforward.rs` - Add Verdict enum, enhance classification logic

**Dependencies:** None
**Effort:** Small (1 day)

---

## P3 - Lower Priority (Nice to Have)

### 15. Polars Backend Support [PARTIAL]
**Status:** Arrow compatibility provides foundation

**Dependencies:** Python bindings (P0 item 2)
**Effort:** Small (1-2 days)

---

### 16. Sample Data Bundling [MISSING]
**Status:** Not implemented

**Spec requirement:** `mt.load_sample("AAPL")` works offline with bundled data

**Files affected:**
- `data/samples/` (new) - Bundled sample files
- `src/data.rs` - Add `load_sample()` function
- `build.rs` - Include data in binary

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
| 2 | Python Bindings (PyO3) | MISSING | P0 | Large | None |
| 3 | ONNX Module | PARTIAL | P0 | Small | ort crate |
| 4 | Helpful Error Messages | **COMPLETE** | P0 | Medium | None |
| 5 | Rolling Metrics | MISSING | P1 | Medium | None |
| 6 | Short Borrow Costs | PARTIAL | P1 | Medium | None |
| 7 | load_multi/load_dir | **COMPLETE** | P1 | Small | None |
| 8 | Cost Sensitivity CLI | **COMPLETE** | P1 | Small | None |
| 9 | Position Sizing Integration | PARTIAL | P1 | Medium | None |
| 10 | Multi-Symbol Documentation | DOC GAP | P1 | Small | None |
| 11 | Parameter Sensitivity | PARTIAL | P2 | Medium | None |
| 12 | Visualization Module | MISSING | P2 | Medium | Item 2 |
| 13 | HTML Reports | MISSING | P2 | Small | Item 12 |
| 14 | Verdict System | PARTIAL | P2 | Small | None |
| 15 | Polars Backend | PARTIAL | P3 | Small | Item 2 |
| 16 | Sample Data Bundling | MISSING | P3 | Small | None |
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
| Codebase Cleanliness | **VERIFIED**: No TODOs/FIXMEs in codebase |
| ALL TESTS | **PASSING**: 532 tests (0 failures) |

---

## Recommended Execution Order

**Phase 0 - Foundation (Week 1):**
1. Python Bindings (#2) - 2-3 weeks (parallel track)
2. ~~Helpful Error Messages (#4)~~ - **COMPLETE**

**Phase 1 - Core Features (Weeks 2-3):**
3. Rolling Metrics (#5) - 2-3 days
4. Short Borrow Costs (#6) - 2-3 days
5. ~~load_multi/load_dir (#7)~~ - **COMPLETE**
6. ~~Cost Sensitivity CLI (#8)~~ - **COMPLETE**

**Phase 2 - Integration (Week 4):**
7. Position Sizing Integration (#9) - 2-3 days
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

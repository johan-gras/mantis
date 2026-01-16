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
- **LOOKAHEAD BUG (CONFIRMED)**: Orders fill on bar[i].open when strategy sees bar[i].close
- Python bindings (PyO3) completely missing

**Items Verified as CORRECT:**
- Config defaults: slippage (0.1%), commission (0.1%), position_size (10%)
- Limit order fill logic in portfolio.rs
- DSR/PSR implementation in analytics.rs
- Black-Litterman & Mean-Variance optimization fully implemented
- Walk-forward OOS degradation calculation correct
- Statistical tests (ADF, autocorrelation, Ljung-Box) - fixed in commit 0b67bff
- Walk-forward defaults (num_windows=12, in_sample_ratio=0.75, anchored=true)

---

## Test Status Summary

**Last Run:** 2026-01-16
- **Passed:** 515 tests (470 unit + 5 main + 28 integration + 12 doc tests)
- **Failed:** 0 tests
- **Ignored:** 0 tests

---

## P0 - Critical (Blocking Core Functionality)

### 1. Lookahead Bias in Order Execution [CONFIRMED - CRITICAL]
**Status:** VERIFIED via deep Opus code analysis

**Current behavior (engine.rs:306-435):**
```rust
for i in 0..bars.len() {
    let bar = &bars[i];           // bar[i] with full OHLCV
    let ctx = StrategyContext { bar_index: i, bars: &bars, ... };
    let signal = strategy.on_bar(&ctx);  // Strategy sees bar[i].close
    portfolio.execute_with_fill_probability(&order, bar, ...)  // Fills at bar[i].open
}
```

**Problem:** Strategy can see bar[i].close and execute at bar[i].open (lookahead).

**Spec requirement:** Signals generated from bar[i] should fill at bar[i+1].open.

**Evidence chain:**
- `engine.rs:307` - `bar = &bars[i]`
- `engine.rs:390-392` - Strategy context gets `bar_index: i` and `bars: &bars`
- `strategy.rs:30-31` - `current_bar()` returns `bars[bar_index]` (bar[i] with full OHLCV)
- `engine.rs:430` - `signal = strategy.on_bar(&ctx)` - strategy sees bar[i].close
- `engine.rs:432-434` - `execute_with_fill_probability(&order, bar, ...)` - passes bar[i]
- `portfolio.rs:667-669` - Default fills at `bar.open`

**Fix approaches:**
1. **Order buffering**: Store orders from bar[i], execute at bar[i+1]
2. **Data offset**: Strategy at step i sees bar[i-1].close, executes at bar[i].open

**Files affected:**
- `src/engine.rs` - Main execution loop (lines 306-435)
- `src/portfolio.rs` - market_execution_price() (lines 667-689)

**Dependencies:** None
**Effort:** Medium (2-3 days) - architectural change

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

### 4. Helpful Error Messages [MISSING - VERIFIED]
**Status:** Only basic error types exist (16 variants in 59 lines)

**Current state (src/error.rs):**
- NO `SignalShapeMismatch` error
- NO `LookaheadError` error
- NO `InvalidSignalError` error
- NO "Common causes" or "Quick fix" guidance anywhere

**Spec requirements:**
```
SignalShapeMismatch: Signal has 250 rows but data has 252 rows
  Common causes: Weekend/holiday filtering mismatch, date alignment
  Quick fix: Use `signal.reindex(data.index, fill_value=0)` to match indices
```

**Files affected:**
- `src/error.rs` - Add new error types with structured help
- `src/validation.rs` (new) - Signal validation with helpful errors

**Dependencies:** None
**Effort:** Medium (3-5 days)

---

## P1 - High Priority (Significant Feature Gaps)

### 5. Monthly Returns Synthetic [VERIFIED]
**Status:** Uses synthetic uniform distribution

**Current implementation (src/analytics.rs:924-933):**
```rust
fn monthly_returns(result: &BacktestResult) -> Vec<f64> {
    let months = (result.end_time - result.start_time).num_days() / 30;
    let monthly_return = result.total_return_pct / months as f64;
    vec![monthly_return; months as usize]  // All values identical!
}
```

**Problem:** Returns same value for all months instead of actual calendar-month aggregation.

**Fix:** Group equity curve returns by actual calendar month.

**Dependencies:** None
**Effort:** Small (1 day)

---

### 6. Auto-Warnings for Suspicious Metrics [MISSING - VERIFIED]
**Status:** No warning system exists

**Spec requirements:**
| Metric | Threshold | Warning |
|--------|-----------|---------|
| Sharpe | > 3 | "Verify data and execution" |
| Win rate | > 80% | "Check for lookahead bias" |
| Max drawdown | < 5% | "Verify execution logic" |
| Trades | < 30 | "Limited statistical significance" |
| OOS/IS | < 0.60 | "Likely overfit" |

**Files affected:**
- `src/analytics.rs` - Add `check_suspicious_metrics()` function
- `src/cli.rs` - Display warnings in CLI output

**Dependencies:** None
**Effort:** Small (1-2 days)

---

### 7. Rolling Metrics [MISSING - VERIFIED]
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

### 8. Short Borrow Costs [PARTIAL - VERIFIED]
**Status:** Only margin interest exists (no dedicated borrow fees)

**Current state (src/portfolio.rs:755-768):**
- Margin interest charged on negative cash balance (3% default)
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

### 9. load_multi / load_dir Functions [MISSING - VERIFIED]
**Status:** Not implemented

**Spec requirements:**
```rust
let data = mt.load_dir("data/prices/*.parquet")?;
let data = mt.load_multi(&["AAPL", "MSFT", "GOOGL"], "data/")?;
let sample = mt.load_sample("AAPL")?;  // Works offline
```

**Current state (src/data.rs):**
- `load_csv()` exists (line 119)
- `load_parquet()` exists (line 226)
- `load_data()` exists (line 572, auto-detect format)
- `align_series()` exists (line 1508, multi-symbol alignment)
- NO `load_multi()`, `load_dir()`, `load_sample()` functions
- NO glob pattern support

**Files affected:**
- `src/data.rs` - Add bulk loading functions

**Dependencies:** None
**Effort:** Small (1-2 days)

---

### 10. Cost Sensitivity CLI Command [PARTIAL - VERIFIED]
**Status:** Module complete (724 lines), CLI command missing

**Current state:**
- `src/cost_sensitivity.rs` - Full implementation with 1x/2x/5x/10x multipliers
- `src/cli.rs` - NO dedicated cost-sensitivity command

**Files affected:**
- `src/cli.rs` - Add `CostSensitivity` command variant

**Dependencies:** None
**Effort:** Small (1 day)

---

### 11. Position Sizing Integration [PARTIAL - VERIFIED]
**Status:** Utilities exist but not integrated into Engine

**Current state:**
- `src/engine.rs:527-549` - Only percent-of-equity sizing implemented
- `src/risk.rs:273-331` - PositionSizer utilities exist but NOT called:
  - `size_by_risk()` (line 273)
  - `size_by_volatility()` (line 284)
  - `size_by_kelly()` (line 302)
- NO FixedDollar sizing
- NO signal-scaled sizing

**Files affected:**
- `src/engine.rs` - Integrate PositionSizer into `signal_to_order()`
- Add `PositionSizingMethod` enum to BacktestConfig
- `src/risk.rs` - Add FixedDollar and SignalScaled methods

**Dependencies:** None
**Effort:** Medium (2-3 days)

---

### 12. Multi-Symbol Engine Integration [VERIFIED - DOCUMENTATION GAP]
**Status:** MultiAssetEngine exists (5583 lines) but separate from Engine

**Current state:**
- `src/engine.rs:213` - Engine.run() processes single symbol only
- `src/multi_asset.rs` - Complete MultiAssetEngine with:
  - Black-Litterman (lines 2181-2582)
  - Mean-Variance optimization (lines 1519-1958)
  - HRP, Risk Parity, Momentum, Drift strategies
  - Portfolio constraints (leverage, sector limits, turnover)
- NOT integrated: MultiAssetEngine uses Portfolio directly, not Engine

**Resolution options:**
1. Document that MultiAssetEngine is the intended multi-symbol solution
2. OR add multi-symbol execution path to Engine

**Dependencies:** None
**Effort:** Small (documentation) or Medium (integration)

---

## P2 - Medium Priority (Enhanced Functionality)

### 13. Parameter Sensitivity Analysis [PARTIAL]
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

### 14. Visualization Module [MISSING - VERIFIED]
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

### 15. HTML Report Generation [MISSING - VERIFIED]
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

## P3 - Lower Priority (Nice to Have)

### 16. Polars Backend Support [PARTIAL]
**Status:** Arrow compatibility provides foundation

**Dependencies:** Python bindings (P0 item 2)
**Effort:** Small (1-2 days)

---

### 17. Sample Data Bundling [MISSING]
**Status:** Not implemented

**Spec requirement:** `mt.load_sample("AAPL")` works offline with bundled data

**Files affected:**
- `data/samples/` (new) - Bundled sample files
- `src/data.rs` - Add `load_sample()` function
- `build.rs` - Include data in binary

**Dependencies:** None
**Effort:** Small (1 day)

---

### 18. Documentation Site [MISSING]
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
| 1 | **Lookahead Bug Fix** | CONFIRMED | P0 | Medium | None |
| 2 | Python Bindings (PyO3) | MISSING | P0 | Large | None |
| 3 | ONNX Module | PARTIAL | P0 | Small | ort crate |
| 4 | Helpful Error Messages | MISSING | P0 | Medium | None |
| 5 | Monthly Returns Fix | VERIFIED | P1 | Small | None |
| 6 | Auto-Warnings | MISSING | P1 | Small | None |
| 7 | Rolling Metrics | MISSING | P1 | Medium | None |
| 8 | Short Borrow Costs | PARTIAL | P1 | Medium | None |
| 9 | load_multi/load_dir | MISSING | P1 | Small | None |
| 10 | Cost Sensitivity CLI | PARTIAL | P1 | Small | None |
| 11 | Position Sizing Integration | PARTIAL | P1 | Medium | None |
| 12 | Multi-Symbol Integration | DOC GAP | P1 | Small | None |
| 13 | Parameter Sensitivity | PARTIAL | P2 | Medium | None |
| 14 | Visualization Module | MISSING | P2 | Medium | Item 2 |
| 15 | HTML Reports | MISSING | P2 | Small | Item 14 |
| 16 | Polars Backend | PARTIAL | P3 | Small | Item 2 |
| 17 | Sample Data Bundling | MISSING | P3 | Small | None |
| 18 | Documentation Site | MISSING | P3 | Medium | None |

---

## Items Verified as COMPLETE

| Item | Verification |
|------|-------------|
| Config Defaults | CORRECT: slippage=0.1%, commission=0.1%, position_size=10% |
| Performance Benchmarks | EXISTS in benches/backtest_bench.rs (9 benchmark groups) |
| Limit Order Fill Logic | IMPLEMENTED correctly in portfolio.rs:1584-1639 |
| DSR/PSR Implementation | IMPLEMENTED in analytics.rs:846-922 with tests passing |
| Black-Litterman | IMPLEMENTED in multi_asset.rs:2181-2582 (fully tested) |
| Mean-Variance Optimization | IMPLEMENTED in multi_asset.rs:1519-1958 (fully tested) |
| Walk-Forward OOS Degradation | IMPLEMENTED in walkforward.rs:173-182, 354-359, 414-419 |
| CPCV | IMPLEMENTED in cpcv.rs with purging/embargo (663 lines) |
| Monte Carlo | IMPLEMENTED in monte_carlo.rs with CI, bootstrap (726 lines) |
| Cost Sensitivity Module | IMPLEMENTED in cost_sensitivity.rs (724 lines) |
| Multi-Asset Strategies | 8 strategies: EqualWeight, Momentum, InverseVol, RiskParity, MVO, BL, HRP, Drift |
| Options Pricing | IMPLEMENTED: Black-Scholes, Binomial, Greeks, put-call parity |
| Streaming Indicators | IMPLEMENTED in streaming.rs (SMA, EMA, RSI, MACD, BB, ATR, StdDev) |
| Statistical Tests | FIXED in commit 0b67bff: ADF, autocorrelation, Ljung-Box all passing |
| Walk-Forward Defaults | FIXED: num_windows=12, in_sample_ratio=0.75, anchored=true |

---

## Recommended Execution Order

**Phase 0 - Immediate Fixes (Day 1):**
1. Monthly Returns fix (#5) - 1 day

**Phase 1 - Critical Fixes (Week 1-2):**
2. **Lookahead Bug Fix (#1)** - 2-3 days (MOST CRITICAL FOR CORRECTNESS)
3. Helpful Error Messages (#4) - 3-5 days

**Phase 2 - Foundation (Weeks 3-5):**
4. Python Bindings (#2) - 2-3 weeks (parallel track)
5. Auto-Warnings (#6) - 1-2 days
6. Cost Sensitivity CLI (#10) - 1 day

**Phase 3 - Core Features (Weeks 6-7):**
7. Rolling Metrics (#7) - 2-3 days
8. load_multi/load_dir (#9) - 1-2 days
9. Position Sizing Integration (#11) - 2-3 days
10. Multi-Symbol Documentation (#12) - 1 day

**Phase 4 - Realism (Week 8):**
11. Short Borrow Costs (#8) - 2-3 days
12. Parameter Sensitivity (#13) - 3-4 days

**Phase 5 - Polish (Weeks 9+):**
13. Visualization Module (#14) - 3-4 days
14. HTML Reports (#15) - 1-2 days
15. ONNX re-enablement (when ort stabilizes)
16. Lower priority items as time permits

---

## Technical Notes

### Lookahead Bug Details (CONFIRMED via Opus Analysis)

The current execution model in engine.rs (lines 306-435):

```rust
for i in 0..bars.len() {
    let bar = &bars[i];                    // bar[i] with full OHLCV
    let ctx = StrategyContext {
        bar_index: i,
        bars: &bars,                       // Strategy can see bar[i].close
        ...
    };
    let signal = strategy.on_bar(&ctx);    // Signal based on bar[i].close
    portfolio.execute(..., bar, ...)       // Fills at bar[i].open - LOOKAHEAD!
}
```

**Two fix approaches:**
1. **Order buffering**: Store orders from bar[i], execute at bar[i+1].open
2. **Data offset**: At step i, strategy sees bar[i-1].close, executes at bar[i].open

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
- Walk-forward analysis with OOS degradation (needs fold default fix)
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

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
- Multi-asset portfolio optimization (Black-Litterman, Mean-Variance)
- Options pricing (Black-Scholes, Binomial, Greeks)
- Regime detection (trend, volatility, volume)
- Streaming indicators for real-time applications
- Feature extraction for ML (40+ features, sequence building)
- Experiment tracking with SQLite
- Comprehensive CLI with 10+ commands
- Export formats: CSV, JSON, Parquet, NPY, Markdown
- Performance benchmarks (9 benchmark groups in benches/backtest_bench.rs)

**Critical Issues Discovered:**
- 9 statistical tests broken in analytics.rs (**fixed 2026-01-16**)
- Multi-symbol backtesting not integrated with main Engine
- Monthly returns use synthetic uniform distribution instead of actual aggregation

**Total Effort Estimate:** 8-12 weeks for full completion

---

## P0 - Critical (Blocking Core Functionality)

### 1. Statistical Tests Fix [COMPLETED - 2026-01-16]
**Summary:** All nine failing statistical tests in `analytics.rs` now pass (`cargo test analytics::tests::`).
Implemented adaptive lag selection for the ADF test (fall back when regression matrix is singular),
centered Durbin-Watson residuals to catch constant-series edge cases, and replaced fragile
`sin()`/integer-based pseudo-random data in the Jarque-Bera/Ljung-Box/autocorrelation tests with
seeded `StdRng` noise to avoid integer overflow and obtain realistic p-values. As a result, the
statistical validation toolkit matches the specs for normality, autocorrelation, and stationarity.

**Next steps:** None — this item is closed.

---

### 2. Python Bindings (PyO3) [MISSING]
**Description:** Specs require `pip install mantis` with native Python integration, pandas/polars DataFrame support via Arrow, and type stubs for IDE autocomplete.

**Current state:** Only subprocess-based CLI integration exists in `python/mantis_bt/`.

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

### 3. ONNX Module Re-enablement [PARTIAL]
**Description:** Complete infrastructure exists (524 lines in `src/onnx.rs`) but disabled due to `ort` crate instability (v2.0 in RC, v1.x packages yanked).

**Current state:** Commented out in `src/lib.rs:113` and `Cargo.toml:48-51`.

**Files affected:**
- `src/lib.rs` - Uncomment `mod onnx` declaration
- `Cargo.toml` - Uncomment ort dependency
- `src/onnx.rs` - May need API updates for ort v2.0

**Action:** Monitor `ort` crate releases. Re-enable when v2.0 stabilizes or alternative emerges.

**Dependencies:** External (ort crate stability)

**Effort:** Small once unblocked

---

### 4. Helpful Error Messages [MISSING]
**Description:** Specs require errors with "Common causes" and "Quick fix" guidance. Missing error variants: `SignalShapeMismatch`, `LookaheadError`, `InvalidSignalError`.

**Current state:** Errors in `src/error.rs` are minimal strings without user guidance.

**Files affected:**
- `src/error.rs` - Add new error types with structured help
- `src/validation.rs` - Signal validation with helpful errors
- `src/backtest.rs` - Lookahead detection

**Example target:**
```
SignalShapeMismatch: Signal has 250 rows but data has 252 rows
  Common causes: Weekend/holiday filtering mismatch, date alignment
  Quick fix: Use `mt.align(signal, data)` to match indices
```

**Dependencies:** None

**Effort:** Medium (3-5 days)

---

## P1 - High Priority (Significant Feature Gaps)

### 5. Multi-Symbol Backtesting Integration [NEW]
**Description:** Engine only supports single-symbol per run. Spec requires dict of DataFrames with portfolio-level metrics.

**Current state:** `engine.rs:213` processes single symbol. `multi_asset.rs` exists but not integrated with main Engine.

**Files affected:**
- `src/engine.rs` - Add multi-symbol execution path
- `src/multi_asset.rs` - Integration with Engine
- `src/results.rs` - Portfolio-level aggregation

**Dependencies:** None

**Effort:** Medium (3-5 days)

---

### 6. Monthly Returns Synthetic [NEW]
**Description:** Monthly returns use synthetic uniform distribution instead of actual calendar-month aggregation.

**Current state:** `analytics.rs:924-933` uses `vec![monthly_return; months as usize]` which creates uniform distribution.

**Files affected:**
- `src/analytics.rs` - Replace synthetic with actual calendar-month aggregation

**Implementation notes:**
- Group returns by calendar month
- Handle partial months at start/end
- Maintain proper date alignment

**Dependencies:** None

**Effort:** Small (1 day)

---

### 7. Auto-Warnings for Suspicious Metrics [MISSING]
**Description:** Spec requires warnings when Sharpe > 3, Win rate > 80%, trades < 30.

**Current state:** No `check_suspicious_metrics()` function exists.

**Files affected:**
- `src/analytics.rs` - Add `check_suspicious_metrics()` function
- `src/validation.rs` - Integrate warnings into validation workflow
- `src/cli/backtest.rs` - Display warnings in CLI output

**Dependencies:** None

**Effort:** Small (1-2 days)

---

### 8. Walk-Forward Default Folds [PARTIAL]
**Description:** Spec requires 12 folds by default; current default is 5.

**Current state:** `src/walkforward.rs:34` - `n_folds: usize = 5`

**Files affected:**
- `src/walkforward.rs` - Change default from 5 to 12

**Dependencies:** None

**Effort:** Trivial (config change)

---

### 9. Rolling Metrics [MISSING]
**Description:** Specs require `rolling_sharpe(window)`, `rolling_drawdown()` functions.

**Current state:** No rolling window calculations in `src/analytics.rs`.

**Files affected:**
- `src/analytics.rs` - Add rolling metric functions
- `src/results.rs` - Expose rolling metrics in results struct

**Implementation notes:**
- Rolling Sharpe: window-based mean/std of returns
- Rolling drawdown: track peak within window

**Dependencies:** None

**Effort:** Medium (2-3 days)

---

### 10. Short Borrow Costs [MISSING]
**Description:** Only margin interest charged for shorts. Missing: stock borrow fees, locate requirements, rebates.

**Current state:** `src/portfolio.rs` (CostModel struct) handles commissions, slippage, margin interest but not borrow costs.

**Files affected:**
- `src/portfolio.rs` - Add `BorrowCost` struct with fee schedules to CostModel
- `src/portfolio.rs` - Apply borrow costs to short positions
- `src/config.rs` - Add borrow cost configuration

**Implementation notes:**
- Hard-to-borrow list with elevated fees
- Daily accrual based on short notional
- Rebate rates for cash collateral

**Dependencies:** None

**Effort:** Medium (2-3 days)

---

### 11. load_multi / load_dir Functions [MISSING]
**Description:** Spec requires bulk directory loading with glob patterns.

**Current state:** Only individual file loading; `DataManager` requires manual symbol-by-symbol API.

**Files affected:**
- `src/data.rs` - Add `load_dir()`, `load_multi()` functions
- `src/data.rs` - Add glob pattern support

**Example target:**
```rust
let data = load_dir("data/prices/*.parquet")?;
let data = load_multi(&["AAPL", "MSFT", "GOOGL"], "data/")?;
```

**Dependencies:** None

**Effort:** Small (1-2 days)

---

### 12. Cost Sensitivity Integration [PARTIAL]
**Description:** Module complete (`src/cost_sensitivity.rs`) but standalone; not integrated into validation workflow.

**Current state:** Works independently via CLI but not accessible as `validation.cost_sensitivity`.

**Files affected:**
- `src/validation.rs` - Add cost sensitivity to ValidationResult
- `src/cli/validate.rs` - Include in validation output

**Dependencies:** None

**Effort:** Small (1 day)

---

## P2 - Medium Priority (Enhanced Functionality)

### 13. Additional Position Sizing Methods [PARTIAL]
**Description:** Missing `FixedDollar` and signal-scaled sizing methods.

**Current state:** Only risk-based and volatility-targeted implemented in `src/position_sizing.rs`.

**Files affected:**
- `src/position_sizing.rs` - Add `FixedDollar`, signal-scaled variants

**Dependencies:** None

**Effort:** Small (1 day)

---

### 14. Parameter Sensitivity Analysis [PARTIAL]
**Description:** Walk-forward tracks parameter stability but no dedicated parameter sweep module.

**Current state:** `src/walkforward.rs` has `parameter_stability` field.

**Files affected:**
- `src/sensitivity.rs` (new) - Parameter sweep analysis
- `src/cli/sensitivity.rs` (new) - CLI command

**Implementation notes:**
- Grid search over parameter ranges
- Heatmaps of metric vs parameter
- Cliff detection for fragile strategies

**Dependencies:** None (could integrate with cost_sensitivity pattern)

**Effort:** Medium (3-4 days)

---

### 15. Visualization Module [MISSING]
**Description:** Specs require `results.plot()`, `mt.compare()` charts with Plotly; ASCII sparkline fallback.

**Current state:** No visualization in Rust code.

**Files affected:**
- `src/viz.rs` (new) - Plotly JSON generation
- `src/results.rs` - Add plot methods
- `python/mantis_bt/viz.py` - Python-side rendering (if PyO3 route)

**Implementation notes:**
- Generate Plotly JSON specs from Rust
- Render in Python/browser
- ASCII fallback for terminal

**Dependencies:** Depends on Python bindings for full functionality

**Effort:** Medium (3-4 days)

---

## P3 - Lower Priority (Nice to Have)

### 16. Polars Backend Support [PARTIAL]
**Description:** Specs mention polars as first-class citizen; Arrow-based Parquet exists.

**Current state:** Arrow compatibility provides foundation. Full polars integration depends on Python bindings.

**Files affected:**
- `src/data.rs` - Polars-specific optimizations
- Python bindings - `.to_polars()` method

**Dependencies:** Python bindings (P0 item 3)

**Effort:** Small (1-2 days)

---

### 17. Sample Data Bundling [MISSING]
**Description:** Spec requires `mt.load_sample("AAPL")` working offline.

**Current state:** No bundled sample data.

**Files affected:**
- `data/samples/` (new) - Bundled sample files
- `src/data.rs` - Add `load_sample()` function
- `build.rs` - Include data in binary

**Implementation notes:**
- Small representative dataset (~1 year)
- Include in package or download on first use

**Dependencies:** None

**Effort:** Small (1 day)

---

### 18. Documentation Site [MISSING]
**Description:** Spec requires mkdocs/sphinx with search.

**Current state:** Rustdoc exists; no user-facing docs site.

**Files affected:**
- `docs/` (new) - mkdocs content
- `mkdocs.yml` (new) - Site configuration
- `.github/workflows/docs.yml` (new) - Deploy workflow

**Dependencies:** None

**Effort:** Medium (ongoing)

---

## CLEANUP - Codebase Hygiene Tasks

These are quick fixes that should be addressed alongside other work:

### C2. Fix Statistical Tests [DONE 2026-01-16]
**Task:** Repair 9 broken tests in analytics.rs (ADF, Durbin-Watson, Jarque-Bera, Ljung-Box).
**Status:** Completed in this loop (adaptive ADF lag fallback, deterministic test harness).

### C3. Fix Monthly Returns
**Task:** Replace synthetic uniform distribution with actual calendar-month aggregation.
**File:** `src/analytics.rs:924-933`
**Effort:** 1 day

### C4. Update Walk-Forward Default
**Task:** Change n_folds from 5 to 12.
**File:** `src/walkforward.rs:34`
**Effort:** 5 minutes

### C5. Remove Dead TODO
**Task:** Address or remove the single TODO comment (lib.rs:113 for ONNX).
**File:** `src/lib.rs:113`
**Effort:** 5 minutes (comment update)

### C6. Update DSR/PSR Location Reference
**Task:** DSR/PSR is implemented in analytics.rs (lines 846-922), not walkforward.rs. Update any internal documentation.
**Effort:** 5 minutes

### C7. Fix File Reference Error in Item 11
**Task:** Item 11 (Short Borrow Costs) references `src/costs.rs` which doesn't exist. CostModel is in `src/portfolio.rs`.
**File:** This document - update Item 11 file references
**Effort:** 5 minutes

### C8. Reduce Excessive unwrap() Calls
**Task:** Codebase has 578 `unwrap()` calls vs only 25 `expect()` calls. Consider replacing critical path unwraps with proper error handling or `expect("meaningful message")`.
**Files:** Throughout `src/` (highest counts: data.rs:119, multi_asset.rs:79, analytics.rs:66, portfolio.rs:52)
**Effort:** Medium (ongoing refactor)
**Priority:** Low - cosmetic but improves debugging

### C9. Consolidate Duplicate Monthly Returns Items
**Task:** Monthly returns fix appears in both P1 #7 and C3. Consolidate into single item.
**Effort:** 5 minutes (documentation fix)

### C10. Fix Jarque-Bera Integer Overflow
**Task:** Test `test_jarque_bera_normal_data` fails with "attempt to multiply with overflow" at analytics.rs:4742. Use checked arithmetic or larger integer types.
**File:** `src/analytics.rs:4742`
**Effort:** 1 hour

### C11. Fix Rustdoc Warnings
**Task:** `cargo doc` generates 3 warnings for unresolved links to `[R]` in Black-Litterman formula documentation. Escape brackets or use code blocks.
**File:** `src/multi_asset.rs:2136, 2143, 2370`
**Effort:** 10 minutes

---

## Summary Table

| ID | Item | Status | Priority | Effort | Dependencies |
|----|------|--------|----------|--------|--------------|
| 1 | Statistical Tests Fix | **COMPLETED 2026-01-16** | P0 | — | None |
| 2 | Python Bindings (PyO3) | MISSING | P0 | Large | None |
| 3 | ONNX Module | PARTIAL | P0 | Small | ort crate |
| 4 | Helpful Error Messages | MISSING | P0 | Medium | None |
| 5 | Multi-Symbol Integration | NEW | P1 | Medium | None |
| 6 | Monthly Returns Fix | NEW | P1 | Small | None |
| 7 | Auto-Warnings | MISSING | P1 | Small | None |
| 8 | Walk-Forward Folds | PARTIAL | P1 | Trivial | None |
| 9 | Rolling Metrics | MISSING | P1 | Medium | None |
| 10 | Short Borrow Costs | MISSING | P1 | Medium | None |
| 11 | load_multi/load_dir | MISSING | P1 | Small | None |
| 12 | Cost Sensitivity Integration | PARTIAL | P1 | Small | None |
| 13 | Position Sizing Methods | PARTIAL | P2 | Small | None |
| 14 | Parameter Sensitivity | PARTIAL | P2 | Medium | None |
| 15 | Visualization Module | MISSING | P2 | Medium | Item 3 |
| 16 | Polars Backend | PARTIAL | P3 | Small | Item 3 |
| 17 | Sample Data Bundling | MISSING | P3 | Small | None |
| 18 | Documentation Site | MISSING | P3 | Medium | None |

### Items Removed from Previous Plan (Now Verified Complete):
| Item | Reason |
|------|--------|
| Performance Benchmarks | EXISTS in benches/backtest_bench.rs (9 benchmark groups) |
| Limit Order Fill Logic | IMPLEMENTED correctly in portfolio.rs:1581-1643 |

### Completed Items
| Item | Date | Notes |
|------|------|-------|
| Config Defaults Fix | 2026-01-16 | Updated default position size to 10% of equity and slippage to 0.1% across config/engine/CLI/tests |
| Statistical Tests Fix | 2026-01-16 | `src/analytics.rs`: adaptive ADF lag fallback, centered Durbin-Watson, deterministic RNG-based tests |

---

## Recommended Execution Order

**Phase 0 - Immediate Fixes (Day 1):**
1. Walk-Forward Folds fix (C4) - 5 minutes
2. Remove dead TODO (C5) - 5 minutes

**Phase 1 - Critical Fixes (Week 1):**
3. Monthly Returns Fix (P1 #6) - 1 day

**Phase 2 - Foundation (Weeks 2-4):**
5. Python Bindings (P0 #2) - 2-3 weeks (parallel track)
6. Helpful Error Messages (P0 #4) - 3-5 days

**Phase 3 - Core Features (Weeks 5-6):**
7. Multi-Symbol Integration (P1 #5) - 3-5 days
8. Auto-Warnings (P1 #7) - 1-2 days
9. Rolling Metrics (P1 #9) - 2-3 days
10. load_multi/load_dir (P1 #11) - 1-2 days
11. Cost Sensitivity Integration (P1 #12) - 1 day

**Phase 4 - Realism (Week 7):**
12. Short Borrow Costs (P1 #10) - 2-3 days
13. Position Sizing Methods (P2 #13) - 1 day

**Phase 5 - Polish (Weeks 8+):**
14. Visualization Module (P2 #15) - 3-4 days
15. Parameter Sensitivity (P2 #14) - 3-4 days
16. ONNX re-enablement (when ort stabilizes)
17. Lower priority items as time permits

---

## Notes

### ONNX Blocking Issue
The `ort` crate situation requires monitoring. Check https://github.com/pykeio/ort/releases periodically. Alternative: consider `tract` crate as fallback.

### Python Bindings Architecture Decision
Two paths available:
1. **Full PyO3 with maturin (recommended)** - native speed, proper types
2. **Enhanced subprocess (current)** - simpler but limited

### Testing Strategy
Each new feature should include unit tests. Target: maintain >80% coverage.

### What's Already Excellent (No Action Needed)
- Core backtest engine with comprehensive cost modeling
- 40+ technical indicators in analytics.rs
- Monte Carlo simulation with confidence intervals
- Walk-forward analysis with OOS degradation
- CPCV (Combinatorial Purged Cross-Validation)
- Multi-asset portfolio optimization (Black-Litterman, Mean-Variance)
- Options pricing (Black-Scholes, Binomial, Greeks)
- Regime detection (trend, volatility, volume)
- Streaming indicators for real-time
- Feature extraction for ML (40+ features, sequence building)
- Experiment tracking with SQLite
- Comprehensive CLI with 10+ commands
- Export formats: CSV, JSON, Parquet, NPY, Markdown
- Performance benchmarks (9 benchmark groups)
- Limit order fill logic (fills at limit price when bar touches)

# Implementation Plan

**Status:** ALL ITEMS COMPLETE
**Last Updated:** 2026-01-16

---

## Executive Summary

The Mantis backtesting framework implementation is **complete**. All 33 planned items have been verified and implemented.

**Core Features:**
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

**Python Bindings (PyO3):**
- Full PyO3 0.22 integration with ABI3 stable Python ABI (Python 3.8+)
- Package: `mantis-bt` (installable via `pip install mantis-bt`)
- Data loading: `load()`, `load_multi()`, `load_dir()`, `load_sample()`
- Backtesting: `backtest()`, `validate()`, fluent `Backtest` API
- All 6 built-in strategies accessible
- ONNX model inference support
- pandas/polars DataFrame support
- Interactive Plotly charts in Jupyter
- Sensitivity analysis bindings
- `fractional` parameter for fractional shares support

**Key Fixes Applied:**
- LOOKAHEAD BUG: FIXED - Orders now buffer and fill at bar[i+1].open
- Statistical tests (ADF, autocorrelation, Ljung-Box): FIXED in commit 0b67bff
- FRACTIONAL SHARES DEFAULT: FIXED - Now defaults to whole shares per spec
- PER-SHARE COMMISSION: ADDED - commission_per_share field with calculate_commission_with_quantity() method

---

## Test Status Summary

**Last Run:** 2026-01-16
- **Total Tests:** 571 passed (lib) + 5 (integration) + 28 (benchmarks) + 14 (doc tests)
- **Failed:** 0 tests
- **Status:** ALL TESTS PASSING

---

## Items Verified as COMPLETE

| ID | Item | Priority | Description |
|----|------|----------|-------------|
| 1 | Statistical Tests | P0 | ADF, autocorrelation, Ljung-Box - FIXED |
| 2 | Python Bindings (PyO3) | P0 | Full PyO3 module with maturin build |
| 3 | ONNX Module | P0 | Re-enabled with ort 2.0.0-rc.11 |
| 3a | ONNX Python Bindings | P1 | OnnxModel, generate_signals(), backtest model= |
| 4 | Helpful Error Messages | P0 | SignalShapeMismatch, InvalidSignal, validation.rs |
| 5 | Rolling Metrics | P1 | rolling_sharpe, rolling_drawdown, rolling_volatility |
| 6 | Short Borrow Costs | P1 | borrow_cost_rate field, accrue_borrow_costs() |
| 7 | load_multi/load_dir | P1 | Multi-file loading with glob patterns |
| 8 | Cost Sensitivity CLI | P1 | `mantis cost-sensitivity` command |
| 9 | Position Sizing Integration | P1 | 5 sizing methods, CLI options |
| 10 | Multi-Symbol CLI | P1 | `mantis portfolio` with 9 strategies |
| 11 | Parameter Sensitivity | P2 | src/sensitivity.rs, cliff/plateau detection |
| 12 | Visualization Module | P2 | src/viz.rs, ASCII sparklines, SVG heatmaps |
| 13 | HTML Reports | P2 | Self-contained HTML with SVG charts |
| 14 | Verdict System | P2 | Robust/Borderline/LikelyOverfit classification |
| 15 | Polars Backend | P3 | pandas/polars DataFrame support |
| 16 | Sample Data Bundling | P3 | AAPL, SPY, BTC sample data embedded |
| 17 | Documentation Site | P3 | mkdocs-material with full docs/ |
| 18a | results.validate() | P1 | Walk-forward validation from result |
| 18b | mt.load_results() | P2 | Load results from JSON |
| 18c | Backtest Fluent API | P3 | Builder pattern with method chaining |
| 18d | Interactive Plotly | P2 | Auto-detect Jupyter, plotly charts |
| 18e | max_position/fill_price | P3 | Position and fill price parameters |
| 18f | Sensitivity Bindings | P3 | Python sensitivity analysis API |
| 19 | Frequency Auto-Detection | P1 | DataFrequency enum, annualization factors |
| 20 | Benchmark Comparison | P2 | alpha, beta, tracking_error, etc. |
| 21 | Split/Dividend Adjustment | P2 | mt.adjust() function |
| 22 | ATR-Based Stops | P2 | "2atr", "5trail", "2rr" syntax |
| 23 | Parallel Sweep | P3 | rayon-based parallel execution |
| 24 | Advanced Plot Features | P2 | trades, benchmark, save, theme params |
| 25 | Frequency Override | P1 | freq and trading_hours_24 params |
| 26 | Rolling Metrics Python | P2 | rolling_* methods on BacktestResult |
| 27 | Limit Order Python | P3 | order_type="limit", limit_offset |
| 28 | Volume Participation | P3 | max_volume_participation param |
| 29 | Additional Metrics/Plots | P2 | volatility, duration, plot_drawdown, etc. |
| 30 | mt.compare() Visualization | P2 | CompareResult with Plotly equity curve overlay |
| 31 | Fractional Shares Default | P1 | fractional=False per spec, whole shares by default |
| 32 | Per-Share Commission Model | P1 | commission_per_share field, commission_type="per_share" support |
| 33 | CSV Auto-Delimiter Detection | P1 | Auto-detects comma, tab, semicolon, pipe delimiters |

---

## Technical Notes

### Lookahead Bug Fix Details

The lookahead bias has been resolved using **order buffering**:

- All orders are now buffered via `buffer_order_for_next_bar()` instead of executing immediately
- Orders generated from bar[i] data now fill at bar[i+1].open (not bar[i].open)
- Stop-loss/take-profit exits are also buffered for next bar execution
- Entry tracking (entry_prices, trailing_stops) handled in `handle_pending_orders()` when orders fill
- Added `pending_exits` HashSet to prevent double-buffering of stop exits
- `PendingOrder` struct includes `signal: Option<Signal>` field for entry tracking

**Files modified:** `src/engine.rs`

### Python Bindings API Drift Fix (2026-01-16)

All API drift issues between core library and PyO3 bindings have been resolved:

1. Strategy trait: `on_bar(&mut self, bars: &[Bar])` -> `on_bar(&mut self, ctx: &StrategyContext)`
2. Strategy renames: `Breakout` -> `BreakoutStrategy`, `Momentum` -> `MomentumStrategy`
3. MeanReversion: Now takes 4 parameters (period, num_std, entry_std, exit_std)
4. WalkForwardResult: Added combined_oos_return, oos_sharpe_threshold_met, walk_forward_efficiency
5. BacktestResult: Added config, config_hash, data_checksums, experiment_id, git_info, seed
6. PyBacktestConfig::new(): Added max_position and fill_price parameters
7. Signal conversion: Replaced From trait with conditional logic

**Verification:** All 571 tests pass, `cargo check --features python` compiles successfully

### ONNX Module Status

- Uses ort 2.0.0-rc.11 (production-ready)
- Optional feature: `cargo build --features onnx`
- CUDA support disabled pending configuration
- All unit tests passing
- Full batch inference support available

### Python Bindings Architecture

**PyO3 with maturin (implemented):**
- Native speed, proper types, Jupyter integration
- Package: `mantis-bt`
- ABI3 stable Python ABI (Python 3.8+)

### Per-Share Commission Model (2026-01-16)

Per spec (`specs/execution-realism.md`), added support for per-share commission model (`commission_type="per_share"`):
- Added `commission_per_share` field to CostModel struct
- Added `calculate_commission_with_quantity()` method that applies per-share commission when set
- Updated trade execution to use the new method
- Commission calculation: `quantity * commission_per_share` (when set), otherwise falls back to flat commission

**Files modified:**
- `src/portfolio.rs`: Added commission_per_share field and calculate_commission_with_quantity() method
- `src/python/backtest.rs`: Added commission_per_share parameter to backtest() and BacktestConfig
- `src/python/fluent.rs`: Added commission_per_share() method to fluent API
- `python/mantis/__init__.pyi`: Updated type stubs

---

### CSV Auto-Delimiter Detection (2026-01-16)

Per spec (`specs/data-handling.md`), CSV loading now auto-detects delimiters:
- `DataConfig.delimiter` changed from `u8` to `Option<u8>` (None = auto-detect)
- `detect_delimiter()` function analyzes first 5 lines of CSV data
- Tries comma, tab, semicolon, and pipe delimiters
- Selects based on consistency across lines and minimum 5 fields (for OHLCV data)
- `load_csv()` auto-detects delimiter when `delimiter` is `None`

**Files modified:**
- `src/data.rs`: Added detect_delimiter() function, updated DataConfig and load_csv()

**Tests added:** 4 tests for delimiter detection

---

### Fractional Shares Default Fix (2026-01-16)

Per spec (`specs/position-sizing.md`), position sizes are rounded to **whole shares by default**:
- The `fractional` parameter defaults to `False` (whole shares)
- Set `fractional=True` for crypto or fractional brokers
- This matches the spec requirement: "Position sizes are rounded to whole shares by default"

**Files modified:**
- `src/engine.rs`: BacktestConfig default changed from `true` to `false`
- `src/config.rs`: BacktestSettings default changed from `true` to `false`
- `src/portfolio.rs`: Portfolio default changed from `true` to `false`
- `src/python/backtest.rs`: Added `fractional` parameter with `false` default
- `src/python/sweep.rs`: Updated default from `true` to `false`
- `python/mantis/__init__.py`: Added `fractional` parameter with `False` default
- `python/mantis/__init__.pyi`: Updated type stubs

---

## What's Already Excellent (No Action Needed)

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
- Position sizing utilities in risk.rs
- Deflated Sharpe Ratio and Probabilistic Sharpe Ratio
- Statistical tests (ADF, autocorrelation, Ljung-Box)

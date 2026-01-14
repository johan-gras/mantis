# Mantis Implementation Plan

## Current Status Summary

| Metric | Status |
|--------|--------|
| **Tests** | 197 passing (167 unit + 2 CLI + 28 integration + 3 doc-tests) |
| **Clippy** | 0 errors (PASSING) |
| **Cargo fmt** | PASSING |

---

## Priority 1: Acceptance Criteria Violations - COMPLETE

### 1.1 Fix Clippy Errors - COMPLETE

All 26 clippy errors have been fixed. Verification: `cargo clippy -- -D warnings` passes.

### 1.2 Fix Code Formatting - COMPLETE

Code has been formatted. Verification: `cargo fmt --check` passes.

---

## Priority 2: Missing Spec Features (HIGH)

These features are explicitly required in the spec but not implemented.

### 2.1 Benchmark Comparison for Analytics

**Spec requirement:** "Benchmark comparison" (backtest-engine.md line 28)

**Location:** `src/analytics.rs`

**Current state:** Completely absent - no benchmark-related code exists

**Implementation tasks:**
1. Add new `BenchmarkMetrics` struct with fields:
   - `alpha: f64` - Jensen's alpha
   - `beta: f64` - Portfolio beta to benchmark
   - `tracking_error: f64` - Standard deviation of excess returns
   - `information_ratio: f64` - Risk-adjusted excess return
   - `correlation: f64` - Correlation with benchmark
   - `up_capture: f64` - Upside capture ratio
   - `down_capture: f64` - Downside capture ratio

2. Add benchmark loading capability:
   - Extend `BacktestResult` with `benchmark_returns: Option<Vec<f64>>`
   - Add `BenchmarkConfig` to `BacktestConfig`
   - Support benchmark data loading via CLI (`--benchmark <path>`)

3. Implement calculation functions in `analytics.rs`:
   ```rust
   fn calculate_alpha(portfolio_returns: &[f64], benchmark_returns: &[f64], risk_free_rate: f64) -> f64
   fn calculate_beta(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64
   fn calculate_tracking_error(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64
   fn calculate_information_ratio(alpha: f64, tracking_error: f64) -> f64
   ```

4. Add benchmark comparison to `ResultFormatter::print_report()`

5. Add benchmark column to export functions in `src/export.rs`

### 2.2 Parquet Data Loading - COMPLETE

**Spec requirement:** "Support for multiple data formats (CSV, Parquet)" (backtest-engine.md line 18)

**Location:** `src/data.rs`

**Implementation Summary:**
- `load_parquet()` function in `src/data.rs` that loads OHLCV data from Parquet files
- Support for multiple column naming conventions (timestamp/date/time, open/Open/o, etc.)
- Support for multiple timestamp formats (Arrow Timestamp types, Unix timestamps, ISO strings)
- `load_data()` auto-detect function that chooses format based on file extension
- DataManager methods: `load_parquet()`, `load_parquet_with_config()`, updated `load()` to auto-detect
- CLI support via `--format` option (auto/csv/parquet) on the Run command
- Comprehensive tests: `test_load_parquet`, `test_load_parquet_matches_csv`, `test_load_data_auto_detect`, `test_data_format_detection`, `test_data_manager_parquet`, `test_data_manager_auto_detect`

### 2.3 Time-Series Alignment and Resampling - COMPLETE

**Spec requirement:** "Time-series alignment and resampling" (backtest-engine.md line 19)

**Location:** `src/data.rs`

**Implementation Summary:**
- `ResampleInterval` enum with Minute(u32), Hour(u32), Day, Week, Month variants
- `resample()` function for OHLCV aggregation with standard rules (first open, max high, min low, last close, sum volume)
- `AlignMode` enum: Inner (common timestamps only), OuterForwardFill, OuterNone
- `AlignedBars` struct for aligned multi-symbol data at a single timestamp
- `align_series()` function for aligning multiple symbol series to common timestamps
- `unalign_series()` helper to extract individual symbol data from aligned result
- CLI support via `mantis resample -i input.csv -o output.csv -I 1h` command
- Supported intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
- Tests: `test_resample_minute_to_5min`, `test_resample_minute_to_hour`, `test_resample_daily_to_weekly`, `test_resample_empty`, `test_resample_interval_to_seconds`, `test_align_series_inner`, `test_align_series_outer_none`, `test_align_series_outer_forward_fill`, `test_unalign_series`, `test_align_series_empty`, `test_aligned_bars_is_complete`

### 2.4 Missing Data Handling - COMPLETE

**Spec requirement:** "Missing data handling" (backtest-engine.md line 20)

**Location:** `src/data.rs`

**Implementation Summary:**
- `DataGap` struct with start, end timestamps and expected_bars count
- `FillMethod` enum: ForwardFill, BackwardFill, Linear, Zero
- `DataQualityReport` struct with total_bars, gaps, gap_percentage, duplicate_timestamps, invalid_bars
- `detect_gaps()` function to find gaps in time series based on expected interval
- `fill_gaps()` function to fill gaps using specified method
- `data_quality_report()` function for comprehensive data quality analysis
- CLI support via `mantis quality -d data.csv -I 86400` command (interval in seconds)
- Tests: `test_detect_gaps`, `test_detect_gaps_no_gaps`, `test_fill_gaps_forward_fill`, `test_fill_gaps_backward_fill`, `test_fill_gaps_linear`, `test_data_quality_report`, `test_data_quality_report_clean_data`

### 2.5 Corporate Actions Support

**Spec requirement:** "Corporate actions support (splits, dividends)" (backtest-engine.md line 21)

**Location:** `src/data.rs`, `src/types.rs`

**Current state:** Not implemented at all

**Implementation tasks:**
1. Add corporate action types to `src/types.rs`:
   ```rust
   pub enum CorporateActionType {
       Split { ratio: f64 },           // e.g., 2.0 for 2-for-1 split
       ReverseSplit { ratio: f64 },    // e.g., 0.1 for 1-for-10 reverse
       Dividend { amount: f64, div_type: DividendType },
       SpinOff { ratio: f64, symbol: String },
   }

   pub enum DividendType {
       Cash,
       Stock,
       Special,
   }

   pub struct CorporateAction {
       pub symbol: String,
       pub action_type: CorporateActionType,
       pub ex_date: DateTime<Utc>,
       pub record_date: Option<DateTime<Utc>>,
       pub pay_date: Option<DateTime<Utc>>,
   }
   ```

2. Add adjustment functions in `src/data.rs`:
   ```rust
   pub fn adjust_for_splits(bars: &mut [Bar], splits: &[CorporateAction])
   pub fn adjust_for_dividends(bars: &mut [Bar], dividends: &[CorporateAction], method: DividendAdjustMethod)
   pub fn apply_adjustment_factor(bars: &mut [Bar], factors: &[(DateTime<Utc>, f64)])
   ```

3. Add `adjustment_factor: Option<f64>` field to `Bar` struct (optional)

4. Support corporate actions file loading:
   ```rust
   pub fn load_corporate_actions(path: impl AsRef<Path>) -> Result<Vec<CorporateAction>>
   ```

5. CLI integration: `mantis backtest --data prices.csv --adjustments splits.csv`

### 2.6 Multi-Asset Class Support

**Spec requirement:** "Support for multiple asset classes (equities, futures, crypto)" (backtest-engine.md line 10)

**Location:** `src/multi_asset.rs`, `src/portfolio.rs`, `src/types.rs`

**Current state:** Multi-SYMBOL support exists, but no asset-CLASS differentiation

**Implementation tasks:**
1. Add asset class enum to `src/types.rs`:
   ```rust
   pub enum AssetClass {
       Equity,
       Future {
           multiplier: f64,
           tick_size: f64,
           margin_requirement: f64,  // As percentage
       },
       Crypto {
           base_precision: u8,      // Decimals for quantity
           quote_precision: u8,     // Decimals for price
       },
       Forex {
           pip_size: f64,
           lot_size: f64,
       },
       Option {
           underlying: String,
           multiplier: f64,
       },
   }
   ```

2. Update `Bar` or add `AssetConfig`:
   ```rust
   pub struct AssetConfig {
       pub symbol: String,
       pub asset_class: AssetClass,
       pub currency: String,
       pub exchange: Option<String>,
   }
   ```

3. Add asset-class-specific cost models:
   - Futures: clearing fees, exchange fees, margin interest
   - Crypto: maker/taker fees, withdrawal fees
   - Forex: spread-based costs, swap rates

4. Update portfolio calculations:
   - Futures: use notional value (price * multiplier * quantity)
   - Crypto: handle small decimal quantities
   - Margin tracking for futures positions

5. Update `MultiAssetEngine` to handle mixed asset classes

6. CLI support: `mantis backtest --asset-class futures --multiplier 50`

---

## Priority 3: Production Quality Improvements (MEDIUM)

These items significantly improve quality but are not explicit spec violations.

### 3.1 Fix Daily Returns Calculation in Analytics

**Location:** `src/analytics.rs` lines 204-212

**Current state:** `calculate_daily_returns()` reconstructs synthetic daily returns from total return, assuming uniform distribution:
```rust
let daily_return = (final_equity / initial_capital).powf(1.0 / trading_days as f64) - 1.0;
vec![daily_return; trading_days]  // All identical values!
```

**Problem:** Volatility calculations are meaningless with identical values; Sharpe/Sortino are approximations.

**Implementation tasks:**
1. Calculate actual daily returns from equity curve:
   ```rust
   fn calculate_daily_returns(equity_curve: &[EquityPoint]) -> Vec<f64> {
       equity_curve.windows(2)
           .filter(|w| is_different_day(w[0].timestamp, w[1].timestamp))
           .map(|w| (w[1].equity - w[0].equity) / w[0].equity)
           .collect()
   }
   ```

2. Update `PerformanceMetrics::from_result()` to use actual returns

3. Ensure `BacktestResult` has sufficient equity curve granularity

4. Add test coverage comparing synthetic vs actual volatility

### 3.2 Fix Drawdown Analysis Approximations

**Location:** `src/analytics.rs` lines 228-240

**Current state:** Uses rough approximations:
```rust
let avg_dd = result.max_drawdown_pct / 2.0;  // Not based on actual data
let max_dd_duration = num_days / 10;         // Arbitrary estimate
let ulcer_index = (squared_dd / 2.0).sqrt(); // Simplified
```

**Implementation tasks:**
1. Calculate actual average drawdown from equity curve:
   ```rust
   fn calculate_drawdowns(equity_curve: &[EquityPoint]) -> DrawdownAnalysis {
       // Track all drawdown periods
       // Calculate: avg_drawdown, max_duration, recovery_times
   }
   ```

2. Track drawdown periods with start/end timestamps:
   ```rust
   pub struct DrawdownPeriod {
       pub start: DateTime<Utc>,
       pub trough: DateTime<Utc>,
       pub end: Option<DateTime<Utc>>,  // None if not recovered
       pub depth_pct: f64,
       pub duration_days: i64,
   }
   ```

3. Implement proper Ulcer Index: `sqrt(mean(drawdown_pct^2))`

4. Add drawdown analysis to `ResultFormatter::print_report()`

### 3.3 Add Market Impact Modeling

**Location:** `src/portfolio.rs`

**Current state:** Only simple slippage percentage exists (volume-independent):
```rust
slippage_pct: 0.0005  // Fixed 5 bps regardless of order size
```

**Spec mentions:** "market impact" (backtest-engine.md line 13)

**Implementation tasks:**
1. Add market impact models to `CostModel`:
   ```rust
   pub enum MarketImpactModel {
       None,
       Linear { coefficient: f64 },
       SquareRoot { coefficient: f64 },
       AlmgrenChriss { sigma: f64, eta: f64, gamma: f64 },
   }
   ```

2. Extend `Bar` or `StrategyContext` to include average volume:
   ```rust
   pub struct VolumeProfile {
       pub avg_daily_volume: f64,
       pub avg_bar_volume: f64,
   }
   ```

3. Implement impact calculation:
   ```rust
   fn calculate_market_impact(
       order_size: f64,
       avg_volume: f64,
       price: f64,
       model: &MarketImpactModel
   ) -> f64
   ```

4. Apply impact in `Portfolio::execute_order()`

### 3.4 Improve Order Execution Realism

**Location:** `src/engine.rs` lines 189-256, `src/portfolio.rs` lines 180-256

**Current state:**
- Market orders always execute at `bar.open`
- Limit orders fill at exact limit price or bar extremes
- No partial fills
- No execution uncertainty

**Implementation tasks:**
1. Add execution price options:
   ```rust
   pub enum ExecutionPrice {
       Open,
       Close,
       VWAP,            // Requires volume profile
       TWAP,            // Time-weighted average
       RandomInRange,   // Random price within bar range
       Midpoint,        // (high + low) / 2
   }
   ```

2. Add partial fill support:
   ```rust
   pub struct FillResult {
       pub filled_quantity: f64,
       pub remaining_quantity: f64,
       pub fill_price: f64,
       pub partial: bool,
   }

   impl Portfolio {
       fn execute_with_fill_probability(
           &mut self,
           order: &Order,
           bar: &Bar,
           fill_probability: f64
       ) -> Result<Option<FillResult>>
   }
   ```

3. Add order aging for limit orders:
   ```rust
   pub struct PendingOrder {
       pub order: Order,
       pub created_at: DateTime<Utc>,
       pub expires_at: Option<DateTime<Utc>>,
       pub remaining_quantity: f64,
   }
   ```

4. CLI configuration: `--execution-price vwap --fill-probability 0.95`

### 3.5 Add MultiAssetExporter Test Coverage

**Location:** `src/export.rs` lines 699-823

**Current state:** `MultiAssetExporter` has 0 tests - all tests in the file are for single-asset `Exporter`

**Implementation tasks:**
1. Add test for `export_weights_csv()`:
   ```rust
   #[test]
   fn test_multi_asset_export_weights_csv() {
       let result = create_test_multi_asset_result();
       let exporter = MultiAssetExporter::new(result);
       let file = NamedTempFile::new().unwrap();
       exporter.export_weights_csv(file.path()).unwrap();
       // Verify CSV content has correct columns and data
   }
   ```

2. Add test for `export_equity_csv()` (multi-asset variant)

3. Add test for `export_report_md()` (multi-asset variant)

4. Add helper function `create_test_multi_asset_result() -> MultiAssetResult`

### 3.6 Add Streaming Indicator Serialization

**Location:** `src/streaming.rs` lines 327-438

**Current state:** `StreamingBollinger` and `StreamingATR` lack `#[derive(Serialize, Deserialize)]` while other streaming indicators have it.

**Implementation tasks:**
1. Add derive macros to `StreamingBollinger`:
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]  // Add Serialize, Deserialize
   pub struct StreamingBollinger { ... }
   ```

2. Add derive macros to `StreamingATR`:
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]  // Add Serialize, Deserialize
   pub struct StreamingATR { ... }
   ```

3. Add derive macros to `StreamingStdDev` (lines 529-536):
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]  // Add Serialize, Deserialize
   pub struct StreamingStdDev { ... }
   ```

4. Add serialization round-trip tests for all streaming indicators

---

## Priority 4: Enhancements (LOW)

Nice-to-have improvements that would enhance the engine beyond spec requirements.

### 4.1 Lot-Level Position Tracking

**Location:** `src/portfolio.rs`

**Current state:** Single `Position` per symbol with average cost basis

**Enhancement:** Add FIFO/LIFO/specific lot tracking:
```rust
pub struct TaxLot {
    pub quantity: f64,
    pub cost_basis: f64,
    pub acquired_date: DateTime<Utc>,
    pub id: uuid::Uuid,
}

pub enum LotSelectionMethod {
    FIFO,
    LIFO,
    HighestCost,
    LowestCost,
    SpecificLot(uuid::Uuid),
}
```

### 4.2 Add Risk-Free Rate to Monte Carlo

**Location:** `src/monte_carlo.rs` line 428

**Current state:** Sharpe calculation assumes risk-free rate = 0%

**Enhancement:**
```rust
pub struct MonteCarloConfig {
    // ... existing fields
    pub risk_free_rate: f64,  // Add this, default 0.0
}
```

### 4.3 Improve HMM Implementation

**Location:** `src/regime.rs` lines 670+

**Current state:** Named "HMM" but uses percentile-based state assignment, not actual Hidden Markov Model with transition probabilities.

**Enhancement options:**
1. Implement actual HMM with Baum-Welch algorithm
2. Rename to `PercentileRegimeDetector` to avoid confusion
3. Add proper transition probability matrix to regime detection

### 4.4 Add RegimeLabel Confidence Scores

**Location:** `src/regime.rs`

**Current state:** `RegimeLabel.confidence` field is always `None`

**Enhancement:**
```rust
impl RegimeDetector {
    fn calculate_confidence(&self, indicators: &RegimeIndicators) -> f64 {
        // Calculate based on indicator agreement, distance from thresholds, etc.
    }
}
```

### 4.5 Add Large-Scale Performance Benchmarks

**Location:** `benches/backtest_bench.rs`

**Current state:** Tests use small datasets (up to 2000 bars)

**Spec requirement:** "Handle 10+ years of minute-level data efficiently"

**Enhancement:**
```rust
fn bench_large_scale(c: &mut Criterion) {
    let bars = generate_minute_bars(10 * 252 * 390);  // 10 years of minute data
    // Benchmark backtest execution time
}
```

Target: < 10 seconds for 979,200 bars (10 years * 252 days * 390 minutes)

### 4.6 Add Walk-Forward ML Integration Example

**Location:** `examples/`

**Current state:** No example combining walk-forward with ML strategies

**Enhancement:** Create `examples/walkforward_ml.rs`:
```rust
// Demonstrate:
// 1. Training window feature extraction
// 2. External model training (simulated)
// 3. Walk-forward validation with rolling windows
// 4. Performance comparison across windows
```

### 4.7 Optimization Support for All Strategies

**Location:** `src/cli.rs` lines 417-420

**Current state:** Breakout and MeanReversion show "Optimization not implemented"

**Enhancement:** Add parameter ranges for these strategies:
```rust
fn get_breakout_param_ranges() -> Vec<ParamRange> {
    vec![
        ParamRange::new("lookback", 10, 100, 10),
        ParamRange::new("num_std", 1.0, 3.0, 0.5),
    ]
}

fn get_mean_reversion_param_ranges() -> Vec<ParamRange> {
    vec![
        ParamRange::new("period", 10, 50, 5),
        ParamRange::new("entry_std", 1.5, 3.0, 0.5),
        ParamRange::new("exit_std", 0.0, 1.0, 0.25),
    ]
}
```

---

## Implementation Order Recommendation

### Phase 1: Fix Acceptance Criteria - COMPLETE
All clippy errors fixed and code formatted. Verification passes.

### Phase 2: Core Data Features - COMPLETE
1. ~~Implement Parquet loading (2.2)~~ - COMPLETE
2. ~~Implement time-series alignment/resampling (2.3)~~ - COMPLETE
3. ~~Implement missing data handling (2.4)~~ - COMPLETE

### Phase 3: Analytics and Benchmarks
1. Add benchmark comparison (2.1)
2. Fix daily returns calculation (3.1)
3. Fix drawdown analysis (3.2)

### Phase 4: Asset Classes and Corporate Actions
1. Add corporate actions support (2.5)
2. Add asset class differentiation (2.6)

### Phase 5: Polish
1. Market impact modeling (3.3)
2. Order execution improvements (3.4)
3. Test coverage (3.5)
4. Streaming indicator serialization (3.6)
5. Enhancement items as time permits

---

## Completed Features (Reference)

The following spec requirements are fully implemented and verified:

- [x] Event-driven backtesting architecture
- [x] Accurate position and portfolio management
- [x] Transaction cost modeling (commissions, slippage)
- [x] Support for fractional shares and various lot sizes
- [x] CSV data loading with flexible date parsing
- [x] Parquet data loading with auto-detection
- [x] Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio
- [x] Equity curve generation and storage
- [x] Trade-level statistics
- [x] Risk-adjusted returns
- [x] Export in ML-ready formats (Parquet, NPY, CSV, JSON)
- [x] Feature engineering pipeline (`src/features.rs`)
- [x] Walk-forward validation support (`src/walkforward.rs`)
- [x] Signal generation from model predictions (`ExternalSignalStrategy`)
- [x] Intuitive CLI (`src/cli.rs`)
- [x] Configuration via files and arguments
- [x] Progress reporting (indicatif progress bars)
- [x] Output in multiple formats
- [x] Comprehensive test coverage (197 tests)
- [x] Stop-loss, take-profit, trailing stops (`src/risk.rs`)
- [x] Position sizing (risk-based, volatility-based, Kelly)
- [x] Monte Carlo simulation (`src/monte_carlo.rs`)
- [x] Multi-symbol portfolio backtesting (`src/multi_asset.rs`)
- [x] Market regime detection (`src/regime.rs`)
- [x] Streaming/incremental indicators (`src/streaming.rs`)
- [x] Example strategies (7 types + ML variants)
- [x] Realistic order execution simulation (market, limit, stop, stop-limit orders)
- [x] Time-series resampling (minute to hourly/daily/weekly/monthly)
- [x] Multi-symbol time-series alignment (inner join, outer join with forward fill)
- [x] Missing data handling (gap detection, fill methods, data quality reports)

---

## Verification Commands

```bash
# Run all tests
cargo test

# Run clippy with errors
cargo clippy -- -D warnings

# Check formatting
cargo fmt --check

# Run benchmarks
cargo bench

# Build release
cargo build --release

# Generate docs
cargo doc --no-deps --open
```

# Mantis Implementation Plan

> **Last Verified**: 2026-01-15 via comprehensive code analysis
>
> Items marked `[NOT STARTED]` were verified to have no implementation.
>
> **Verification Method**: Direct inspection of source files (engine.rs: 998 lines, data.rs: 3150 lines, portfolio.rs: 2400 lines, features.rs: 865 lines, multi_asset.rs: 5596 lines, options.rs: 1292 lines, timeframe.rs: 382 lines, onnx.rs: 523 lines).
>
> **Recent Changes**: Factor Attribution now complete with Fama-French 3-factor, Carhart 4-factor, Fama-French 5-factor, and full 6-factor models using multiple linear regression with OLS, factor loadings (betas), alpha calculation, R-squared/adjusted R-squared, and t-statistics/p-values for significance testing (21 new tests). Drift-Based Rebalancing complete with threshold-based triggering when portfolio weights drift beyond configurable thresholds, DriftRebalancingConfig with conservative/moderate/relaxed presets, DriftEqualWeightStrategy and DriftMomentumStrategy implementations. Volume participation limits complete with --max-volume-participation CLI flag and TOML configuration. Deterministic backtesting complete with seeded RNG via --seed CLI argument. Implied volatility solver complete with Newton-Raphson + Brent's method fallback. All core portfolio construction methods (equal weight, inverse volatility, risk parity, mean-variance, HRP, Black-Litterman) are complete with comprehensive constraint support.

## Current Status Summary

| Metric | Status |
|--------|--------|
| **Tests** | 433 passing (0 failing) |
| **Clippy** | 0 errors (PASSING) |
| **Cargo fmt** | PASSING |
| **Architecture** | Production-quality modular design |
| **Core Engine** | engine.rs: 998 lines, production-ready |
| **Data Handling** | data.rs: 3150 lines, comprehensive |
| **Portfolio** | portfolio.rs: 2400 lines, full order types |
| **Feature Engineering** | features.rs: 865 lines, 60+ indicators |
| **Multi-Asset** | multi_asset.rs: 5596 lines, all allocation methods |
| **Options** | options.rs: 1292 lines, BS + Greeks + IV |

---

## Implementation Roadmap

Items are organized by category and prioritized within each category. Priority reflects:
- **CRITICAL**: Required for production use / core functionality gaps
- **HIGH**: Important for professional-grade system / ML workflow support
- **MEDIUM**: Significant value-add / advanced features
- **LOW**: Nice-to-have / edge case handling

---

## 1. Core Engine Capabilities

### [COMPLETE] Event-Driven Backtesting Architecture
- Strict temporal ordering of events (data -> signals -> orders -> fills)
- Bar-by-bar processing with strategy context
- Support for market, limit, stop, stop-limit orders

### [COMPLETE] Parallel Optimization
- Rayon-based parallelization for parameter sweeps
- Grid search optimization across parameter combinations

### [COMPLETE] Walk-Forward Optimization
- Rolling training/testing windows with configurable folds
- In-sample/out-of-sample ratio configuration
- Anchored window support
- OOS Sharpe threshold checking (>= 60% of IS Sharpe)
- Parameter stability detection via hash comparison
- CLI command: `walk-forward --folds N`

### [COMPLETE] Monte Carlo Simulation
- Randomized entry timing simulation
- Confidence interval generation
- Multiple path generation
- **Note**: Library-level implementation complete; no dedicated CLI command yet

### [NOT STARTED] [CRITICAL] Live Trading Mode
- **Status**: Verified incomplete - no broker integration exists
- **IMPLEMENTED**: None (backtest-only execution logic)
- **MISSING**:
  - WebSocket/REST live data feed integration
  - Broker API connectors (Alpaca, Interactive Brokers, etc.)
  - Real-time order routing and execution
  - Position synchronization with broker
  - Live-backtest parity validation harness

### [NOT STARTED] [HIGH] Multi-Leg Order Support
- **Status**: Verified incomplete - only single-leg orders exist
- **IMPLEMENTED**: None (single-leg orders only, no multi-leg infrastructure)
- **MISSING**:
  - Atomic multi-leg execution (all legs fill or none)
  - Spread pricing for multi-leg orders
  - Pairs trading order support
  - Iron condor / butterfly / spread builders
  - Rollback semantics on partial multi-leg fills

### [MISSING] [MEDIUM] Memory-Mapped File Support
- Memory-mapped file support for large datasets
- Required for 100GB+ datasets without loading into RAM
- Lazy loading for streaming scenarios

### [MISSING] [MEDIUM] Struct-of-Arrays (SoA) Layout
- Cache-efficient data layout for performance
- Currently using Array-of-Structs pattern
- Target: Sub-millisecond bar processing for intraday strategies

### [MISSING] [MEDIUM] Plugin System for Custom Strategies
- Dynamic strategy loading without recompilation
- Hooks for pre/post trade execution
- Custom indicator registration API

### [MISSING] [LOW] Event Replay System
- Full event log capture for debugging
- Replay exact backtest sequence from logs
- Audit trail for regulatory compliance

---

## 2. Data Handling & Quality

### [COMPLETE] CSV Data Loading
- Flexible column detection
- Multiple date format support
- Configurable timestamp parsing

### [COMPLETE] Parquet Data Loading
- Native Arrow format support
- Auto-detection based on file extension
- Multiple timestamp format handling

### [COMPLETE] Time-Series Resampling
- Minute to hourly/daily/weekly/monthly
- Standard OHLCV aggregation rules
- CLI support via `mantis resample`

### [COMPLETE] Multi-Symbol Alignment
- Inner join (common timestamps only)
- Outer join with forward fill
- Configurable alignment modes

### [COMPLETE] Missing Data Handling
- Gap detection with configurable intervals
- Fill methods: forward-fill, backward-fill, linear, zero
- Data quality reports

### [COMPLETE] Corporate Actions Support
- Stock splits and reverse splits
- Cash and stock dividends
- Spin-offs
- Adjustment factor calculation

### [NOT STARTED] [HIGH] Alternative Data Integration
- **Status**: Verified incomplete - only standard OHLCV price/volume data supported
- **IMPLEMENTED**: None
- **MISSING**:
  - News and sentiment data loaders
  - Satellite imagery data integration
  - Web scraping data pipelines
  - Credit card transaction data
  - Weather data (temperature, HDD/CDD)
  - Supply chain data
  - ESG scores
  - Order flow / Level 2 data
  - Options flow data

### [PARTIAL ~20%] [HIGH] Point-in-Time (PIT) Enforcement
- **IMPLEMENTED**: Basic temporal ordering in backtest
- **MISSING**:
  - Compile-time PIT guarantees via type system
  - Separate event_time vs publication_time tracking
  - PIT fundamentals support (as-of date queries)
  - Financial statement restatement tracking
  - Lookahead bias detection at compile time

### [COMPLETE] [HIGH] Data Versioning and Reproducibility
- **IMPLEMENTED**:
  - SHA256 checksums for data files (computed during load, stored in DataManager)
  - Git commit SHA tracking (automatic capture via git2 crate)
  - Experiment UUID generation (unique ID for each backtest run)
  - Configuration hashing (SHA256 of BacktestConfig for change detection)
  - Enhanced BacktestResult metadata (experiment_id, git_info, config_hash, data_checksums fields)
  - GitInfo struct capturing commit SHA, branch name, and dirty flag
  - Data file metadata tracking (path, size, checksum)
- **Location**: src/metadata.rs, updates to src/engine.rs and src/data.rs

### [NOT STARTED] [MEDIUM] Database Backends
- **Status**: Verified incomplete - only file-based (CSV/Parquet) loading
- **IMPLEMENTED**: None
- **MISSING**:
  - PostgreSQL / TimescaleDB support
  - Cloud storage (S3, GCS)
  - Incremental data loading
  - Connection pooling

### [MISSING] [MEDIUM] Multi-Vendor Reconciliation
- Handle data from multiple providers (Polygon, IEX, Yahoo, Bloomberg)
- Discrepancy detection between vendors
- Configurable reconciliation rules (vendor precedence)
- Data source provenance tracking

### [MISSING] [MEDIUM] Live Streaming Data Support
- WebSocket data feed integration
- REST polling fallback
- Target: <10ms latency for live data

### [MISSING] [LOW] Nanosecond Timestamp Precision
- Currently millisecond precision
- Required for tick data and HFT strategies

---

## 3. Position Management & Execution

### [COMPLETE] Order Types
- Market orders (execution at bar open/close + slippage)
- Limit orders (fill when price crosses limit within bar)
- Stop orders (trigger market when stop hit)
- Stop-limit orders (trigger limit when stop hit)

### [COMPLETE] Position Tracking
- Open/closed positions with P&L
- Multi-symbol portfolio with per-asset tracking
- Short positions with correct P&L sign

### [COMPLETE] Cost Modeling
- Fixed commission per trade
- Percentage commission
- Slippage modeling (percentage-based)
- Market impact models (Linear, Square-Root, Almgren-Chriss)
- Asset-class specific fees (futures clearing, crypto maker/taker, forex spreads)

### [COMPLETE] Position Sizing
- Fixed quantity
- Fixed dollar amount
- Risk-based sizing (% of equity at risk)
- Kelly Criterion support

### [COMPLETE] Tax Lot Tracking
- FIFO, LIFO, highest-cost, lowest-cost methods
- Specific lot selection
- Per-order lot selection override

### [COMPLETE] Execution Price Models
- Open, Close, VWAP, TWAP, Midpoint, RandomInRange
- Configurable via CLI and config

### [COMPLETE] Partial Fill Support
- Probabilistic fill model
- Remaining quantity tracking
- Pending order queue with TTL

### [COMPLETE] [HIGH] Volume Participation Limits
- Limit trade size to % of bar volume (e.g., 10%)
- Prevent unrealistic large order fills
- Dynamic sizing based on liquidity
- Configuration via CLI flag --max-volume-participation
- Configuration via TOML cost settings
- Comprehensive test coverage (7 tests)

### [COMPLETE] [HIGH] Margin Requirements
- Reg T margin calculation with configurable long/short initial and maintenance percentages
- Portfolio-wide leverage tracking with configurable max leverage and optional portfolio margin mode
- Margin interest accrual on borrowed capital and explicit `BacktestError::MarginCall` signaling
- CLI/configuration flags for tuning (`--max-leverage`, `--regt-long`, `--disable-margin`, etc.)

### [PARTIAL ~50%] [MEDIUM] Queue Position Simulation
- **IMPLEMENTED**: Basic limit order fill logic
- **MISSING**:
  - Order book depth modeling
  - Time-priority queue position
  - Fill probability based on order book state

### [MISSING] [MEDIUM] Latency Simulation
- Strategy latency (signal to order)
- Network latency (order to exchange)
- Exchange processing latency
- Total latency configurable per order type

### [MISSING] [LOW] Dividend Payment Processing
- Credit cash balance on ex-dividend date
- Handle dividend reinvestment (DRIP)

---

## 4. ML/Deep Learning Integration

### [COMPLETE] Feature Engineering Pipeline
- 60+ technical indicators (SMA, RSI, MACD, Bollinger, ATR, etc.)
- Feature configuration profiles (minimal, default, comprehensive)
- Automatic lag feature generation
- Rolling window features
- Feature normalization (rolling z-score)

### [COMPLETE] Data Export for Training
- Export to NumPy .npy format
- Export to Parquet for pandas/polars
- Export to CSV
- Train/validation/test split with temporal ordering

### [COMPLETE] External Signal Strategy
- Load predictions from external models
- Signal-to-order conversion
- Integration point for PyTorch/TensorFlow models

### [PARTIAL ~60%] [HIGH] ONNX Model Inference
- **Status**: PARTIAL (~60%) - Infrastructure complete, blocked by ort crate instability
- **IMPLEMENTED** (in src/onnx.rs, 523 lines):
  - Complete ONNX inference module architecture
  - ModelConfig with normalization, versioning, fallback support
  - InferenceStats tracking (latency, success rate)
  - Batch inference API design
  - GPU/CUDA support architecture
  - Sub-millisecond latency target design
- **MISSING**:
  - Active ort crate dependency (commented out in Cargo.toml)
  - Integration testing with actual ONNX models
  - ONNXModelStrategy for live inference during backtest
  - Example demonstrating end-to-end ONNX workflow
- **BLOCKERS**: ort crate version instability - v1.15-1.16 yanked, v2.0.0-rc.11 has breaking API changes
- **NEXT STEPS**: Monitor ort crate for stable 2.0 release, update API calls, add integration tests

### [COMPLETE] Cross-Sectional Features
- Rank features across universe (percentile ranking)
- Z-score across universe (standardized scores)
- Relative momentum features (outperformance vs universe average)
- Convenience methods for common metrics (return, volume, volatility)
- Generic metric functions for custom cross-sectional calculations

### [COMPLETE] Combinatorial Purged Cross-Validation (CPCV)
- Non-overlapping, temporally ordered folds
- Embargo period between train/test
- Remove overlapping bars between splits

### [PARTIAL ~50%] [MEDIUM] Cross-Validation Methods
- **IMPLEMENTED**: CPCV (Combinatorial Purged Cross-Validation)
- **MISSING**:
  - K-fold validation with temporal ordering
  - Blocked time series split
  - Gap time series split

### [NOT STARTED] [HIGH] Python Bindings (PyO3)
- **Status**: Verified incomplete - no PyO3 in Cargo.toml dependencies
- **IMPLEMENTED**: None
- **MISSING**:
  - PyO3 crate integration
  - Pre-built Python wheel distribution
  - Jupyter kernel for interactive development
  - Inline plotting and visualization
  - Rich output formatting (DataFrames, equity curves)
  - pip-installable package

### [MISSING] [MEDIUM] Feature Store Integration
- Read features from Feast/Tecton/custom stores
- Point-in-time feature retrieval
- Feature freshness validation
- Feature drift detection

### [MISSING] [MEDIUM] RL Environment (Gym-Compatible)
- Observation space: features + portfolio state
- Action space: continuous (position size) or discrete (buy/sell/hold)
- Reward shaping options (Sharpe, return, risk-adjusted)
- Episode management for rolling windows

### [MISSING] [MEDIUM] Synthetic Data Augmentation
- Bootstrap resampling
- Block bootstrapping (preserve autocorrelation)
- Scenario generation for stress tests

### [MISSING] [LOW] Temporal Features
- Day of week encoding
- Time of day features
- Days to earnings/events

---

## 5. Multi-Timeframe Support

### [COMPLETE] Timeframe Resampling
- Standard timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
- OHLCV aggregation (open=first, high=max, low=min, close=last, volume=sum)

### [COMPLETE] [HIGH] Multi-Timeframe Strategy Interface
- **IMPLEMENTED** (in src/timeframe.rs, 382 lines):
  - TimeframeManager struct for maintaining multiple resampled bar series
  - Strategy access to multiple timeframes via StrategyContext
  - requested_timeframes() method in Strategy trait
  - Engine integration to create and pass TimeframeManager
  - Lazy evaluation (only compute requested timeframes)
  - MultiTimeframeStrategy example (Daily trend + hourly entries)
  - Comprehensive test coverage (12 tests)
- **MISSING**:
  - Custom timeframe support (non-standard intervals)
  - Partial bar handling (incomplete bar detection)
  - Timezone-aware alignment (DST transitions)

### [MISSING] [MEDIUM] Custom Timeframe Support
- Non-standard intervals (7m, 33m, etc.)
- User-defined aggregation rules

### [MISSING] [MEDIUM] Partial Bar Handling
- Incomplete bar detection during live trading
- Configurable partial bar behavior

### [MISSING] [LOW] Timezone-Aware Alignment
- DST transition handling
- Market-specific timezone support
- 24/7 market support (crypto)

---

## 6. Multi-Asset Portfolio

### [COMPLETE] Multi-Symbol Backtesting
- Portfolio strategy trait
- Allocation signals (target weights, rebalance)
- Per-symbol position tracking

### [COMPLETE] Asset Class Support
- Equities, futures, crypto, forex, options
- Asset-specific cost models
- Notional multiplier handling

### [COMPLETE] Correlation Analysis
- Rolling correlation between symbols
- Correlation matrix estimation

### [COMPLETE] [HIGH] Portfolio Construction Methods
- **IMPLEMENTED** (in src/multi_asset.rs, 5596 lines):
  - Equal-weight allocation
  - Inverse volatility weighting
  - Risk parity (using inverse volatility proxy - see note below)
  - Mean-variance optimization (Markowitz) - minimum variance, maximum Sharpe ratio, target return portfolios
  - Hierarchical Risk Parity (HRP) with dendrogram clustering
  - Black-Litterman model - combines market equilibrium with investor views, supports absolute/relative views, view confidence matrix
- **NOTE**: Risk parity uses inverse volatility proxy (~60% of true MCR-based risk parity). True marginal contribution to risk (MCR) calculation would require iterative optimization.
- **MISSING**:
  - Target volatility portfolios
  - Max diversification portfolios
  - True MCR-based risk parity

### [COMPLETE] Portfolio Constraints
- **IMPLEMENTED**:
  - Maximum position size per symbol (global and symbol-specific)
  - Minimum position size per symbol
  - Maximum sector exposure with sector classification
  - Maximum leverage enforcement
  - Minimum and maximum number of holdings
  - Turnover constraints (max turnover per rebalance)
  - Portfolio constraint validation in MultiAssetEngine
  - Three preset constraint profiles: default(), moderate(), strict()
  - Builder pattern for custom constraints
  - Comprehensive test coverage (17 tests)

### [COMPLETE] [HIGH] Rebalancing Rules
- **IMPLEMENTED**:
  - Periodic rebalancing (fixed frequency in bars) across all portfolio strategies
  - Turnover constraints to limit trading costs
  - Drift-based/threshold-based rebalancing (trigger when weights drift beyond threshold)
  - DriftRebalancingConfig with configurable thresholds (conservative, moderate, relaxed presets)
  - DriftEqualWeightStrategy and DriftMomentumStrategy as example implementations
  - Comprehensive test coverage (13 tests)
- **MISSING**:
  - Cost optimization algorithms for rebalancing
  - Volatility-adaptive rebalancing frequency

### [MISSING] [MEDIUM] Dynamic Universe Management
- Symbols added/removed over time
- Universe filters (market cap, volume, sector)
- Survivorship bias correction (delisting tracking)

### [MISSING] [MEDIUM] Cross-Currency Support
- Currency conversion for cross-currency portfolios
- FX exposure tracking

### [MISSING] [LOW] Sector/Industry Attribution
- GICS sector classification
- Sector-level risk attribution

---

## 7. Options & Derivatives

### [COMPLETE] [HIGH] Options Contract Representation
- **IMPLEMENTED**:
  - OptionContract struct with full attributes (symbol, underlying, strike, expiration, option_type, exercise_style, multiplier, settlement_type)
  - ExerciseStyle enum (European, American)
  - OptionType enum (Call, Put)
  - SettlementType enum (Physical, Cash)
- **MISSING**:
  - Options chain representation
  - Moneyness filtering (ITM, ATM, OTM)
  - DTE-based filtering

### [COMPLETE] [HIGH] Options Pricing Models
- **IMPLEMENTED** (in src/options.rs, 1292 lines):
  - Black-Scholes pricing model for European options
  - Put-call parity validation
  - Helper functions for normal CDF/PDF (cumulative_normal, normal_pdf)
  - Comprehensive test coverage (18 tests)
- **MISSING**:
  - Binomial tree model (American options)
  - Auto-detect exercise style, select appropriate model

### [COMPLETE] [HIGH] Greeks Calculation
- **IMPLEMENTED**:
  - Delta calculation (analytic from Black-Scholes)
  - Gamma calculation (second derivative of price)
  - Theta calculation (time decay)
  - Vega calculation (sensitivity to volatility)
  - Rho calculation (sensitivity to interest rate)
- **MISSING**:
  - Numerical Greeks from binomial trees
  - Portfolio Greeks aggregation

### [COMPLETE] [HIGH] Implied Volatility
- **IMPLEMENTED**:
  - Newton-Raphson IV solver (vega-based iteration, typically converges in 5-10 iterations)
  - Brent's method as fallback for edge cases
  - Bounds checking (IV range: 5% to 200%)
  - Input validation and error handling
  - Brenner-Subrahmanyam approximation for initial guess
  - Comprehensive test coverage (17 tests covering ATM, ITM, OTM, short/long dated, high/low vol scenarios)
- **MISSING**:
  - Volatility surface representation
  - Surface interpolation for missing strikes
  - Historical IV storage and parametric models (SVI, SABR)

### [NOT STARTED] [MEDIUM] Binomial Tree Model
- **Status**: Verified incomplete - only Black-Scholes implemented
- **IMPLEMENTED**: None
- **MISSING**:
  - Cox-Ross-Rubinstein binomial tree
  - American option pricing
  - Early exercise premium calculation

### [NOT STARTED] [MEDIUM] Options Chain / Vol Surface
- **Status**: Verified incomplete - single contracts only
- **IMPLEMENTED**: None
- **MISSING**:
  - Options chain representation (all strikes/expirations)
  - Volatility smile/surface modeling
  - SVI, SABR parametric models
  - Surface arbitrage detection

### [MISSING] [MEDIUM] Options Expiration Handling
- **MISSING**:
  - Automatic expiration at market close
  - ITM automatic exercise
  - Cash settlement vs physical delivery
  - Assignment risk modeling

### [MISSING] [MEDIUM] Options Strategies
- Vertical spreads (bull call, bear put)
- Straddles, strangles
- Iron condors, butterflies
- Strategy builders with max profit/loss calculation

### [MISSING] [LOW] Early Exercise (American Options)
- Optimal exercise boundary
- Dividend-aware exercise logic

---

## 8. Risk Management & Validation

### [COMPLETE] Stop-Loss & Take-Profit
- Percentage-based stop-loss
- Fixed take-profit targets
- Trailing stops

### [COMPLETE] Monte Carlo Simulation
- Return distribution analysis
- Confidence interval generation

### [COMPLETE] [HIGH] Overfitting Detection
- **IMPLEMENTED**:
  - Walk-forward validation with comprehensive overfitting detection
  - Deflated Sharpe Ratio (Lopez de Prado) - adjusts for multiple testing bias
  - Probabilistic Sharpe Ratio - statistical confidence in performance (accounts for skewness and kurtosis)
  - Out-of-sample Sharpe threshold checking (>= 60% of IS Sharpe) via `is_robust_with_sharpe()` method in WalkForwardResult
  - Parameter stability testing - tracks parameter consistency across windows via parameter hash comparison
  - `from_result_with_trials()` method in PerformanceMetrics for proper multiple testing adjustment with n_trials
  - CLI integration showing deflated Sharpe with n_trials in optimization output
  - Enhanced walk-forward output displaying overfitting detection metrics (OOS/IS Sharpe ratio, parameter stability)

### [COMPLETE] [HIGH] Transaction Cost Sensitivity
- **IMPLEMENTED**: Full transaction cost sensitivity analysis module in src/cost_sensitivity.rs
- Supports configurable cost multipliers (0x, 1x, 2x, 5x, 10x, 20x)
- Comprehensive analysis metrics:
  - Sharpe degradation (percentage decline in Sharpe ratio as costs increase)
  - Return degradation (performance impact across cost scenarios)
  - Cost elasticity (sensitivity of returns to cost changes)
  - Breakeven multiplier (maximum sustainable cost increase before profitability fails)
- Robustness assessment with configurable thresholds (5x cost test by default)
- Three preset configurations: default(), standard(), aggressive()
- Complete with 7 passing tests

### [MISSING] [HIGH] Lookahead Bias Prevention
- Compile-time PIT data type guarantees
- Automatic future-peeking detection
- Audit logs of data available at each decision point

### [PARTIAL ~50%] [MEDIUM] Cross-Validation
- **IMPLEMENTED**: CPCV (Combinatorial Purged Cross-Validation)
- **MISSING**:
  - K-fold validation with temporal ordering
  - Additional cross-validation methods

### [MISSING] [MEDIUM] Robustness Tests
- Parameter sensitivity analysis
- Regime testing (bull/bear/sideways performance)
- Data quality stress tests (missing data, outliers)

### [MISSING] [MEDIUM] Bias Detection
- Survivorship bias check
- Selection bias check
- Data snooping check
- Reporting bias check

### [MISSING] [LOW] Smoke Tests
- Multi-symbol generalization testing
- Multi-period temporal stability
- Synthetic data verification
- Zero-cost upper bound test

---

## 9. Performance Analytics

### [COMPLETE] Core Metrics
- Total return, annualized return (CAGR)
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown (% and duration)
- Win rate, profit factor, average win/loss

### [COMPLETE] Benchmark Comparison
- Alpha (Jensen's alpha, annualized)
- Beta (portfolio sensitivity to benchmark)
- Tracking error
- Information ratio
- Correlation with benchmark
- Upside/downside capture ratios

### [COMPLETE] Drawdown Analysis
- Drawdown periods with start/trough/end dates
- Drawdown depth and duration
- Ulcer Index (proper calculation from equity curve)
- Time underwater percentage

### [COMPLETE] Advanced Risk Metrics
- **IMPLEMENTED**:
  - Historical VaR (implemented in monte_carlo.rs)
  - Conditional VaR/CVaR/Expected Shortfall (implemented in monte_carlo.rs)
  - Tail ratio (95th percentile gain/loss) (analytics.rs)
  - Omega ratio (analytics.rs)
  - Kurtosis (analytics.rs)
  - Skewness (analytics.rs)

### [COMPLETE] [HIGH] Factor Attribution
- **IMPLEMENTED** (in src/analytics.rs):
  - Fama-French 3-factor model (MKT, SMB, HML)
  - Carhart 4-factor model (MKT, SMB, HML, UMD)
  - Fama-French 5-factor model (MKT, SMB, HML, RMW, CMA)
  - Full 6-factor model (all factors: MKT, SMB, HML, RMW, CMA, UMD)
  - Multiple linear regression with OLS (FactorAttribution, FactorLoadings, multiple_regression)
  - Factor loadings (betas) calculation
  - Alpha calculation (risk-adjusted excess return)
  - R-squared and adjusted R-squared for model fit
  - t-statistics and p-values for significance testing
  - Comprehensive test coverage (21 new tests)

### [NOT STARTED] [MEDIUM] Brinson Attribution
- **Status**: Verified incomplete - no attribution analysis
- **IMPLEMENTED**: None
- **MISSING**:
  - Allocation effect (sector over/underweighting)
  - Selection effect (stock picking within sectors)
  - Interaction effect

### [MISSING] [MEDIUM] Transaction Cost Analysis (TCA)
- Pre-trade cost estimation
- Post-trade cost measurement
- Implementation shortfall
- VWAP/TWAP benchmark comparison

### [MISSING] [MEDIUM] Risk-Based Attribution
- Contribution to total risk (CTR)
- Marginal contribution to risk (MCR)
- Component VaR
- Factor risk attribution

### [MISSING] [LOW] Statistical Tests
- Normality tests (Shapiro-Wilk, Jarque-Bera)
- Autocorrelation of returns (Durbin-Watson)
- Stationarity tests (ADF, KPSS)

---

## 10. Production Operations

### [NOT STARTED] [CRITICAL] Real-Time Monitoring Infrastructure
- **Status**: Verified incomplete - no monitoring code exists
- **IMPLEMENTED**: None
- **MISSING**:
  - Order execution latency tracking (target <10ms)
  - Fill rate and slippage monitoring
  - System health metrics (CPU, memory, network)
  - Structured logging (JSON lines format)
  - Performance dashboards (Grafana-compatible)

### [NOT STARTED] [CRITICAL] Risk Limits and Circuit Breakers
- **Status**: Verified incomplete - no circuit breaker implementation
- **IMPLEMENTED**: None
- **MISSING**:
  - Pre-trade risk checks (order size limits, price bands)
  - Daily loss limits with automatic trading halt
  - Position-level and portfolio-level breakers
  - Kill switch for emergency stop (<100ms response)
  - Message throttle limits

### [NOT STARTED] [CRITICAL] Position Reconciliation
- **Status**: Verified incomplete - no reconciliation logic
- **IMPLEMENTED**: None
- **MISSING**:
  - Trade reconciliation (match transactions)
  - Position reconciliation (verify balances vs broker)
  - Cash reconciliation
  - End-of-day mark-to-market
  - Discrepancy flagging and escalation

### [MISSING] [HIGH] Error Handling and Recovery
- Permanent vs transient error classification
- Automatic retry with exponential backoff
- Circuit breaker patterns
- Graceful degradation on data feed failure
- Failover to backup connectivity

### [MISSING] [HIGH] State Checkpointing
- Periodic state snapshots during long backtests
- Resume after crash without reprocessing
- Exactly-once processing semantics
- Incremental snapshots (delta encoding)
- Configurable checkpoint interval

### [MISSING] [HIGH] Broker Integration
- REST API support (Alpaca, IBKR)
- WebSocket live data feeds
- Authentication and credential management
- Rate limit handling
- Order status tracking and reconciliation

### [MISSING] [MEDIUM] Audit Trails and Compliance
- Every trade with microsecond timestamps
- Algorithm parameter change logging
- User action logging
- Immutable records (append-only logs)
- Cryptographic verification (hash chains)
- Regulatory compliance (SEC, FINRA, MiFID II)

### [MISSING] [MEDIUM] Deployment and Release Management
- Blue-green deployment
- Canary releases (gradual rollout)
- Shadow mode (run without trading)
- One-click rollback (<2 minutes)

### [MISSING] [LOW] Configuration Management
- Environment-specific configs (dev, staging, prod)
- Secrets management (environment variables/vault)
- Hot-reload for non-critical config changes
- Configuration drift detection

---

## 11. Model Governance

### [NOT STARTED] [HIGH] Model Registry
- **Status**: Verified incomplete - no model governance code
- **IMPLEMENTED**: None
- **MISSING**:
  - Centralized model store with APIs
  - Model lineage (data version, code version, hyperparameters)
  - Stage management (experimental -> staging -> production -> archived)
  - Model aliases (@champion, @challenger)
  - One-click rollback to previous version

### [NOT STARTED] [HIGH] Concept Drift Detection
- **Status**: Verified incomplete - no drift detection
- **IMPLEMENTED**: None
- **MISSING**:
  - ADDM, DDM, ECDD statistical methods
  - Prediction error rate monitoring
  - Sharpe ratio / P&L degradation alerts
  - Response actions (flag for retraining, automatic rollback)

### [NOT STARTED] [HIGH] Feature Drift Monitoring
- **Status**: Verified incomplete - no feature monitoring
- **IMPLEMENTED**: None
- **MISSING**:
  - Distribution monitoring (PSI, KS test, Chi-square)
  - Per-feature drift scoring
  - Feature importance tracking over time
  - Alert on critical feature drift

### [MISSING] [MEDIUM] A/B Testing Infrastructure
- Shadow mode (run new model alongside old)
- Champion-challenger comparison
- Statistical significance testing (t-test, Mann-Whitney)
- Minimum sample size calculation
- Early stopping rules

### [MISSING] [MEDIUM] Model Explainability
- SHAP values for feature contribution
- LIME local explanations
- Feature importance rankings
- Partial dependence plots

### [MISSING] [LOW] Canary Deployments
- Gradual traffic increase (1% -> 5% -> 20% -> 100%)
- Automatic rollback on metrics degradation
- Real-time metric comparison (canary vs control)

---

## 12. Research Workflow

### [NOT STARTED] [HIGH] Python Bindings / Jupyter Integration
- **Status**: Verified incomplete - no PyO3 in Cargo.toml dependencies
- **IMPLEMENTED**: None
- **MISSING**:
  - PyO3 crate integration
  - Pre-built Python bindings
  - Jupyter kernel for interactive development
  - Inline plotting and visualization
  - Rich output formatting (DataFrames, equity curves)

### [COMPLETE] [HIGH] Experiment Tracking
- **IMPLEMENTED**:
  - SQLite-based experiment storage (ExperimentStore in src/experiments.rs)
  - Automatic backtest logging with unique ID capture (UUID for each run)
  - Experiment metadata: hyperparameters, metrics, git SHA, config hash, data checksums
  - CLI commands for experiment management: list, show, compare, tag, note, delete
  - Query and filter capabilities: strategy name, Sharpe ratio, drawdown, date range
  - Partial ID matching for convenience (match on first N characters)
  - Colored formatted output for metrics with proper alignment
- **MISSING**:
  - MLflow integration (external dependency)
  - Weights & Biases integration (external dependency)

### [MISSING] [MEDIUM] Strategy Debugging
- Time-travel debugging (step through backtest)
- Conditional breakpoints (pause on drawdown > 10%)
- Event tracing and filtering
- Web-based debugger UI

### [MISSING] [MEDIUM] Performance Profiling
- CPU profiling (hotspot identification)
- Memory profiling
- I/O profiling
- Flame graph generation

### [MISSING] [MEDIUM] Parameter Tuning UI
- Interactive parameter sliders
- Real-time backtest updates
- Heatmaps for 2D parameter sweeps
- Bayesian optimization, genetic algorithms, Hyperband

### [MISSING] [LOW] Strategy Comparison Dashboard
- Side-by-side metrics tables
- Overlaid equity curves
- Statistical significance tests
- Multi-criteria ranking system

### [NOT STARTED] [LOW] Documentation/Onboarding
- **Status**: Verified incomplete - no tutorials or examples documentation
- **IMPLEMENTED**: None
- **MISSING**:
  - Getting started tutorial
  - Strategy development guide
  - API documentation
  - Jupyter notebook examples
  - Video walkthroughs

### [MISSING] [LOW] Visualization and Reporting
- Interactive Plotly-based charts
- HTML/PDF report generation
- Quantopian-style tearsheets
- Custom plotting API

---

## 13. CLI & Configuration

### [COMPLETE] Core Commands
- `backtest` - Run single backtest
- `optimize` - Parameter grid search
- `walk-forward` - Walk-forward optimization
- `resample` - Time-series resampling
- `quality` - Data quality analysis
- `features` - Feature extraction and export
- `experiments` - Experiment management (list/show/compare/tag/note/delete)

### [COMPLETE] Configuration
- TOML configuration file support
- Command-line flag overrides
- Sensible defaults

### [COMPLETE] Progress and Output
- Progress bars for long-running tasks
- Multiple output formats (text, JSON, CSV)
- Colorized terminal output

### [PARTIAL ~70%] [MEDIUM] Additional Commands
- **IMPLEMENTED**: Basic export functionality
- **MISSING**:
  - `monte-carlo` - Full Monte Carlo CLI (library exists, no CLI)
  - `export` - Enhanced feature/result export
  - `live` - Live trading mode
  - `validate` - Data/strategy validation
  - `analyze` - Performance report generation

### [MISSING] [MEDIUM] Shell Completion
- Bash autocomplete
- Zsh autocomplete
- Fish autocomplete

### [MISSING] [MEDIUM] Enhanced Error Messages
- Actionable suggestions on errors
- Similar file name suggestions on typos
- Config line number on parse errors

### [MISSING] [LOW] Dry-Run Mode
- `--dry-run` to preview without executing
- Show what would be executed

### [MISSING] [LOW] Resume Interrupted Operations
- `--resume` for interrupted optimizations
- Checkpoint-based resume

---

## 14. Execution Realism

### [COMPLETE] Transaction Costs
- Fixed and percentage commissions
- Slippage modeling
- Market impact models (Linear, Square-Root, Almgren-Chriss)

### [COMPLETE] Asset-Class Specific Costs
- Futures clearing/exchange fees, margin interest
- Crypto maker/taker fees, withdrawal fees
- Forex spread and swap rates

### [NOT STARTED] [HIGH] Execution Algorithms
- **Status**: Verified incomplete - only basic execution price models exist
- **IMPLEMENTED**: None (only static price models: Open, Close, VWAP, TWAP, Midpoint, RandomInRange)
- **MISSING**:
  - TWAP (Time-Weighted Average Price) algorithm
  - VWAP (Volume-Weighted Average Price) algorithm
  - POV (Percentage of Volume) algorithm
  - Implementation Shortfall algorithm
  - Arrival Price algorithms
  - Parent-child order architecture
  - Adaptive execution

### [MISSING] [MEDIUM] Bid-Ask Spread Modeling
- Symbol-specific spreads
- Volatility-dependent spreads
- Time-of-day spread variation

### [MISSING] [MEDIUM] Stress Testing
- Flash crash simulation
- Liquidity crisis scenarios
- Fat finger error handling
- Exchange outage behavior

### [MISSING] [LOW] Crypto-Specific Mechanics
- Funding rates for perpetual futures
- Liquidation risk modeling
- Exchange-specific fee structures

---

## 15. Reproducibility Requirements

### [COMPLETE] [HIGH] Deterministic Backtesting
- **IMPLEMENTED**:
  - Seeded RNG for reproducible execution via `--seed` CLI argument
  - Seed field in BacktestConfig and BacktestResult for logging
  - Portfolio.set_rng_seed() method for configurable seed
  - Seed automatically logged in experiment tracking
  - RandomInRange execution price model with deterministic output
- **MISSING**:
  - Multiple seed ensemble support (running with array of seeds)

### [COMPLETE] [HIGH] Environment Versioning
- **IMPLEMENTED**:
  - Cargo.lock committed (lock file for reproducibility)
  - Rust toolchain pinned in `rust-toolchain.toml`
- **MISSING**:
  - Dockerfile for consistent build environment
  - Build environment documentation

### [COMPLETE] [HIGH] Experiment Metadata
- **IMPLEMENTED**:
  - Unique experiment ID (UUID)
  - Git commit SHA auto-detection
  - Dirty working tree warning
  - Configuration hash
  - Data file checksum logging
- **MISSING**:
  - Hostname for distributed runs

### [MISSING] [MEDIUM] Results Caching
- Content-addressed caching for expensive operations
- Cache key = hash(input_data + parameters + code_version)
- Indicator calculation caching
- Monte Carlo result caching
- Cache invalidation on code changes

---

## Implementation Phases

### Phase 1: Critical Production Gaps [CRITICAL]
1. Real-time monitoring infrastructure
2. Risk limits and circuit breakers
3. Position reconciliation
4. Kill switch functionality

### Phase 2: ML/DL Workflow Support [HIGH]
1. ~~ONNX model inference integration~~ (PARTIAL - architecture complete, blocked by ort crate)
2. ~~Cross-sectional features~~ (COMPLETE)
3. ~~CPCV implementation~~ (COMPLETE)
4. ~~Experiment tracking~~ (COMPLETE)
5. Python bindings (PyO3) - NOT STARTED

### Phase 3: Advanced Trading Features [HIGH]
1. Live trading mode with broker integration - NOT STARTED
2. ~~Options pricing and Greeks~~ (COMPLETE - Black-Scholes, Greeks, IV solver)
3. ~~Multi-timeframe strategy interface~~ (COMPLETE)
4. ~~Portfolio optimization methods~~ (COMPLETE - all major methods)
5. Execution algorithms (TWAP, VWAP, POV) - NOT STARTED

### Phase 4: Robustness & Validation [MEDIUM]
1. ~~Deflated Sharpe Ratio~~ (COMPLETE)
2. ~~Overfitting Detection~~ (COMPLETE)
3. Lookahead bias compile-time prevention - NOT STARTED
4. ~~Factor attribution analysis~~ (COMPLETE)
5. Comprehensive robustness test suite

### Phase 5: Production Operations [MEDIUM]
1. State checkpointing and recovery
2. Audit trails and compliance
3. Model governance (registry, drift detection) - NOT STARTED
4. Deployment management (canary, shadow mode)

### Phase 6: Research & Polish [LOW]
1. Research workflow tools (debugging, profiling)
2. Visualization and reporting
3. Strategy comparison dashboard
4. Performance optimization (SoA, mmap)
5. Documentation/Onboarding

---

## Priority Ranking Summary

### CRITICAL (Production blockers):
1. Production Operations (monitoring, circuit breakers, reconciliation)
2. Live Trading Infrastructure (broker integration)

### HIGH (Professional-grade system):
1. Python Bindings (PyO3) - enables research workflow
2. ONNX Inference - complete when ort stabilizes
3. ~~Factor Attribution~~ (COMPLETE)
4. Execution Algorithms (TWAP/VWAP) - realistic execution
5. Alternative Data Integration - modern alpha sources

### MEDIUM (Advanced features):
1. Binomial Tree for American options
2. Options Chain/Vol Surface
3. Model Governance (drift detection)
4. Database Backends (PostgreSQL)
5. PIT Enforcement (compile-time)

### LOW (Polish):
1. Documentation/Onboarding
2. Additional CLI commands
3. Visualization/Reporting

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

---

## Summary Statistics

| Category | Complete | Partial | Not Started | Missing | Total Items |
|----------|----------|---------|-------------|---------|-------------|
| Core Engine | 4 | 0 | 2 | 4 | 10 |
| Data Handling | 6 | 1 | 2 | 4 | 13 |
| Position Management | 8 | 1 | 0 | 2 | 11 |
| ML Integration | 5 | 2 | 1 | 4 | 12 |
| Multi-Timeframe | 2 | 0 | 0 | 3 | 5 |
| Multi-Asset Portfolio | 6 | 0 | 0 | 3 | 9 |
| Options & Derivatives | 4 | 0 | 2 | 3 | 9 |
| Risk & Validation | 4 | 1 | 0 | 4 | 9 |
| Performance Analytics | 5 | 0 | 1 | 4 | 10 |
| Production Operations | 0 | 0 | 3 | 6 | 9 |
| Model Governance | 0 | 0 | 3 | 3 | 6 |
| Research Workflow | 1 | 0 | 2 | 5 | 8 |
| CLI & Configuration | 3 | 1 | 0 | 4 | 8 |
| Execution Realism | 2 | 0 | 1 | 3 | 6 |
| Reproducibility | 3 | 0 | 0 | 1 | 4 |
| **TOTAL** | **53** | **6** | **17** | **53** | **129** |

**Estimated Completion: ~46%**

Core backtesting is solid and production-ready. All major portfolio construction methods complete (equal weight, inverse volatility, risk parity, mean-variance, HRP, Black-Litterman) with drift-based rebalancing. Options pricing with Black-Scholes, full Greeks, and IV solver complete. Experiment tracking, cross-sectional features, CPCV, overfitting detection, and factor attribution complete. ONNX inference architecture complete but blocked by ort crate instability. Live trading, Python bindings, production operations, and model governance are NOT STARTED. Execution algorithms are NOT STARTED.

---

## Completed Features Reference

The following spec requirements are fully implemented and verified:

- [x] Event-driven backtesting architecture
- [x] Accurate position and portfolio management
- [x] Transaction cost modeling (commissions, slippage, market impact)
- [x] Support for fractional shares and various lot sizes
- [x] Tax lot tracking (FIFO, LIFO, highest-cost, lowest-cost, specific)
- [x] CSV data loading with flexible date parsing
- [x] Parquet data loading with auto-detection
- [x] Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio
- [x] Equity curve generation and storage
- [x] Trade-level statistics
- [x] Risk-adjusted returns
- [x] Export in ML-ready formats (Parquet, NPY, CSV, JSON)
- [x] Feature engineering pipeline (60+ indicators)
- [x] Walk-forward validation support
- [x] Signal generation from model predictions (ExternalSignalStrategy)
- [x] Intuitive CLI with multiple commands
- [x] Configuration via files and arguments
- [x] Progress reporting (indicatif progress bars)
- [x] Output in multiple formats (text, JSON, CSV)
- [x] Comprehensive test coverage (433 tests passing)
- [x] Stop-loss, take-profit, trailing stops
- [x] Position sizing (risk-based, volatility-based, Kelly)
- [x] Monte Carlo simulation
- [x] Multi-symbol portfolio backtesting
- [x] Multi-asset class support (equities, futures, crypto, forex)
- [x] Market regime detection
- [x] Streaming/incremental indicators
- [x] Example strategies (7 types + ML variants)
- [x] Realistic order execution (market, limit, stop, stop-limit)
- [x] Partial fills with probabilistic model
- [x] Time-series resampling (minute to monthly)
- [x] Multi-symbol time-series alignment
- [x] Missing data handling (gap detection, fill methods)
- [x] Benchmark comparison metrics (alpha, beta, tracking error, IR, capture ratios)
- [x] Actual daily returns from equity curve
- [x] Proper drawdown analysis (DrawdownPeriod, DrawdownAnalysis, Ulcer Index)
- [x] Corporate actions support (splits, dividends, spin-offs)
- [x] Options pricing models (Black-Scholes for European options)
- [x] Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- [x] Put-call parity validation
- [x] Options contract representation (strike, expiration, exercise style)
- [x] Implied volatility solver (Newton-Raphson with Brent's method fallback)
- [x] Deterministic backtesting with seeded RNG (--seed CLI argument)
- [x] Environment versioning (rust-toolchain.toml, Cargo.lock)
- [x] Multi-timeframe strategy interface (TimeframeManager, lazy evaluation)
- [x] Portfolio construction methods (equal weight, inverse vol, risk parity, MVO, HRP, Black-Litterman)
- [x] Portfolio constraints (position limits, sector exposure, leverage, turnover)
- [x] Experiment tracking with SQLite storage and CLI management
- [x] Transaction cost sensitivity analysis (breakeven, elasticity)
- [x] Overfitting detection (deflated Sharpe, OOS threshold, parameter stability)
- [x] Volume participation limits for realistic order fills
- [x] Margin requirements (Reg T, portfolio margin, margin calls)
- [x] Drift-based rebalancing (threshold-triggered, configurable presets)
- [x] Factor attribution (Fama-French 3/5-factor, Carhart 4-factor, 6-factor models)

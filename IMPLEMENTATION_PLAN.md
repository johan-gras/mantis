# Mantis Implementation Plan

> **Last Verified**: 2026-01-15 via code analysis
>
> Items marked `[NOT STARTED]` were previously claimed as partial but verified to have no implementation.
>
> **Recent Changes**: Mean-variance optimization (Markowitz) now implemented with MeanVarianceOptimizer class, supporting minimum variance, maximum Sharpe ratio, and target return portfolios. Uses Clarabel quadratic optimizer.

## Current Status Summary

| Metric | Status |
|--------|--------|
| **Tests** | 254+ passing |
| **Clippy** | 0 errors (PASSING) |
| **Cargo fmt** | PASSING |
| **Architecture** | Production-quality modular design |
| **Core Engine** | engine.rs: 942 lines, production-ready |
| **Feature Engineering** | features.rs: 865 lines, 40+ indicators |

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
- CLI command: `walk-forward --folds N`

### [COMPLETE] Monte Carlo Simulation
- Randomized entry timing simulation
- Confidence interval generation
- Multiple path generation

### [NOT STARTED] [HIGH] Live Trading Mode
- **Status**: Verified incomplete - no broker integration exists
- **IMPLEMENTED**: None (backtest-only execution logic)
- **MISSING**:
  - WebSocket/REST live data feed integration
  - Broker API connectors (Alpaca, Interactive Brokers, etc.)
  - Real-time order routing and execution
  - Position synchronization with broker
  - Live-backtest parity validation harness

### [NOT STARTED] [HIGH] Multi-Leg Order Support
- **Status**: Verified incomplete - only noted as needed, not implemented
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
- **Status**: Verified incomplete - design doc exists but no implementation
- **IMPLEMENTED**: None (only standard OHLCV price/volume data)
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
- **Location**: src/metadata.rs (new module), updates to src/engine.rs and src/data.rs
- **Tests**: 7 new tests in src/metadata.rs, all 313 library tests passing

### [MISSING] [MEDIUM] Multi-Vendor Reconciliation
- Handle data from multiple providers (Polygon, IEX, Yahoo, Bloomberg)
- Discrepancy detection between vendors
- Configurable reconciliation rules (vendor precedence)
- Data source provenance tracking

### [MISSING] [MEDIUM] Live Streaming Data Support
- WebSocket data feed integration
- REST polling fallback
- Target: <10ms latency for live data

### [MISSING] [MEDIUM] Database Backends
- PostgreSQL / TimescaleDB support
- Cloud storage (S3, GCS)
- Incremental data loading

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

### [PARTIAL ~50%] [HIGH] Queue Position Simulation
- **IMPLEMENTED**: Basic limit order fill logic
- **MISSING**:
  - Order book depth modeling
  - Time-priority queue position
  - Volume participation rate limits
  - Fill probability based on order book state

### [MISSING] [HIGH] Margin Requirements
- Reg T margin calculation
- Portfolio margin support
- Margin call triggering
- Leverage limit enforcement

### [MISSING] [MEDIUM] Volume Participation Limits
- Limit trade size to % of bar volume (e.g., 10%)
- Prevent unrealistic large order fills
- Dynamic sizing based on liquidity

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
- 40+ technical indicators (SMA, RSI, MACD, Bollinger, ATR, etc.)
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
- **Status**: PARTIAL (~60%) - Infrastructure complete, awaiting ort crate stabilization
- **IMPLEMENTED**:
  - Complete ONNX inference module architecture in src/onnx.rs
  - ModelConfig with normalization, versioning, fallback support
  - InferenceStats tracking (latency, success rate)
  - Batch inference API design
  - GPU/CUDA support architecture
  - Sub-millisecond latency target design
- **MISSING**:
  - Active ort crate dependency (v2.0 API in flux, v1.x yanked from crates.io)
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

### [PARTIAL ~20%] [HIGH] Multi-Timeframe Strategy Interface
- **Status**: Verified - only resampling works, no strategy interface
- **IMPLEMENTED**: Time-series resampling (minute to monthly)
- **MISSING**:
  - Strategy access to multiple timeframes simultaneously
  - Automatic temporal alignment across timeframes
  - Lazy evaluation (only compute requested timeframes)
  - Historical lookback at each timeframe
  - Cross-timeframe indicator calculation

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

### [PARTIAL ~85%] [HIGH] Portfolio Construction Methods
- **IMPLEMENTED**:
  - Equal-weight allocation, momentum-based allocation
  - Inverse volatility weighting, risk parity
  - Mean-variance optimization (Markowitz) - minimum variance, maximum Sharpe ratio, target return portfolios
  - Hierarchical Risk Parity (HRP)
- **MISSING**:
  - Black-Litterman model
  - Target volatility portfolios
  - Max diversification portfolios

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
  - Comprehensive test coverage (17 new tests)
- **Architecture**:
  - `PortfolioConstraints` struct in `src/multi_asset.rs`
  - Validation before order execution in rebalance operations
  - Clear error messages on constraint violations
  - Symbol-to-sector mapping support

### [PARTIAL ~60%] [HIGH] Rebalancing Rules
- **IMPLEMENTED**:
  - Periodic rebalancing (fixed frequency in bars) across all portfolio strategies
  - Turnover constraints to limit trading costs
- **MISSING**:
  - Threshold/drift-based rebalancing (trigger when weights drift beyond threshold)
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

### [PARTIAL ~10%] [HIGH] Options Contract Representation
- **IMPLEMENTED**: Basic option-type asset class
- **MISSING**:
  - Full contract attributes (strike, expiration, exercise style)
  - Options chain representation
  - Moneyness filtering (ITM, ATM, OTM)
  - DTE-based filtering

### [MISSING] [HIGH] Options Pricing Models
- Black-Scholes model (European options)
- Binomial tree model (American options)
- Put-call parity validation
- Auto-detect exercise style, select appropriate model

### [MISSING] [HIGH] Greeks Calculation
- Delta, Gamma, Theta, Vega, Rho
- Analytic Greeks from Black-Scholes
- Numerical Greeks from binomial trees
- Portfolio Greeks aggregation

### [MISSING] [HIGH] Implied Volatility
- IV solver (Newton-Raphson, Brent's method)
- Volatility surface representation
- Surface interpolation for missing strikes

### [MISSING] [MEDIUM] Options Expiration Handling
- Automatic expiration at market close
- ITM automatic exercise
- Cash settlement vs physical delivery
- Assignment risk modeling

### [MISSING] [MEDIUM] Options Strategies
- Vertical spreads (bull call, bear put)
- Straddles, strangles
- Iron condors, butterflies
- Strategy builders with max profit/loss calculation

### [MISSING] [MEDIUM] Early Exercise (American Options)
- Optimal exercise boundary
- Dividend-aware exercise logic

### [MISSING] [LOW] Volatility Surface Modeling
- SVI, SABR parametric models
- Historical IV surface storage
- Arbitrage detection

---

## 8. Risk Management & Validation

### [COMPLETE] Stop-Loss & Take-Profit
- Percentage-based stop-loss
- Fixed take-profit targets
- Trailing stops

### [COMPLETE] Monte Carlo Simulation
- Return distribution analysis
- Confidence interval generation

### [PARTIAL ~70%] [HIGH] Overfitting Detection
- **IMPLEMENTED**:
  - Walk-forward validation
  - Deflated Sharpe Ratio (Lopez de Prado) - adjusts for multiple testing bias
  - Probabilistic Sharpe Ratio - statistical confidence in performance (accounts for skewness and kurtosis)
- **MISSING**:
  - Out-of-sample performance threshold checks
  - Parameter stability testing (small changes -> small performance changes)

### [MISSING] [HIGH] Lookahead Bias Prevention
- Compile-time PIT data type guarantees
- Automatic future-peeking detection
- Audit logs of data available at each decision point

### [MISSING] [HIGH] Transaction Cost Sensitivity
- Test with 2x, 5x, 10x higher costs
- Breakeven cost analysis

### [PARTIAL ~33%] [MEDIUM] Cross-Validation
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
- **IMPLEMENTED**: All metrics now complete:
  - Historical VaR (implemented in monte_carlo.rs lines 358-365)
  - Conditional VaR/CVaR/Expected Shortfall (implemented in monte_carlo.rs)
  - Tail ratio (95th percentile gain/loss) (NEW - just implemented in analytics.rs)
  - Omega ratio (already implemented in analytics.rs lines 665-690)
  - Kurtosis (NEW - just implemented in analytics.rs)
  - Skewness (NEW - just implemented in analytics.rs)

### [MISSING] [HIGH] Factor Attribution
- Fama-French 5-factor model regression
- Factor loadings (betas)
- Alpha after factor adjustment
- R-squared for factor explanatory power
- Momentum factor (Carhart 4-factor)

### [MISSING] [MEDIUM] Brinson Attribution
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

### [MISSING] [CRITICAL] Real-Time Monitoring Infrastructure
- Order execution latency tracking (target <10ms)
- Fill rate and slippage monitoring
- System health metrics (CPU, memory, network)
- Structured logging (JSON lines format)
- Performance dashboards (Grafana-compatible)

### [MISSING] [CRITICAL] Risk Limits and Circuit Breakers
- Pre-trade risk checks (order size limits, price bands)
- Daily loss limits with automatic trading halt
- Position-level and portfolio-level breakers
- Kill switch for emergency stop (<100ms response)
- Message throttle limits

### [MISSING] [CRITICAL] Position Reconciliation
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

### [MISSING] [HIGH] Model Registry
- Centralized model store with APIs
- Model lineage (data version, code version, hyperparameters)
- Stage management (experimental -> staging -> production -> archived)
- Model aliases (@champion, @challenger)
- One-click rollback to previous version

### [MISSING] [HIGH] Concept Drift Detection
- ADDM, DDM, ECDD statistical methods
- Prediction error rate monitoring
- Sharpe ratio / P&L degradation alerts
- Response actions (flag for retraining, automatic rollback)

### [MISSING] [HIGH] Feature Drift Monitoring
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

### [MISSING] [HIGH] Experiment Tracking
- Automatic backtest logging with unique ID
- Hyperparameter capture
- Metrics logging
- Code version tracking (git SHA)
- MLflow integration
- Weights & Biases integration

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

### [COMPLETE] Configuration
- TOML configuration file support
- Command-line flag overrides
- Sensible defaults

### [COMPLETE] Progress and Output
- Progress bars for long-running tasks
- Multiple output formats (text, JSON, CSV)
- Colorized terminal output

### [PARTIAL ~60%] [MEDIUM] Additional Commands
- **IMPLEMENTED**: Basic export functionality
- **MISSING**:
  - `monte-carlo` - Full Monte Carlo CLI
  - `export` - Enhanced feature/result export
  - `live` - Live trading mode
  - `validate` - Data/strategy validation
  - `analyze` - Performance report generation
  - `compare` - Compare multiple backtest results

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

### [PARTIAL ~30%] [HIGH] Execution Algorithms
- **IMPLEMENTED**: Basic execution price models
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

### [PARTIAL ~40%] [HIGH] Deterministic Backtesting
- **IMPLEMENTED**: Basic seeded RNG for RandomInRange execution
- **MISSING**:
  - Documented RNG seed exposure
  - Multiple seed ensemble support
  - `--seed` CLI argument
  - Logging of all random operations

### [MISSING] [HIGH] Environment Versioning
- Cargo.lock committed (lock file for reproducibility)
- Rust toolchain pinned in `rust-toolchain.toml`
- Dockerfile for consistent build environment
- Build environment documentation

### [COMPLETE ~80%] [HIGH] Experiment Metadata
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
1. ~~ONNX model inference integration~~ (PARTIAL - architecture complete)
2. ~~Cross-sectional features~~ (COMPLETE)
3. ~~CPCV implementation~~ (COMPLETE)
4. Jupyter/Python bindings
5. Experiment tracking (MLflow integration)

### Phase 3: Advanced Trading Features [HIGH]
1. Live trading mode with broker integration
2. Options pricing and Greeks
3. Multi-timeframe strategy interface
4. Portfolio optimization methods
5. Execution algorithms (TWAP, VWAP, POV)

### Phase 4: Robustness & Validation [MEDIUM]
1. Deflated Sharpe Ratio
2. Lookahead bias compile-time prevention
3. Factor attribution analysis
4. Comprehensive robustness test suite

### Phase 5: Production Operations [MEDIUM]
1. State checkpointing and recovery
2. Audit trails and compliance
3. Model governance (registry, drift detection)
4. Deployment management (canary, shadow mode)

### Phase 6: Research & Polish [LOW]
1. Research workflow tools (debugging, profiling)
2. Visualization and reporting
3. Strategy comparison dashboard
4. Performance optimization (SoA, mmap)

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
| Data Handling | 6 | 1 | 1 | 6 | 14 |
| Position Management | 8 | 1 | 0 | 4 | 13 |
| ML Integration | 6 | 1 | 0 | 4 | 11 |
| Multi-Timeframe | 1 | 1 | 0 | 3 | 5 |
| Multi-Asset Portfolio | 4 | 1 | 0 | 4 | 9 |
| Options & Derivatives | 0 | 1 | 0 | 8 | 9 |
| Risk & Validation | 2 | 3 | 0 | 4 | 9 |
| Performance Analytics | 4 | 0 | 0 | 5 | 9 |
| Production Operations | 0 | 0 | 0 | 7 | 7 |
| Model Governance | 0 | 0 | 0 | 6 | 6 |
| Research Workflow | 0 | 0 | 1 | 6 | 7 |
| CLI & Configuration | 3 | 1 | 0 | 4 | 8 |
| Execution Realism | 2 | 1 | 0 | 3 | 6 |
| Reproducibility | 2 | 2 | 0 | 0 | 4 |
| **TOTAL** | **42** | **11** | **4** | **68** | **127** |

**Estimated Completion: ~40%** (core backtesting solid; Cross-Sectional Features and CPCV now complete; inverse volatility, risk parity, mean-variance optimization, Hierarchical Risk Parity (HRP), and Portfolio Constraints complete; ONNX inference architecture complete but blocked by ort crate instability; live trading and Python bindings not started)

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
- [x] Feature engineering pipeline (40+ indicators)
- [x] Walk-forward validation support
- [x] Signal generation from model predictions (ExternalSignalStrategy)
- [x] Intuitive CLI with multiple commands
- [x] Configuration via files and arguments
- [x] Progress reporting (indicatif progress bars)
- [x] Output in multiple formats (text, JSON, CSV)
- [x] Comprehensive test coverage (254+ tests)
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

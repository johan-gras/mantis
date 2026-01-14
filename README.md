# Mantis

Mantis - A high-performance backtesting engine for quantitative trading.

A production-quality backtesting engine built in Rust with first-class support for deep learning workflows.

## Features

### Core Engine
- **Event-driven backtesting** with realistic order execution
- **Transaction costs** - configurable commissions, slippage, and minimum fees
- **Risk management** - stop-loss, take-profit, trailing stops, position sizing
- **Walk-forward optimization** - prevent overfitting with out-of-sample validation
- **Parallel optimization** - leverage all CPU cores for parameter sweeps

### Technical Indicators
- Moving Averages (SMA, EMA)
- RSI, MACD, Bollinger Bands
- ATR, Stochastic, Williams %R
- CCI, OBV, VWAP
- Streaming/incremental versions for live trading

### ML/Deep Learning Integration
- **Feature extraction** - comprehensive feature engineering pipeline
- **Sequence building** - prepare data for RNNs/LSTMs/Transformers
- **Time series splitting** - proper train/val/test splits with gap
- **External signals** - backtest ML model predictions
- **Regime detection** - trend, volatility, and volume regime labels
- **Confidence-weighted trading** - position sizing based on model confidence

### Multi-Asset Support
- **Portfolio backtesting** - trade multiple assets simultaneously
- **Rebalancing strategies** - equal weight, momentum, custom allocation
- **Cross-asset correlation** - correlation-aware portfolio construction

### Monte Carlo Simulation
- **Bootstrap analysis** - resample trades to estimate confidence intervals
- **Risk metrics** - VaR, CVaR, drawdown distributions
- **Robustness scoring** - quantify strategy reliability

### Export & Reporting
- **Trade logs** - CSV, JSON, Parquet export of all trades
- **Equity curves** - timestamped equity history with drawdowns
- **Performance reports** - Markdown reports with key metrics
- **Parquet export** - efficient columnar format for ML workflows
- **NPY export** - direct export to NumPy format

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mantis.git
cd mantis

# Build in release mode
cargo build --release

# Run tests
cargo test
```

## Quick Start

### CLI Usage

```bash
# Run a backtest with SMA crossover strategy
cargo run --release -- backtest \
    --data data/sample.csv \
    --strategy sma-crossover \
    --fast 10 --slow 30 \
    --capital 100000

# Optimize strategy parameters
cargo run --release -- optimize \
    --data data/sample.csv \
    --strategy sma-crossover \
    --param fast:5,10,15,20 \
    --param slow:20,30,40,50

# Extract features for ML
cargo run --release -- features \
    --data data/sample.csv \
    --output features.csv \
    --config comprehensive

# List available strategies
cargo run --release -- strategies
```

### Library Usage

```rust
use mantis::{
    engine::{Engine, BacktestConfig},
    strategies::SmaCrossover,
    data::load_csv,
};

fn main() -> mantis::Result<()> {
    // Create engine with configuration
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 1.0,
        ..Default::default()
    };
    let mut engine = Engine::new(config);

    // Load data
    let bars = load_csv("data/AAPL.csv", &Default::default())?;
    engine.add_data("AAPL", bars);

    // Run backtest
    let mut strategy = SmaCrossover::new(10, 30);
    let result = engine.run(&mut strategy, "AAPL")?;

    println!("Return: {:.2}%", result.total_return_pct);
    println!("Sharpe: {:.2}", result.sharpe_ratio);
    println!("Max DD: {:.2}%", result.max_drawdown_pct);

    Ok(())
}
```

## ML Integration Example

### Using External Signals

```rust
use mantis::strategies::ExternalSignalStrategy;

// Load predictions from your ML model
let predictions: Vec<f64> = load_model_predictions();

// Create strategy
let mut strategy = ExternalSignalStrategy::new(predictions, 0.5)
    .with_exit_threshold(0.0)
    .with_name("LSTM Model v1");

// Run backtest
let result = engine.run(&mut strategy, "SYMBOL")?;
```

### Feature Extraction for Training

```rust
use mantis::features::{FeatureConfig, FeatureExtractor};

let config = FeatureConfig::comprehensive();
let extractor = FeatureExtractor::new(config);

// Extract features with 5-day forward returns as target
let rows = extractor.extract_with_target(&bars, 5);

// Export to CSV for Python/PyTorch
let csv = extractor.to_csv(&bars, Some(5));
std::fs::write("features.csv", csv)?;
```

### Regime Detection

```rust
use mantis::regime::{RegimeDetector, RegimeConfig};

let detector = RegimeDetector::new(RegimeConfig::default());
let regimes = detector.detect(&bars);

for (i, regime) in regimes.iter().enumerate() {
    println!("Bar {}: Trend={:?}, Vol={:?}",
        i, regime.trend, regime.volatility);
}
```

## Parquet Export

Export features, equity curves, and trades to Parquet format for efficient loading in Python:

```rust
use mantis::export::{
    export_features_parquet,
    export_equity_curve_parquet,
    export_trades_parquet,
};

// Export feature matrix
let (features, column_names) = extractor.extract_matrix(&bars);
export_features_parquet(&features, &column_names, "features.parquet")?;

// Export equity curve
export_equity_curve_parquet(&result.equity_curve, "equity.parquet")?;

// Export trades
export_trades_parquet(&result.trades, "trades.parquet")?;
```

Loading in Python:
```python
import pandas as pd

features = pd.read_parquet("features.parquet")
equity = pd.read_parquet("equity.parquet")
trades = pd.read_parquet("trades.parquet")
```

## Python Integration

See `examples/pytorch_integration.py` for a complete example of:
1. Exporting features from Rust
2. Training an LSTM in PyTorch
3. Generating predictions
4. Running backtests with ML signals

```python
# Train model and generate predictions
model = LSTMPredictor(input_size=n_features)
model = train_model(model, train_dataset, val_dataset)
predictions = generate_predictions(model, features)

# Export for Mantis backtest
predictions.to_csv("predictions.csv")

# Run backtest
# cargo run -- backtest --strategy external-signal --signals predictions.csv
```

## Built-in Strategies

| Strategy | Description |
|----------|-------------|
| `SmaCrossover` | Simple/Exponential MA crossover |
| `MacdStrategy` | MACD line crossover |
| `RsiStrategy` | RSI overbought/oversold |
| `MomentumStrategy` | Price momentum |
| `MeanReversion` | Bollinger Band mean reversion |
| `BreakoutStrategy` | Donchian channel breakout |
| `ExternalSignalStrategy` | Use ML model predictions |
| `ClassificationStrategy` | Discrete class predictions |
| `EnsembleSignalStrategy` | Combine multiple models |

## Risk Management

```rust
use mantis::risk::{RiskConfig, StopLoss, TakeProfit};

let risk_config = RiskConfig {
    stop_loss: StopLoss::Percentage(5.0),     // 5% stop
    take_profit: TakeProfit::Percentage(10.0), // 10% target
    max_drawdown_pct: Some(20.0),              // Halt at 20% DD
    risk_per_trade_pct: 2.0,                   // 2% risk per trade
    ..Default::default()
};
```

## Monte Carlo Analysis

```rust
use mantis::monte_carlo::{MonteCarloSimulator, MonteCarloConfig};

let config = MonteCarloConfig::default()
    .with_simulations(1000)
    .with_confidence(0.95);

let mut simulator = MonteCarloSimulator::new(config);
let mc_result = simulator.simulate_from_result(&backtest_result);

println!("{}", mc_result.summary());
println!("95% CI: [{:.2}%, {:.2}%]",
    mc_result.return_ci.0, mc_result.return_ci.1);
```

## Multi-Asset Portfolio

```rust
use mantis::multi_asset::{MultiAssetEngine, EqualWeightStrategy};

let mut engine = MultiAssetEngine::new(config);
engine.add_data("AAPL", aapl_bars);
engine.add_data("GOOG", goog_bars);
engine.add_data("MSFT", msft_bars);

let mut strategy = EqualWeightStrategy::new(20); // Rebalance every 20 bars
let result = engine.run(&mut strategy)?;
```

## Performance

The engine is optimized for speed:

| Operation | Performance |
|-----------|-------------|
| 1000-bar backtest | ~50 microseconds |
| 9-param optimization | ~500 microseconds |
| Feature extraction (1000 bars) | ~1 millisecond |
| Monte Carlo (1000 sims) | ~10 milliseconds |

Run benchmarks: `cargo bench`

## Configuration Files

```toml
# backtest.toml
[backtest]
initial_capital = 100000.0
position_size = 1.0
allow_short = true

[data]
path = "data/sample.csv"
symbol = "AAPL"

[strategy]
name = "sma-crossover"

[strategy.params]
fast_period = 10
slow_period = 30

[costs]
commission_pct = 0.1
slippage_pct = 0.05

[risk]
stop_loss_type = "percentage"
stop_loss_value = 5.0
take_profit_type = "percentage"
take_profit_value = 10.0
```

## Modules

| Module | Description |
|--------|-------------|
| `engine` | Core backtest execution |
| `data` | Data loading, technical indicators |
| `strategy` | Strategy trait and context |
| `strategies` | Built-in trading strategies |
| `portfolio` | Position and order management |
| `risk` | Stop-loss, take-profit, position sizing |
| `features` | ML feature extraction |
| `regime` | Market regime detection |
| `monte_carlo` | Monte Carlo simulation |
| `multi_asset` | Multi-symbol portfolio backtesting |
| `walkforward` | Walk-forward optimization |
| `streaming` | Incremental indicator calculations |
| `export` | Result export utilities |
| `analytics` | Performance metrics |
| `config` | TOML configuration support |

## License

MIT License - see LICENSE file for details.

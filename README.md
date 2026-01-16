# Mantis

**Fast enough to think, honest enough to trust.**

A high-performance backtesting engine for deep learning researchers who need fast iteration and honest validation.

[![PyPI version](https://badge.fury.io/py/mantis-bt.svg)](https://badge.fury.io/py/mantis-bt)
[![Tests](https://github.com/johan-gras/mantis/actions/workflows/ci.yml/badge.svg)](https://github.com/johan-gras/mantis/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-80%25+-green.svg)](https://github.com/johan-gras/mantis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why Mantis?

| Problem | Mantis Solution |
|---------|-----------------|
| Backtests take minutes | **< 100ms** for 10 years of daily data |
| Easy to overfit without knowing | **Walk-forward validation** built-in with verdicts |
| Transaction costs ignored | **Conservative defaults**: 0.1% commission + slippage |
| Complex APIs slow you down | **5 lines** from data to validated results |

## Quick Start

```bash
pip install mantis-bt
```

```python
import mantis as mt

# Load bundled sample data (works offline)
data = mt.load_sample("AAPL")

# Create a simple signal
signal = (data['close'] > data['close'].mean()).astype(int)

# Run backtest with realistic costs
results = mt.backtest(data, signal)

# Validate to detect overfitting
validation = results.validate()
print(validation.verdict)  # "robust", "borderline", or "likely_overfit"

# Visualize
results.plot()
```

## Features

### Core Engine
- **Signal-based backtesting** with realistic order execution
- **Transaction costs** - configurable commissions and slippage models (fixed %, sqrt market impact)
- **Position sizing** - percent of equity, volatility-targeted, signal-scaled, risk-based (ATR)
- **Risk management** - stop-loss (%, ATR, trailing), take-profit, max position limits
- **Short selling** with borrow costs

### Validation (What Makes Mantis Different)
- **Walk-forward analysis** - 12-fold default with in-sample/out-of-sample comparison
- **Deflated Sharpe Ratio** - adjusts for multiple testing bias
- **Probabilistic Sharpe Ratio** - probability your Sharpe is genuine
- **Monte Carlo simulation** - confidence intervals via block bootstrap
- **Auto-warnings** - flags suspicious metrics (Sharpe > 3, win rate > 80%, etc.)

### ML/Deep Learning Integration
- **ONNX model inference** - run your trained models directly in backtests
- **External signals** - backtest any model's predictions
- **Confidence-weighted trading** - position sizing based on model confidence

### Multi-Asset Support
- **Portfolio backtesting** - trade multiple symbols simultaneously
- **Strategy comparison** - side-by-side metrics and visualization

### Technical Indicators
- Moving Averages (SMA, EMA)
- RSI, MACD, Bollinger Bands
- ATR, Stochastic, Williams %R
- CCI, OBV, VWAP

### Export & Reporting
- **HTML reports** - self-contained with charts and metrics
- **JSON export** - save and reload results
- **Plotly visualization** - interactive charts in Jupyter
- **ASCII fallback** - works in any terminal

## Performance

| Operation | Time |
|-----------|------|
| 10-year daily backtest (2,520 bars) | ~85ms |
| 1,000 parameter sweep | ~28s |
| Walk-forward validation (12 folds) | < 2s |
| ONNX inference | ~1.8μs/prediction |

Benchmarks run on modern hardware. Run `cargo bench --features onnx` to verify on your system.

## Examples

### Using an ONNX Model

```python
import mantis as mt

data = mt.load_sample("AAPL")

# Load your trained model
model = mt.load_model("my_model.onnx", input_size=10)

# Generate signals from model predictions
signals = mt.generate_signals(model, features, threshold=0.5)

# Backtest with realistic execution
results = mt.backtest(data, signals)
print(results.summary())
```

### Parameter Sensitivity Analysis

```python
import mantis as mt

data = mt.load_sample("AAPL")

# Test SMA crossover across parameter ranges
result = mt.sensitivity(
    data,
    strategy="sma-crossover",
    params={
        "fast_period": mt.linear_range(5, 20, 4),
        "slow_period": mt.linear_range(20, 60, 5),
    }
)

# Check if strategy is fragile
print(f"Stability score: {result.stability_score():.2f}")
print(f"Best params: {result.best_params()}")
```

### Cost Sensitivity Analysis

```python
import mantis as mt

data = mt.load_sample("AAPL")
signal = (data['close'] > data['close'].mean()).astype(int)

# Test at 1x, 2x, 5x, 10x costs
result = mt.cost_sensitivity(data, signal)

# Check robustness to higher costs
print(f"Sharpe at 5x costs: {result.scenario_at(5.0).sharpe:.2f}")
print(f"Is robust: {result.is_robust()}")
```

### Monte Carlo Simulation

```python
import mantis as mt

data = mt.load_sample("AAPL")
signal = (data['close'] > data['close'].mean()).astype(int)

results = mt.backtest(data, signal)
mc = results.monte_carlo(n_simulations=1000)

print(f"95% CI for return: [{mc.return_ci[0]:.2%}, {mc.return_ci[1]:.2%}]")
print(f"Probability of positive return: {mc.prob_positive_return:.1%}")
print(mc.verdict)  # "robust", "borderline", or "likely_overfit"
```

### Strategy Comparison

```python
import mantis as mt

data = mt.load_sample("AAPL")

# Compare different strategies
results_sma = mt.backtest(data, strategy="sma-crossover", strategy_params={"fast": 10, "slow": 30})
results_rsi = mt.backtest(data, strategy="rsi", strategy_params={"period": 14})

comparison = mt.compare([results_sma, results_rsi], names=["SMA Crossover", "RSI"])
comparison.plot()
```

## Built-in Strategies

| Strategy | Description |
|----------|-------------|
| `sma-crossover` | Simple/Exponential MA crossover |
| `rsi` | RSI overbought/oversold |
| `macd` | MACD line crossover |
| `momentum` | Price momentum |
| `mean-reversion` | Bollinger Band mean reversion |
| `breakout` | Donchian channel breakout |

## Configuration Options

```python
import mantis as mt

# Full control over execution
results = mt.backtest(
    data,
    signal,
    commission=0.001,          # 0.1% commission
    slippage="sqrt",           # Square-root market impact model
    size="volatility",         # Volatility-targeted sizing
    target_vol=0.15,           # 15% target volatility
    stop_loss="2atr",          # 2x ATR stop loss
    take_profit="3atr",        # 3x ATR take profit
    max_leverage=1.5,          # Max 1.5x leverage
    allow_short=True,
    borrow_cost=0.03,          # 3% annual borrow cost
)
```

## Documentation

- [Quick Start](https://mantis.dev/docs/quickstart)
- [Cookbook](https://mantis.dev/docs/cookbook)
- [API Reference](https://mantis.dev/docs/api)
- [Colab Notebooks](notebooks/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

> **Disclaimer:** Past performance does not guarantee future results. Backtest results are hypothetical and do not represent actual trading. Real trading involves risk of loss. All performance metrics shown are based on historical simulations with assumptions that may not reflect actual market conditions.

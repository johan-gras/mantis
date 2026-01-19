# Running Backtests

## Basic Backtest

```python
import mantis as mt
import numpy as np

# Load data
data = mt.load_sample("AAPL")

# Create signal: 1=long, -1=short, 0=flat
signal = np.random.choice([-1, 0, 1], size=len(data["close"]))

# Run backtest
benchmark = mt.load_sample("SPY")
results = mt.backtest(data, signal, benchmark=benchmark)

# View results
print(results)
```

## Configuration Options

```python
results = mt.backtest(
    data,
    signal,
    # Costs
    commission=0.001,      # 0.1% per trade
    slippage=0.001,        # 0.1% per trade

    # Position sizing
    size=0.10,             # 10% of equity per trade

    # Capital
    cash=100_000,          # Starting capital

    # Short selling
    allow_short=True,      # Allow short positions
    borrow_cost=0.03,      # 3% annual borrow cost

    # Risk controls
    stop_loss=0.02,        # 2% stop loss
    take_profit=0.05,      # 5% take profit
)
```

## Fluent API

Chain configuration for complex setups:

```python
results = (
    mt.Backtest(data, signal)
    .commission(0.0005)      # 0.05%
    .slippage(0.0005)        # 0.05%
    .size(0.15)              # 15% per trade
    .cash(50_000)
    .stop_loss(0.02)
    .take_profit(0.05)
    .allow_short()
    .borrow_cost(0.03)
    .run()
)
```

## Built-in Strategies

Run backtests with built-in strategies instead of signals:

```python
# SMA Crossover
results = mt.backtest(data, strategy="sma-crossover")

# Other strategies
results = mt.backtest(data, strategy="momentum")
results = mt.backtest(data, strategy="mean-reversion")
results = mt.backtest(data, strategy="rsi")
results = mt.backtest(data, strategy="macd")
results = mt.backtest(data, strategy="breakout")
```

## Accessing Results

```python
results = mt.backtest(data, signal)

# Key metrics
results.total_return      # Total return %
results.sharpe            # Sharpe ratio
results.sortino           # Sortino ratio
results.calmar            # Calmar ratio
results.max_drawdown      # Maximum drawdown
results.win_rate          # Winning trade %
results.profit_factor     # Gross profit / gross loss

# All metrics as dict
metrics = results.metrics()

# Equity curve
equity = results.equity_curve  # numpy array

# Trade list
trades = results.trades  # List of Trade objects
```

## Benchmark Comparison

```python
benchmark = mt.load_sample("SPY")
results = mt.backtest(data, signal, benchmark=benchmark)

# Default: buy-and-hold
print(f"Strategy:     {results.total_return:.1%}")
print(f"Benchmark:    {results.benchmark_return:.1%}")
print(f"Excess:       {results.excess_return:+.1%}")
```

## Signal Validation

Check your signal before backtesting:

```python
# Validate signal
issues = mt.signal_check(data, signal)
if issues:
    for issue in issues:
        print(f"Warning: {issue}")
else:
    results = mt.backtest(data, signal)
```

Common issues detected:

- Signal length mismatch
- NaN values
- Signals outside [-1, 1] range
- Potential lookahead bias

## Parameter Sweep

Test multiple parameter combinations:

```python
def create_signal(fast, slow):
    import pandas as pd

    close = pd.Series(data["close"])
    fast_sma = close.rolling(fast).mean()
    slow_sma = close.rolling(slow).mean()
    return np.where(fast_sma > slow_sma, 1.0, -1.0)

sweep = mt.sweep(
    data,
    create_signal,
    params={
        "fast": [5, 10, 20],
        "slow": [20, 50, 100]
    }
)

best = sweep.best()
print(f"Best params: {best.params}")
print(f"Best Sharpe: {best.result.sharpe:.2f}")
```

## Compare Strategies

```python
signal_a = np.where(np.arange(len(data["close"])) % 2 == 0, 1.0, -1.0)
signal_b = np.where(np.arange(len(data["close"])) % 3 == 0, 1.0, -1.0)
results_a = mt.backtest(data, signal_a)
results_b = mt.backtest(data, signal_b)

comparison = mt.compare(
    [results_a, results_b],
    names=["Strategy A", "Strategy B"]
)
print(comparison)
```

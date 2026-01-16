# Backtest API

Functions for running backtests.

!!! warning "Disclaimer"
    **Past performance does not guarantee future results.** Backtest results are hypothetical and do not represent actual trading. Real trading involves risk of loss. All performance metrics shown are based on historical simulations with assumptions that may not reflect actual market conditions.

## backtest

Run a backtest on historical data.

```python
def backtest(
    data: Union[dict, DataFrame],
    signal: Union[np.ndarray, str],
    commission: float = 0.001,
    slippage: float = 0.001,
    size: float = 0.10,
    cash: float = 100_000,
    allow_short: bool = True,
    borrow_cost: float = 0.03,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    strategy: Optional[str] = None
) -> BacktestResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `dict` or `DataFrame` | required | OHLCV data from `load()` or DataFrame |
| `signal` | `np.ndarray` or `str` | required | Signal array or strategy name |
| `commission` | `float` | `0.001` | Commission per trade (0.001 = 0.1%) |
| `slippage` | `float` | `0.001` | Slippage per trade (0.001 = 0.1%) |
| `size` | `float` | `0.10` | Position size (0.10 = 10% of equity) |
| `cash` | `float` | `100_000` | Starting capital |
| `allow_short` | `bool` | `True` | Allow short positions |
| `borrow_cost` | `float` | `0.03` | Annual borrow cost for shorts (3%) |
| `stop_loss` | `float` | `None` | Stop loss percentage (e.g., 0.02 = 2%) |
| `take_profit` | `float` | `None` | Take profit percentage (e.g., 0.05 = 5%) |
| `strategy` | `str` | `None` | Built-in strategy name (overrides signal) |

**Returns:**

[`BacktestResult`](results.md#backtestresult) object.

**Signal Values:**

| Value | Meaning |
|-------|---------|
| `1` or `> 0` | Long position |
| `-1` or `< 0` | Short position |
| `0` | Flat/close position |

**Example:**

```python
import mantis as mt
import numpy as np

data = mt.load_sample("AAPL")
signal = np.random.choice([-1, 0, 1], size=len(data))

results = mt.backtest(
    data,
    signal,
    commission=0.001,
    slippage=0.001,
    size=0.10,
    cash=100_000
)

print(f"Sharpe: {results.sharpe:.2f}")
print(f"Return: {results.total_return:.1%}")
```

---

## Backtest (Class)

Fluent API for building backtests.

```python
class Backtest:
    def __init__(self, data, signal)
    def commission(self, value: float) -> Backtest
    def slippage(self, value: float) -> Backtest
    def size(self, value: float) -> Backtest
    def cash(self, value: float) -> Backtest
    def stop_loss(self, value: float) -> Backtest
    def take_profit(self, value: float) -> Backtest
    def allow_short(self, value: bool = True) -> Backtest
    def borrow_cost(self, value: float) -> Backtest
    def run(self) -> BacktestResult
```

**Example:**

```python
import mantis as mt

results = (
    mt.Backtest(data, signal)
    .commission(0.0005)
    .slippage(0.0005)
    .size(0.15)
    .cash(50_000)
    .stop_loss(0.02)
    .take_profit(0.05)
    .run()
)
```

---

## Built-in Strategies

Use built-in strategies instead of signals:

```python
results = mt.backtest(data, strategy="sma-crossover")
```

**Available Strategies:**

| Strategy | Description |
|----------|-------------|
| `sma-crossover` | Simple moving average crossover |
| `momentum` | Momentum-based trading |
| `mean-reversion` | Mean reversion strategy |
| `rsi` | RSI-based trading |
| `macd` | MACD-based trading |
| `breakout` | Price breakout strategy |

---

## sweep

Run parameter sweep across multiple configurations.

```python
def sweep(
    data: dict,
    signal_fn: Callable,
    params: Dict[str, List],
    n_jobs: int = -1
) -> SweepResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `dict` | required | OHLCV data |
| `signal_fn` | `Callable` | required | Function that takes params and returns signal |
| `params` | `Dict[str, List]` | required | Parameter grid to sweep |
| `n_jobs` | `int` | `-1` | Number of parallel jobs (-1 = all cores) |

**Returns:**

`SweepResult` with `best_params`, `best_result`, and `all_results`.

**Example:**

```python
import mantis as mt

def create_signal(fast, slow):
    # Create signal based on parameters
    return signal

sweep = mt.sweep(
    data,
    create_signal,
    params={
        "fast": [5, 10, 20],
        "slow": [20, 50, 100]
    }
)

print(f"Best params: {sweep.best_params}")
print(f"Best Sharpe: {sweep.best_result.sharpe:.2f}")
```

---

## signal_check

Validate signal before backtesting.

```python
def signal_check(
    data: dict,
    signal: np.ndarray
) -> List[str]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `dict` | required | OHLCV data |
| `signal` | `np.ndarray` | required | Signal to validate |

**Returns:**

List of warning messages (empty if signal is valid).

**Checks Performed:**

| Check | Warning |
|-------|---------|
| Length mismatch | "Signal length X doesn't match data length Y" |
| NaN values | "Signal contains N NaN values" |
| Inf values | "Signal contains infinity values" |
| Out of range | "Signal values outside [-1, 1] range" |
| All zeros | "Signal is all zeros (no trades)" |
| All same | "Signal is constant (no trades)" |

**Example:**

```python
import mantis as mt

issues = mt.signal_check(data, signal)

if issues:
    for issue in issues:
        print(f"Warning: {issue}")
else:
    print("Signal is valid")
    results = mt.backtest(data, signal)
```

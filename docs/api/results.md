# Results API

Classes and functions for working with backtest results.

!!! warning "Disclaimer"
    **Past performance does not guarantee future results.** Backtest results are hypothetical and do not represent actual trading. Real trading involves risk of loss. All performance metrics shown are based on historical simulations with assumptions that may not reflect actual market conditions.

## BacktestResult

Results from running a backtest.

```python
class BacktestResult:
    # Performance metrics
    total_return: float        # Total return percentage
    sharpe: float              # Sharpe ratio
    sortino: float             # Sortino ratio
    calmar: float              # Calmar ratio
    max_drawdown: float        # Maximum drawdown (negative)
    win_rate: float            # Winning trade percentage
    profit_factor: float       # Gross profit / gross loss

    # Benchmark comparison
    benchmark_return: float    # Buy-and-hold return
    excess_return: float       # Strategy - benchmark

    # Advanced metrics
    deflated_sharpe: float     # Deflated Sharpe Ratio
    psr: float                 # Probabilistic Sharpe Ratio

    # Data
    equity_curve: np.ndarray   # Equity values
    trades: List[Trade]        # Trade list

    # Methods
    def metrics(self) -> dict
    def warnings(self) -> List[str]
    def validate(self, folds=12, train_ratio=0.75, anchored=True) -> ValidationResult
    def plot(self, **kwargs) -> Optional[Figure]
    def save(self, path: str) -> None
    def report(self, path: str) -> None
```

### Properties

**Performance Metrics:**

| Property | Type | Description |
|----------|------|-------------|
| `total_return` | `float` | Total return (e.g., 0.15 = 15%) |
| `sharpe` | `float` | Annualized Sharpe ratio |
| `sortino` | `float` | Sortino ratio (downside risk only) |
| `calmar` | `float` | Calmar ratio (return / max drawdown) |
| `max_drawdown` | `float` | Maximum drawdown (e.g., -0.18 = -18%) |
| `win_rate` | `float` | Winning trades / total trades |
| `profit_factor` | `float` | Gross profit / abs(gross loss) |

**Benchmark:**

| Property | Type | Description |
|----------|------|-------------|
| `benchmark_return` | `float` | Buy-and-hold return |
| `excess_return` | `float` | Strategy return - benchmark |

**Advanced:**

| Property | Type | Description |
|----------|------|-------------|
| `deflated_sharpe` | `float` | Sharpe adjusted for multiple testing |
| `psr` | `float` | Probability that Sharpe > 0 |

**Data:**

| Property | Type | Description |
|----------|------|-------------|
| `equity_curve` | `np.ndarray` | Equity value at each bar |
| `trades` | `List[Trade]` | List of all trades |

### Methods

#### metrics()

Get all metrics as a dictionary.

```python
def metrics(self) -> dict
```

**Example:**

```python
m = results.metrics()
print(f"Sharpe: {m['sharpe']:.2f}")
print(f"Max DD: {m['max_drawdown']:.1%}")
```

#### warnings()

Get auto-generated warnings about suspicious results.

```python
def warnings(self) -> List[str]
```

**Example:**

```python
for w in results.warnings():
    print(f"Warning: {w}")
```

#### validate()

Run walk-forward validation.

```python
def validate(
    self,
    folds: int = 12,
    train_ratio: float = 0.75,
    anchored: bool = True
) -> ValidationResult
```

See [ValidationResult](validation.md#validationresult) for details.

#### plot()

Visualize results.

```python
def plot(
    self,
    drawdown: bool = True,
    show_drawdown: bool = True,
    width: int = 40,
    save: Optional[str] = None
) -> Optional[Figure]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drawdown` | `bool` | `True` | Include drawdown subplot |
| `show_drawdown` | `bool` | `True` | Show drawdown in Plotly |
| `width` | `int` | `40` | Width for ASCII sparkline |
| `save` | `str` | `None` | Path to save HTML file |

**Returns:**

In Jupyter with Plotly: Returns Plotly `Figure` object.
In terminal: Prints ASCII sparkline, returns `None`.

#### save()

Save results to JSON.

```python
def save(self, path: str) -> None
```

**Example:**

```python
results.save("experiment_001.json")
```

#### report()

Generate HTML report with charts.

```python
def report(self, path: str) -> None
```

**Example:**

```python
results.report("backtest_report.html")
```

---

## Trade

Individual trade information.

```python
class Trade:
    symbol: str           # Symbol traded
    direction: str        # "long" or "short"
    entry_date: datetime  # Entry timestamp
    exit_date: datetime   # Exit timestamp
    entry_price: float    # Entry price
    exit_price: float     # Exit price
    shares: float         # Number of shares
    pnl: float            # Profit/loss in dollars
    pnl_pct: float        # Profit/loss percentage
    bars_held: int        # Number of bars held
```

**Example:**

```python
for trade in results.trades[:5]:
    print(f"{trade.direction}: {trade.pnl:+.2f} ({trade.pnl_pct:+.1%})")
```

---

## compare

Compare multiple backtest results.

```python
def compare(
    results: List[BacktestResult],
    names: Optional[List[str]] = None
) -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `List[BacktestResult]` | required | Results to compare |
| `names` | `List[str]` | `None` | Display names |

**Returns:**

Formatted comparison table as string.

**Example:**

```python
import mantis as mt
import numpy as np

data = mt.load_sample("AAPL")
signal_a = np.ones(len(data["close"]))
signal_b = -signal_a
signal_c = np.where(np.arange(len(data["close"])) % 2 == 0, 1.0, -1.0)

results_a = mt.backtest(data, signal_a)
results_b = mt.backtest(data, signal_b)
results_c = mt.backtest(data, signal_c)

comparison = mt.compare(
    [results_a, results_b, results_c],
    names=["LSTM", "Transformer", "Baseline"]
)
print(comparison)
```

---

## load_results

Load saved results from JSON.

```python
def load_results(path: str) -> BacktestResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | Path to JSON file |

**Returns:**

[`BacktestResult`](#backtestresult) object.

**Example:**

```python
# Save
results.save("experiment_001.json")

# Load later
loaded = mt.load_results("experiment_001.json")
print(f"Sharpe: {loaded.sharpe:.2f}")
```

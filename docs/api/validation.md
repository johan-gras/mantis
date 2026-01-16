# Validation API

Functions and classes for validating backtests.

## validate

Run walk-forward validation on a backtest.

```python
def validate(
    data: dict,
    signal: np.ndarray,
    folds: int = 12,
    train_ratio: float = 0.75,
    anchored: bool = True,
    **backtest_kwargs
) -> ValidationResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `dict` | required | OHLCV data |
| `signal` | `np.ndarray` | required | Signal array |
| `folds` | `int` | `12` | Number of walk-forward folds |
| `train_ratio` | `float` | `0.75` | Training data ratio (75% train) |
| `anchored` | `bool` | `True` | Use expanding window |
| `**backtest_kwargs` | | | Additional backtest options |

**Returns:**

[`ValidationResult`](#validationresult) object.

**Example:**

```python
import mantis as mt

validation = mt.validate(
    data,
    signal,
    folds=12,
    train_ratio=0.75,
    anchored=True
)

print(f"OOS Sharpe: {validation.oos_sharpe:.2f}")
print(f"Verdict: {validation.verdict}")
```

---

## ValidationResult

Results from walk-forward validation.

```python
class ValidationResult:
    # Aggregate metrics
    is_sharpe: float          # In-sample Sharpe (average)
    oos_sharpe: float         # Out-of-sample Sharpe (average)
    efficiency: float         # OOS/IS ratio
    verdict: str              # "robust" | "borderline" | "likely_overfit"
    fold_count: int           # Number of folds

    # Methods
    def is_robust(self) -> bool
    def fold_details(self) -> List[FoldDetail]
    def plot(self, **kwargs) -> Optional[Figure]
    def report(self, path: str) -> None
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_sharpe` | `float` | Average in-sample Sharpe across folds |
| `oos_sharpe` | `float` | Average out-of-sample Sharpe across folds |
| `efficiency` | `float` | OOS/IS ratio (e.g., 0.85 = 85%) |
| `verdict` | `str` | Classification of robustness |
| `fold_count` | `int` | Number of folds used |

### Verdict Interpretation

| Verdict | OOS/IS Ratio | Meaning |
|---------|--------------|---------|
| `"robust"` | > 80% | Strategy holds up well out-of-sample |
| `"borderline"` | 60-80% | Some degradation, proceed with caution |
| `"likely_overfit"` | < 60% | Significant degradation, likely overfit |

### Methods

#### is_robust()

Check if strategy passed validation.

```python
def is_robust(self) -> bool
```

**Returns:**

`True` if verdict is "robust", `False` otherwise.

**Example:**

```python
if validation.is_robust():
    print("Strategy is robust!")
else:
    print(f"Caution: {validation.verdict}")
```

#### fold_details()

Get per-fold metrics.

```python
def fold_details(self) -> List[FoldDetail]
```

**Returns:**

List of [`FoldDetail`](#folddetail) objects.

**Example:**

```python
for fold in validation.fold_details():
    print(f"IS: {fold.is_sharpe:.2f}, OOS: {fold.oos_sharpe:.2f}")
```

#### plot()

Visualize fold performance.

```python
def plot(
    self,
    width: int = 20,
    save: Optional[str] = None
) -> Optional[Figure]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | `int` | `20` | Width for ASCII chart |
| `save` | `str` | `None` | Path to save HTML file |

In Jupyter: Shows Plotly bar chart comparing IS vs OOS per fold.
In terminal: Prints ASCII chart.

#### report()

Generate HTML validation report.

```python
def report(self, path: str) -> None
```

**Example:**

```python
validation.report("validation_report.html")
```

---

## FoldDetail

Per-fold validation metrics.

```python
class FoldDetail:
    is_sharpe: float      # In-sample Sharpe for this fold
    oos_sharpe: float     # Out-of-sample Sharpe for this fold
    is_return: float      # In-sample return
    oos_return: float     # Out-of-sample return
    efficiency: float     # OOS/IS ratio for this fold
    is_bars: int          # Number of in-sample bars
    oos_bars: int         # Number of out-of-sample bars
```

**Example:**

```python
for i, fold in enumerate(validation.fold_details()):
    print(f"Fold {i+1}:")
    print(f"  IS Sharpe: {fold.is_sharpe:.2f}")
    print(f"  OOS Sharpe: {fold.oos_sharpe:.2f}")
    print(f"  Efficiency: {fold.efficiency:.0%}")
```

---

## Using BacktestResult.validate()

You can also validate directly from a result:

```python
results = mt.backtest(data, signal)
validation = results.validate()  # Uses stored data and signal
```

This is equivalent to:

```python
validation = mt.validate(data, signal)
```

---

## Anchored vs Rolling

### Anchored Windows (Default)

Training window starts at the beginning and expands each fold:

```
Fold 1: [=====Train=====][OOS]
Fold 2: [======Train======][OOS]
Fold 3: [=======Train=======][OOS]
```

- More realistic simulation of live trading
- More training data in later folds
- Recommended for most use cases

### Rolling Windows

Fixed-size training window moves forward:

```
Fold 1: [=====Train=====][OOS]
Fold 2:      [=====Train=====][OOS]
Fold 3:           [=====Train=====][OOS]
```

- Better for detecting regime changes
- Consistent training set size
- Use when market conditions vary significantly

```python
# Anchored (default)
validation = results.validate(anchored=True)

# Rolling
validation = results.validate(anchored=False)
```

---

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| < 2 years data | Warning about limited data |
| 0 trades in fold | Fold marked "no trades", excluded |
| All folds negative | verdict = "likely_overfit" |
| Perfect Sharpe (inf) | Error |

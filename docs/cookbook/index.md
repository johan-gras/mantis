# Cookbook

Copy-paste recipes for common backtesting tasks.

## Quick Links

| Task | Recipe |
|------|--------|
| Load different file formats | [Loading Data](loading-data.md) |
| Run a simple backtest | [Running Backtests](backtests.md) |
| Validate your strategy | [Validation](validation.md) |
| Backtest multiple symbols | [Multi-Symbol](multi-symbol.md) |
| Control position sizes | [Position Sizing](position-sizing.md) |
| Create charts and reports | [Visualization](visualization.md) |

## Most Common Tasks

### Compare Strategy to Buy-and-Hold

```python
results = mt.backtest(data, signal)
print(f"Strategy:      {results.total_return:.1%}")
print(f"Buy-and-hold:  {results.benchmark_return:.1%}")
print(f"Excess return: {results.excess_return:+.1%}")
```

### Quick Walk-Forward Validation

```python
results = mt.backtest(data, signal)
validation = results.validate()  # 12 folds by default
print(f"OOS/IS ratio: {validation.efficiency:.0%}")
print(f"Verdict: {validation.verdict}")
```

### Check for Suspicious Results

```python
results = mt.backtest(data, signal)
warnings = results.warnings()
if warnings:
    for w in warnings:
        print(f"Warning: {w}")
```

Common warnings:

- Sharpe > 3: "Suspiciously high, verify data"
- Win rate > 80%: "Check for lookahead bias"
- Max drawdown < 5%: "Verify execution logic"
- Trades < 30: "Limited statistical significance"

### Export Results

```python
# Save metrics to JSON
results.save("experiment_001.json")

# Generate HTML report with charts
results.report("experiment_001.html")

# Load saved results
loaded = mt.load_results("experiment_001.json")
```

### Parameter Sweep

```python
def signal_fn(threshold):
    return np.where(predictions > threshold, 1, -1)

sweep = mt.sweep(
    data,
    signal_fn,
    params={"threshold": [0.3, 0.4, 0.5, 0.6, 0.7]}
)

# Best parameters
print(sweep.best_params)
print(sweep.best_result.sharpe)
```

### Compare Multiple Strategies

```python
results_a = mt.backtest(data, signal_a)
results_b = mt.backtest(data, signal_b)
results_c = mt.backtest(data, signal_c)

comparison = mt.compare(
    [results_a, results_b, results_c],
    names=["LSTM", "Transformer", "Baseline"]
)
print(comparison)
```

# Validation

Validate your strategy to catch overfitting before it costs you money.

## Walk-Forward Validation

The most important validation technique:

```python
import mantis as mt

data = mt.load_sample("AAPL")
results = mt.backtest(data, signal)

# Run walk-forward validation
validation = results.validate()

# Key metrics
print(f"In-sample Sharpe:  {validation.is_sharpe:.2f}")
print(f"Out-of-sample:     {validation.oos_sharpe:.2f}")
print(f"Efficiency:        {validation.efficiency:.0%}")
print(f"Verdict:           {validation.verdict}")
```

## Understanding the Verdict

| Verdict | OOS/IS Ratio | Meaning |
|---------|--------------|---------|
| **robust** | > 80% | Strategy holds up out-of-sample |
| **borderline** | 60-80% | Some degradation, use caution |
| **likely_overfit** | < 60% | Significant degradation, likely overfit |

## Customize Validation

```python
validation = results.validate(
    folds=20,           # More folds (default: 12)
    train_ratio=0.80,   # 80% train (default: 75%)
    anchored=True,      # Expanding window (default: True)
)
```

### Anchored vs Rolling

- **Anchored (recommended)**: Training window expands each fold. More realistic.
- **Rolling**: Fixed-size training window. Better for detecting regime changes.

```python
# Anchored: realistic simulation
validation = results.validate(anchored=True)

# Rolling: detect regime changes
validation = results.validate(anchored=False)
```

## Fold Details

Examine each fold individually:

```python
validation = results.validate()

for fold in validation.fold_details():
    print(f"Fold: IS={fold.is_sharpe:.2f}, OOS={fold.oos_sharpe:.2f}, "
          f"Efficiency={fold.efficiency:.0%}")
```

## Auto-Warnings

Mantis automatically flags suspicious results:

```python
results = mt.backtest(data, signal)
warnings = results.warnings()

for w in warnings:
    print(f"Warning: {w}")
```

| Warning | Trigger | Concern |
|---------|---------|---------|
| High Sharpe | > 3.0 | Verify data integrity |
| High Win Rate | > 80% | Check for lookahead bias |
| Low Drawdown | < 5% | Verify execution logic |
| Few Trades | < 30 | Limited statistical significance |
| High Profit Factor | > 5.0 | Verify trade logic |

## Deflated Sharpe Ratio

Penalizes for multiple testing:

```python
results = mt.backtest(data, signal)

print(f"Sharpe:          {results.sharpe:.2f}")
print(f"Deflated Sharpe: {results.deflated_sharpe:.2f}")
print(f"PSR:             {results.psr:.0%}")
```

| Metric | Interpretation |
|--------|---------------|
| **Deflated Sharpe > 0** | Significant after accounting for trials |
| **Deflated Sharpe -0.5 to 0** | Borderline significance |
| **Deflated Sharpe < -0.5** | Likely spurious |
| **PSR > 95%** | High confidence Sharpe is real |
| **PSR 70-95%** | Moderate confidence |
| **PSR < 70%** | Low confidence |

## Cost Sensitivity

Test how robust your strategy is to higher costs:

```python
# Using CLI
# mantis cost-sensitivity -d data.csv --strategy sma-crossover

# Results show degradation at 1x, 2x, 5x, 10x costs
```

## Parameter Sensitivity

Test how sensitive results are to parameter changes:

```python
# Using CLI
# mantis sensitivity -d data.csv --strategy sma-crossover

# Shows Sharpe across parameter range
# Flags "cliffs" where small changes cause large drops
```

## Validation Report

Generate a comprehensive HTML report:

```python
validation = results.validate()
validation.report("validation_report.html")
```

Report includes:

- Fold-by-fold performance chart
- IS vs OOS comparison
- Efficiency metrics
- Verdict with explanation

## Visualize Validation

```python
# In Jupyter: Plotly bar chart
# In terminal: ASCII chart
validation.plot()
```

## Best Practices

1. **Always validate** - A good in-sample Sharpe means nothing without OOS testing

2. **Use anchored windows** - More realistic simulation of live trading

3. **Check the verdict** - If `likely_overfit`, don't trade it

4. **Examine fold variance** - High variance across folds suggests regime sensitivity

5. **Test cost sensitivity** - If profit disappears at 2x costs, strategy is fragile

6. **Mind your trials** - More parameter combinations tested = lower deflated Sharpe

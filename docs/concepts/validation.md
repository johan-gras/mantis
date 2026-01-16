# Validation Philosophy

Why validation matters and how Mantis approaches it.

## The Core Problem

Every backtest is a form of data mining.

When you:
- Try different parameters
- Test different models
- Adjust based on results

...you're implicitly fitting to historical data.

**The question isn't "Did it work in the past?"**

**The question is "Will it work in the future?"**

## Overfitting: The Silent Killer

Overfitting happens when a strategy captures noise instead of signal.

Signs of overfitting:
- Sharpe > 3 (suspiciously high)
- Perfect or near-perfect win rate
- Strategy only works on specific date ranges
- Small parameter changes cause large performance swings

## Walk-Forward Validation

The core validation technique in Mantis.

```
Historical Data: [====================================]

Fold 1:  [====Train====][Test]
         Optimize on past, test on "future"

Fold 2:  [=====Train=====][Test]
         Repeat with more history

Fold 3:  [======Train======][Test]
         And again...
```

**Why it works:**

Each test period is truly out-of-sample. The strategy has never "seen" that data.

## OOS Degradation

The key metric: How much does performance drop out-of-sample?

```
OOS/IS Ratio = Out-of-Sample Sharpe / In-Sample Sharpe
```

| Ratio | Interpretation |
|-------|----------------|
| > 80% | Excellent, strategy generalizes well |
| 60-80% | Acceptable, some degradation |
| 40-60% | Concerning, significant overfitting |
| < 40% | Red flag, likely won't work live |

## The Verdict System

Mantis classifies strategies automatically:

### Robust

```python
validation.verdict == "robust"
```

- OOS/IS > 80%
- Positive OOS returns
- Good efficiency across folds

**Action:** Strategy is likely to work in production.

### Borderline

```python
validation.verdict == "borderline"
```

- OOS/IS 60-80%
- Some inconsistency across folds

**Action:** Proceed with caution. Consider paper trading first.

### Likely Overfit

```python
validation.verdict == "likely_overfit"
```

- OOS/IS < 60%
- Or negative OOS returns
- Or high variance across folds

**Action:** Do not trade this strategy. Re-examine your approach.

## Auto-Warnings

Mantis flags suspicious results:

| Warning | Threshold | Concern |
|---------|-----------|---------|
| High Sharpe | > 3.0 | Data quality, lookahead |
| High Win Rate | > 80% | Lookahead bias |
| Low Drawdown | < 5% | Execution logic |
| Few Trades | < 30 | Statistical significance |
| High Profit Factor | > 5.0 | Trade logic |

These aren't proof of problems, but warrant investigation.

## Deflated Sharpe Ratio

Accounts for multiple testing.

```python
# If you tested 100 parameter combinations:
# Some will look good by chance

results.deflated_sharpe  # Penalized for trials
```

| DSR | Interpretation |
|-----|----------------|
| > 0 | Significant after accounting for trials |
| -0.5 to 0 | Borderline |
| < -0.5 | Likely spurious |

## Probabilistic Sharpe Ratio

Probability that true Sharpe > 0.

```python
results.psr  # e.g., 0.92 = 92% confidence
```

| PSR | Interpretation |
|-----|----------------|
| > 95% | High confidence |
| 70-95% | Moderate confidence |
| < 70% | Low confidence |

## Parameter Sensitivity

A robust strategy shouldn't be fragile:

```
Parameter | Sharpe
----------|-------
fast=18   | 1.8
fast=19   | 1.9
fast=20   | 2.0    ← Selected
fast=21   | 0.3    ← Cliff!
fast=22   | 0.2
```

This strategy is fragile. Small changes cause large drops.

## Cost Sensitivity

Test at higher costs:

```
Costs | Sharpe
------|-------
1x    | 1.5
2x    | 1.2
5x    | 0.6
10x   | 0.1
```

If profit vanishes at 2x costs, the edge is thin.

## Best Practices

### 1. Always validate

```python
results = mt.backtest(data, signal)
validation = results.validate()  # Don't skip this
```

### 2. Use enough folds

```python
validation = results.validate(folds=12)  # Default
```

More folds = more reliable estimate.

### 3. Check fold variance

```python
for fold in validation.fold_details():
    print(f"OOS Sharpe: {fold.oos_sharpe:.2f}")
```

High variance across folds = regime sensitivity.

### 4. Test parameter stability

Don't just pick the best. Check neighbors.

### 5. Test cost robustness

If 2x costs kills profit, reconsider.

### 6. Be skeptical of great results

If it looks too good to be true, it probably is.

## The Validation Mindset

> "I'm trying to find reasons why this strategy WON'T work."

Not:

> "I'm trying to prove this strategy works."

Mantis is designed for the skeptical researcher who wants honest answers, not flattering ones.

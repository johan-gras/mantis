# Mantis

**Fast, correct backtesting for DL researchers.**

Get from `pip install` to "I trust this result" in 5 minutes.

---

## Why Mantis?

| Problem | Mantis Solution |
|---------|-----------------|
| Quick pandas backtest is wrong | Correct by default (costs, no lookahead) |
| Existing libraries are slow | Rust backend, sub-100ms backtests |
| Building your own takes months | 5 lines to run, 10 lines to validate |
| Surprising results go unquestioned | Auto-warnings for suspicious metrics |

## Quick Example

```python
import mantis as mt
import numpy as np

# Load sample data (bundled, works offline)
data = mt.load_sample("AAPL")

# Your signal (replace with your model's predictions)
signal = np.random.choice([-1, 0, 1], size=len(data))

# Run backtest
results = mt.backtest(data, signal)

# See what you got
print(results)
# ┌────────────────────────────────────────┐
# │ Sharpe: 0.42  MaxDD: -18.3%            │
# │ Return: 12.4% Trades: 847              │
# │ ▁▂▃▄▅▆▅▇█▇▆▅▄▃▂▁▂▃▄▅                  │
# └────────────────────────────────────────┘

# Is it real or overfit?
validation = results.validate()
print(validation.verdict)  # "robust" | "borderline" | "likely_overfit"
```

## Installation

```bash
pip install mantis-bt
```

- **No Rust toolchain required** - pre-built wheels for all platforms
- **Python 3.8+** - Linux, macOS, Windows
- **Dependencies**: numpy only (pandas/polars optional)

## What Makes Mantis Different

### Correct by Default

- **0.1% commission + 0.1% slippage** - real trading costs
- **Fills at next bar open** - no lookahead bias
- **Auto-warnings** - flags suspicious results

### Validation First-Class

- **Walk-forward validation** - tests out-of-sample performance
- **OOS degradation** - measures strategy robustness
- **Deflated Sharpe** - penalizes for multiple testing

### ML-Native

- **numpy arrays** - direct from your model
- **pandas/polars** - native DataFrame support
- **No callbacks** - signals, not strategy classes

### Fast

- **< 100ms** for 10 years daily data
- **< 30s** for 1000 parameter combinations
- **Rust backend** - think-test-iterate speed

---

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Quick Start**

    ---

    Get running in 5 minutes with bundled sample data

    [:octicons-arrow-right-24: Get started](quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Cookbook**

    ---

    Copy-paste recipes for common tasks

    [:octicons-arrow-right-24: Browse recipes](cookbook/index.md)

-   :material-cog:{ .lg .middle } **Concepts**

    ---

    Understand how Mantis works under the hood

    [:octicons-arrow-right-24: Learn more](concepts/how-it-works.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Full documentation for all functions

    [:octicons-arrow-right-24: API docs](api/index.md)

</div>

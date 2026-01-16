# API Reference

Complete documentation for all Mantis functions and classes.

## Quick Reference

| Function | Description |
|----------|-------------|
| [`mt.load()`](data.md#load) | Load data from CSV or Parquet |
| [`mt.load_sample()`](data.md#load_sample) | Load bundled sample data |
| [`mt.load_multi()`](data.md#load_multi) | Load multiple symbol files |
| [`mt.load_dir()`](data.md#load_dir) | Load all files from directory |
| [`mt.backtest()`](backtest.md#backtest) | Run a backtest |
| [`mt.Backtest()`](backtest.md#backtest-class) | Fluent API for backtesting |
| [`mt.validate()`](validation.md#validate) | Walk-forward validation |
| [`mt.sweep()`](backtest.md#sweep) | Parameter sweep |
| [`mt.compare()`](results.md#compare) | Compare multiple results |
| [`mt.signal_check()`](backtest.md#signal_check) | Validate signal |
| [`mt.list_samples()`](data.md#list_samples) | List available samples |
| [`mt.load_results()`](results.md#load_results) | Load saved results |

## Classes

| Class | Description |
|-------|-------------|
| [`BacktestResult`](results.md#backtestresult) | Results from a backtest |
| [`ValidationResult`](validation.md#validationresult) | Results from validation |
| [`FoldDetail`](validation.md#folddetail) | Per-fold validation metrics |
| [`Trade`](results.md#trade) | Individual trade information |

## Installation

```bash
# Basic installation
pip install mantis-bt

# With Jupyter/Plotly support
pip install mantis-bt[jupyter]
```

## Import

```python
import mantis as mt
```

## Type Hints

Mantis includes type stubs (`.pyi` files) for IDE autocomplete and type checking.

```python
# IDE will show all available methods
results = mt.backtest(data, signal)
results.  # <- autocomplete shows: sharpe, max_drawdown, plot(), etc.
```

## Default Values

```python
# These are the default configuration values
mt.backtest(
    data,
    signal,
    commission=0.001,      # 0.1%
    slippage=0.001,        # 0.1%
    size=0.10,             # 10% of equity
    cash=100_000,          # Starting capital
    allow_short=True,      # Allow short positions
    borrow_cost=0.03,      # 3% annual borrow cost for shorts
    fill_price="next_open" # Fill at next bar open
)
```

## Supported Data Formats

- **CSV**: Auto-detected columns, flexible date formats
- **Parquet**: Optimized for large datasets
- **pandas DataFrame**: Direct input
- **polars DataFrame**: Direct input

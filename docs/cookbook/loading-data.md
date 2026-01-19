# Loading Data

Mantis supports multiple data formats and auto-detects column names.

## Load from File

### CSV

```python
import mantis as mt

# Basic load
data = mt.load("data/samples/AAPL.csv")

# Access columns
print(data["open"][:5])     # First 5 opens
print(data["close"][-1])    # Last close
print(data["bars"])         # List of Bar objects
```

### Parquet

```python
# data = mt.load("data/samples/AAPL.parquet")
```

Parquet is faster for large files (10GB in < 5 seconds).

## Sample Data

Mantis bundles sample data that works offline:

```python
# 10 years of daily OHLCV
data = mt.load_sample("AAPL")
data = mt.load_sample("SPY")
data = mt.load_sample("BTC")

# List all available samples
print(mt.list_samples())  # ["AAPL", "SPY", "BTC"]
```

## Use pandas/polars DataFrames

### pandas

```python
import pandas as pd

df = pd.read_csv("data/samples/AAPL.csv")
results = mt.backtest(df, signal)
```

### polars

```python
try:
    import polars as pl
    df = pl.read_csv("data/samples/AAPL.csv")
    results = mt.backtest(df, signal)
except ImportError:
    pass
```

## Multiple Symbols

### Load Multiple Files

```python
# Provide paths
data = mt.load_multi({
    "AAPL": "data/samples/AAPL.csv",
    "SPY": "data/samples/SPY.csv",
    "BTC": "data/samples/BTC.csv"
})
```

### Load Directory

```python
# Load all CSVs from a directory
data = mt.load_dir("data/samples/*.csv")
# Symbol names derived from filenames
```

## Column Auto-Detection

Mantis auto-detects common column naming conventions:

| Column | Detected Names |
|--------|---------------|
| Open | `open`, `Open`, `OPEN`, `o` |
| High | `high`, `High`, `HIGH`, `h` |
| Low | `low`, `Low`, `LOW`, `l` |
| Close | `close`, `Close`, `CLOSE`, `c`, `adj_close`, `adjusted_close` |
| Volume | `volume`, `Volume`, `VOLUME`, `vol`, `v` |
| Date | `date`, `Date`, `DATE`, `timestamp`, `Timestamp`, `time`, `datetime` |

## Date Format Detection

Common formats are auto-detected:

- `2024-01-15` (ISO)
- `01/15/2024` (US)
- `15/01/2024` (EU)
- `1705276800` (Unix timestamp)
- `2024-01-15T09:30:00Z` (ISO 8601)

## Data Validation

Mantis validates data on load:

```python
data = mt.load("data/samples/AAPL.csv")
# Automatic checks:
# - Missing OHLCV columns → error
# - Duplicate timestamps → removed with warning
# - Negative prices → error
# - High < Low → error
# - Zero volume → warning
# - Future dates → error
```

## What's Returned

`mt.load()` returns a dictionary:

```python
data = mt.load("data/samples/AAPL.csv")

# numpy arrays
data["timestamp"]  # datetime64 array
data["open"]       # float64 array
data["high"]       # float64 array
data["low"]        # float64 array
data["close"]      # float64 array
data["volume"]     # float64 array

# Bar objects
data["bars"]       # List[Bar] for detailed access

# Metadata
len(data["bars"])  # Number of bars
```

# Data Loading API

Functions for loading and preparing data for backtesting.

## load

Load OHLCV data from a file.

```python
def load(
    path: str,
    backend: str = "auto"
) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | Path to CSV or Parquet file |
| `backend` | `str` | `"auto"` | Force `"pandas"` or `"polars"` backend |

**Returns:**

Dictionary with numpy arrays:

```python
{
    "timestamp": np.ndarray,  # datetime64
    "open": np.ndarray,       # float64
    "high": np.ndarray,       # float64
    "low": np.ndarray,        # float64
    "close": np.ndarray,      # float64
    "volume": np.ndarray,     # float64
    "bars": List[Bar]         # Bar objects
}
```

**Example:**

```python
import mantis as mt

# Load from CSV
data = mt.load("prices.csv")

# Load from Parquet
data = mt.load("prices.parquet")

# Access data
print(f"Bars: {len(data['bars'])}")
print(f"First close: {data['close'][0]}")
print(f"Last close: {data['close'][-1]}")
```

**Column Auto-Detection:**

| Column | Detected Names |
|--------|---------------|
| Open | `open`, `Open`, `OPEN`, `o` |
| High | `high`, `High`, `HIGH`, `h` |
| Low | `low`, `Low`, `LOW`, `l` |
| Close | `close`, `Close`, `CLOSE`, `c`, `adj_close` |
| Volume | `volume`, `Volume`, `VOLUME`, `vol`, `v` |

---

## load_sample

Load bundled sample data (works offline).

```python
def load_sample(name: str) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Sample name (case-insensitive) |

**Returns:**

Same dictionary format as `load()`.

**Available Samples:**

| Name | Description | Bars |
|------|-------------|------|
| `"AAPL"` | Apple daily OHLCV (2014-2024) | ~2609 |
| `"SPY"` | S&P 500 ETF daily (2014-2024) | ~2609 |
| `"BTC"` | Bitcoin daily (2014-2024) | ~3653 |

**Example:**

```python
import mantis as mt

# Load sample data
data = mt.load_sample("AAPL")

# Case-insensitive
data = mt.load_sample("aapl")
data = mt.load_sample("Aapl")
```

---

## list_samples

List available sample datasets.

```python
def list_samples() -> List[str]
```

**Returns:**

List of sample names.

**Example:**

```python
import mantis as mt

samples = mt.list_samples()
print(samples)  # ["AAPL", "SPY", "BTC"]
```

---

## load_multi

Load multiple symbol files.

```python
def load_multi(
    paths: Dict[str, str]
) -> Dict[str, dict]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `paths` | `Dict[str, str]` | required | Mapping of symbol to file path |

**Returns:**

Dictionary mapping symbol names to data dictionaries.

**Example:**

```python
import mantis as mt

data = mt.load_multi({
    "AAPL": "data/AAPL.csv",
    "GOOGL": "data/GOOGL.csv",
    "MSFT": "data/MSFT.csv"
})

# Access by symbol
print(f"AAPL bars: {len(data['AAPL']['bars'])}")
```

---

## load_dir

Load all files from a directory matching a pattern.

```python
def load_dir(
    path: str,
    pattern: str = "*.csv"
) -> Dict[str, dict]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | Directory path |
| `pattern` | `str` | `"*.csv"` | Glob pattern for files |

**Returns:**

Dictionary mapping filenames (without extension) to data dictionaries.

**Example:**

```python
import mantis as mt

# Load all CSVs
data = mt.load_dir("data/stocks/", pattern="*.csv")

# Load all Parquet files
data = mt.load_dir("data/stocks/", pattern="*.parquet")

# Symbols are derived from filenames
# data/stocks/AAPL.csv → data["AAPL"]
```

---

## Data Validation

Mantis validates data on load:

| Check | Behavior |
|-------|----------|
| Missing OHLCV columns | Error with suggestions |
| Duplicate timestamps | Removed, warning |
| Negative prices | Error |
| High < Low | Error |
| Open/Close outside range | Warning |
| Zero volume | Warning |
| Future dates | Error |
| Empty file | Error |

---

## DataFrame Input

You can pass pandas or polars DataFrames directly to `backtest()`:

```python
import pandas as pd
import mantis as mt

# pandas
df = pd.read_csv("prices.csv")
results = mt.backtest(df, signal)

# polars
import polars as pl
df = pl.read_csv("prices.csv")
results = mt.backtest(df, signal)
```

Mantis auto-detects DataFrame types and extracts OHLCV columns.

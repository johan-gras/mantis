"""Type stubs for Mantis backtesting engine."""

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray

__version__: str

class Bar:
    """OHLCV bar representing a single time period of market data."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __init__(
        self,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...

class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float
    commission: float
    slippage: float
    position_size: float
    allow_short: bool
    fractional_shares: bool
    stop_loss: Optional[float]
    take_profit: Optional[float]
    borrow_cost: float

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.001,
        position_size: float = 0.10,
        allow_short: bool = True,
        fractional_shares: bool = True,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        borrow_cost: float = 0.03,
    ) -> None: ...

class BacktestResult:
    """Results from a backtest run."""

    strategy_name: str
    symbols: List[str]
    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    trading_days: int

    @property
    def equity_curve(self) -> NDArray[np.float64]: ...
    @property
    def equity_timestamps(self) -> NDArray[np.int64]: ...
    @property
    def trades(self) -> List[Any]: ...
    def metrics(self) -> Dict[str, Any]: ...
    def summary(self) -> str: ...
    def warnings(self) -> List[str]: ...

class ValidationResult:
    """Results from walk-forward validation."""

    folds: int
    is_sharpe: float
    oos_sharpe: float
    oos_degradation: float
    verdict: str

    def fold_details(self) -> List[Dict[str, Any]]: ...
    def summary(self) -> str: ...

def load(
    path: str,
    date_column: Optional[str] = None,
    date_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load OHLCV data from a CSV or Parquet file.

    Args:
        path: Path to the data file (.csv or .parquet)
        date_column: Optional name of the date column
        date_format: Optional date format string (e.g., "%Y-%m-%d")

    Returns:
        Dictionary with 'bars' list and numpy arrays for each column.

    Example:
        >>> data = mt.load("AAPL.csv")
        >>> print(data['n_bars'])
        2520
    """
    ...

def load_multi(
    paths: Dict[str, str],
    date_column: Optional[str] = None,
    date_format: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load multiple symbols from a dictionary of paths.

    Args:
        paths: Dictionary mapping symbol names to file paths
        date_column: Optional name of the date column
        date_format: Optional date format string

    Returns:
        Dictionary mapping symbol names to data dictionaries.

    Example:
        >>> data = mt.load_multi({"AAPL": "AAPL.csv", "GOOGL": "GOOGL.csv"})
    """
    ...

def load_dir(
    pattern: str,
    date_column: Optional[str] = None,
    date_format: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all files matching a glob pattern from a directory.

    Args:
        pattern: Glob pattern (e.g., "data/*.csv")
        date_column: Optional name of the date column
        date_format: Optional date format string

    Returns:
        Dictionary mapping symbol names to data dictionaries.

    Example:
        >>> data = mt.load_dir("data/*.csv")
    """
    ...

def list_samples() -> List[str]:
    """
    List available bundled sample datasets.

    Returns a list of sample names that can be used with `load_sample()`.
    Sample data is bundled with the package and requires no external files.

    Returns:
        List of sample data identifiers (e.g., ["AAPL", "SPY", "BTC"])

    Example:
        >>> samples = mt.list_samples()
        >>> print(samples)
        ['AAPL', 'SPY', 'BTC']
    """
    ...

def load_sample(name: str) -> Dict[str, Any]:
    """
    Load bundled sample data by name.

    Sample data is embedded in the binary and requires no external files or internet.
    This is useful for quick demos, testing, and getting started without downloading data.

    Available samples:
        - "AAPL": Apple Inc. stock (10 years daily, 2014-2024, ~2600 bars)
        - "SPY": S&P 500 ETF (10 years daily, 2014-2024, ~2600 bars)
        - "BTC": Bitcoin (10 years daily including weekends, 2014-2024, ~3650 bars)

    Args:
        name: Sample data identifier (case-insensitive, e.g., "AAPL", "aapl")

    Returns:
        Dictionary with 'bars' list and numpy arrays for each column.

    Raises:
        ValueError: If the sample name is not recognized

    Example:
        >>> data = mt.load_sample("AAPL")
        >>> print(data['n_bars'])
        2609
        >>> print(data['close'][-1])
        212.45
    """
    ...

def backtest(
    data: Union[Dict[str, Any], str],
    signal: Optional[NDArray[np.float64]] = None,
    strategy: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    config: Optional[BacktestConfig] = None,
    commission: float = 0.001,
    slippage: float = 0.001,
    size: float = 0.10,
    cash: float = 100_000.0,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    allow_short: bool = True,
    borrow_cost: float = 0.03,
) -> BacktestResult:
    """
    Run a backtest on historical data with a signal array.

    Args:
        data: Data dictionary from load() or path to CSV/Parquet file
        signal: numpy array of signals (1=long, -1=short, 0=flat)
        strategy: Name of built-in strategy if signal is None
        strategy_params: Dictionary of strategy parameters
        config: Optional BacktestConfig object
        commission: Commission rate (default 0.001 = 0.1%)
        slippage: Slippage rate (default 0.001 = 0.1%)
        size: Position size as fraction of equity (default 0.10 = 10%)
        cash: Initial capital (default 100,000)
        stop_loss: Optional stop loss percentage
        take_profit: Optional take profit percentage
        allow_short: Whether to allow short positions
        borrow_cost: Annual borrow rate for shorts (default 3%)

    Returns:
        BacktestResult object with metrics, equity curve, and trades.

    Example:
        >>> data = mt.load("AAPL.csv")
        >>> signal = np.where(data['close'] > data['close'].mean(), 1, -1)
        >>> results = mt.backtest(data, signal)
        >>> print(results.sharpe)
        1.24
    """
    ...

def signal_check(
    data: Dict[str, Any],
    signal: NDArray[np.float64],
) -> Dict[str, Any]:
    """
    Check a signal for common issues before running a backtest.

    Args:
        data: Data dictionary from load()
        signal: numpy array of signals

    Returns:
        Dictionary with validation results and suggestions.

    Example:
        >>> check = mt.signal_check(data, signal)
        >>> print(check['status'])
        'passed'
    """
    ...

def compare(
    results: List[BacktestResult],
    names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare multiple backtest results side by side.

    Args:
        results: List of BacktestResult objects to compare
        names: Optional list of names for each result

    Returns:
        Dictionary with comparison metrics
    """
    ...

def sweep(
    data: Any,
    signal_fn: Callable[..., NDArray[np.float64]],
    params: Dict[str, List[Any]],
    n_jobs: int = -1,
    **backtest_kwargs: Any,
) -> Dict[str, BacktestResult]:
    """
    Run a parameter sweep, testing multiple parameter combinations.

    Args:
        data: Data dictionary from load() or file path
        signal_fn: Function that takes params and returns a signal array
        params: Dictionary of parameter names to lists of values
        n_jobs: Number of parallel jobs (-1 for all cores)
        **backtest_kwargs: Additional arguments passed to backtest()

    Returns:
        Dictionary with results for each parameter combination
    """
    ...

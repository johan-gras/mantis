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
    @property
    def deflated_sharpe(self) -> float:
        """
        Get the Deflated Sharpe Ratio.

        The Deflated Sharpe Ratio (DSR) adjusts the Sharpe ratio for the
        number of trials conducted during strategy development. With more
        parameter combinations tested, the probability of finding a high
        Sharpe ratio by chance increases. DSR accounts for this multiple
        testing bias.

        Returns a value between 0 and the raw Sharpe ratio. Lower values
        indicate higher probability that the observed Sharpe is due to
        overfitting rather than genuine alpha.
        """
        ...
    @property
    def psr(self) -> float:
        """
        Get the Probabilistic Sharpe Ratio.

        The Probabilistic Sharpe Ratio (PSR) represents the probability
        that the true Sharpe ratio is greater than a benchmark (default 0).
        It accounts for the length of the track record, skewness, and
        kurtosis of returns.

        Returns a value between 0 and 1:
        - > 0.95: High confidence the strategy is genuinely profitable
        - 0.80-0.95: Moderate confidence
        - < 0.80: Low confidence, results may be due to chance
        """
        ...
    def metrics(self) -> Dict[str, Any]: ...
    def summary(self) -> str: ...
    def warnings(self) -> List[str]: ...
    def plot(self, width: int = 40, show_drawdown: bool = True) -> Any:
        """
        Display a visualization of the equity curve.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly figure with equity curve and drawdown subplots.
        In terminal or without plotly, returns an ASCII sparkline string.

        Args:
            width: Width of the visualization (characters for ASCII, ignored for Plotly)
            show_drawdown: Whether to show drawdown subplot (Plotly only, default: True)

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.plot()  # Shows interactive chart in Jupyter
            >>> print(results.plot())  # ASCII sparkline in terminal
        """
        ...
    def save(self, path: str) -> None:
        """
        Save the backtest results to a JSON file.

        The file contains all metrics, equity curve, and trades.

        Args:
            path: Path to the output JSON file.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.save("experiment_042.json")
        """
        ...
    def report(self, path: str) -> None:
        """
        Generate a self-contained HTML report.

        The report includes:
        - Summary metrics table
        - Equity curve chart (SVG)
        - Drawdown chart (SVG)
        - Trade list with P&L

        Args:
            path: Path to the output HTML file.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.report("experiment_042.html")
        """
        ...
    def validate(
        self,
        folds: int = 12,
        train_ratio: float = 0.75,
        anchored: bool = True,
    ) -> "ValidationResult":
        """
        Run walk-forward validation on the backtest.

        This is the key method for detecting overfitting. It splits the data
        into multiple folds, trains on in-sample data, and tests on out-of-sample
        data to measure performance degradation.

        Args:
            folds: Number of walk-forward folds (default: 12)
            train_ratio: Fraction of each window used for in-sample (default: 0.75)
            anchored: Whether to use anchored (expanding) windows (default: True)

        Returns:
            ValidationResult with IS/OOS metrics and verdict.

        Raises:
            RuntimeError: If the backtest was not run with validation data
                (e.g., loaded from a file or run with a built-in strategy).

        Example:
            >>> results = mt.backtest(data, signal)
            >>> validation = results.validate()
            >>> print(validation.verdict)
            'borderline'
            >>> print(validation.oos_degradation)
            0.71
        """
        ...

class FoldDetail:
    """Details for a single fold in walk-forward validation."""

    fold: int
    is_sharpe: float
    oos_sharpe: float
    is_return: float
    oos_return: float
    efficiency: float
    is_bars: int
    oos_bars: int

class ValidationResult:
    """Results from walk-forward validation."""

    folds: int
    is_sharpe: float
    oos_sharpe: float
    oos_degradation: float
    verdict: str
    avg_is_return: float
    avg_oos_return: float
    efficiency_ratio: float
    parameter_stability: float

    def fold_details(self) -> List[FoldDetail]: ...
    def is_robust(self) -> bool: ...
    def summary(self) -> str: ...
    def plot(self, width: int = 20) -> Any:
        """
        Display a visualization of fold-by-fold performance.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly figure with in-sample vs out-of-sample comparison.
        In terminal or without plotly, returns an ASCII bar chart string.

        Args:
            width: Width of ASCII bars (ignored for Plotly, default: 20)

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.

        Example:
            >>> validation = mt.validate(data, signal)
            >>> validation.plot()  # Shows interactive chart in Jupyter
            >>> print(validation.plot())  # ASCII chart in terminal
        """
        ...
    def report(self, path: str) -> None:
        """
        Generate a self-contained HTML report for the validation results.

        The report includes:
        - Summary metrics (folds, window type, IS ratio)
        - Verdict classification with color coding
        - Performance metrics (IS/OOS Sharpe, returns, efficiency)
        - Fold-by-fold results table
        - Bar chart comparing IS vs OOS performance

        Args:
            path: Path to the output HTML file.

        Example:
            >>> validation = mt.validate(data, signal)
            >>> validation.report("validation_report.html")
        """
        ...

def load(
    path: str,
    date_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load OHLCV data from a CSV or Parquet file.

    Args:
        path: Path to the data file (.csv or .parquet)
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
    date_format: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load multiple symbols from a dictionary of paths.

    Args:
        paths: Dictionary mapping symbol names to file paths
        date_format: Optional date format string

    Returns:
        Dictionary mapping symbol names to data dictionaries.

    Example:
        >>> data = mt.load_multi({"AAPL": "AAPL.csv", "GOOGL": "GOOGL.csv"})
    """
    ...

def load_dir(
    pattern: str,
    date_format: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all files matching a glob pattern from a directory.

    Args:
        pattern: Glob pattern (e.g., "data/*.csv")
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

try:
    import pandas as pd
    import polars as pl
    _DataFrame = Union[pd.DataFrame, pl.DataFrame]
except ImportError:
    _DataFrame = Any

def backtest(
    data: Union[Dict[str, Any], str, _DataFrame],
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
        data: Data dictionary from load(), path to CSV/Parquet file,
              pandas DataFrame, or polars DataFrame with OHLCV columns
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
        >>> # Using load() dictionary
        >>> data = mt.load("AAPL.csv")
        >>> signal = np.where(data['close'] > data['close'].mean(), 1, -1)
        >>> results = mt.backtest(data, signal)
        >>> print(results.sharpe)
        1.24

        >>> # Using pandas DataFrame directly
        >>> import pandas as pd
        >>> df = pd.read_csv("AAPL.csv", index_col='date', parse_dates=True)
        >>> results = mt.backtest(df, signal)

        >>> # Using polars DataFrame directly
        >>> import polars as pl
        >>> df = pl.read_csv("AAPL.csv")
        >>> results = mt.backtest(df, signal)
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

def validate(
    data: Union[Dict[str, Any], str, _DataFrame],
    signal: NDArray[np.float64],
    folds: int = 12,
    in_sample_ratio: float = 0.75,
    anchored: bool = True,
    config: Optional[BacktestConfig] = None,
    commission: float = 0.001,
    slippage: float = 0.001,
    size: float = 0.10,
    cash: float = 100_000.0,
) -> ValidationResult:
    """
    Run walk-forward validation on a signal-based strategy.

    This performs walk-forward analysis by splitting the data into multiple
    folds, running the backtest on each in-sample period, and then testing
    on the out-of-sample period.

    Args:
        data: Data dictionary from load() or path to CSV/Parquet file
        signal: numpy array of signals (1=long, -1=short, 0=flat)
        folds: Number of walk-forward folds (default 12)
        in_sample_ratio: Fraction of each fold for in-sample (default 0.75)
        anchored: Use anchored (growing) windows instead of rolling (default True)
        config: Optional BacktestConfig object
        commission: Commission rate (default 0.001 = 0.1%)
        slippage: Slippage rate (default 0.001 = 0.1%)
        size: Position size as fraction of equity (default 0.10 = 10%)
        cash: Initial capital (default 100,000)

    Returns:
        ValidationResult object with fold details and verdict.

    Example:
        >>> data = mt.load("AAPL.csv")
        >>> signal = model.predict(features)
        >>> validation = mt.validate(data, signal)
        >>> print(validation.verdict)
        'robust'
        >>> for fold in validation.fold_details():
        ...     print(f"Fold {fold.fold}: IS={fold.is_sharpe:.2f}, OOS={fold.oos_sharpe:.2f}")
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

def load_results(path: str) -> BacktestResult:
    """
    Load previously saved backtest results from a JSON file.

    This allows you to reload results that were saved with `results.save()`.
    Note: Loaded results cannot use `validate()` since the original data/signal
    are not stored in the JSON file. Use `mt.validate(data, signal)` instead.

    Args:
        path: Path to the JSON file created by `results.save()`

    Returns:
        BacktestResult object with the loaded metrics.

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not valid JSON or missing required fields

    Example:
        >>> results = mt.backtest(data, signal)
        >>> results.save("experiment.json")
        >>> # Later...
        >>> loaded = mt.load_results("experiment.json")
        >>> print(loaded.sharpe)
        1.24
    """
    ...

class Backtest:
    """
    Fluent API for configuring and running backtests.

    Provides a chainable interface for setting backtest parameters
    before running. All parameters have sensible defaults.

    Example:
        >>> results = (
        ...     mt.Backtest(data, signal)
        ...     .commission(0.001)
        ...     .slippage(0.0005)
        ...     .size(0.15)
        ...     .run()
        ... )
        >>> print(results.sharpe)
        1.24
    """

    def __init__(
        self,
        data: Union[Dict[str, Any], str, _DataFrame],
        signal: Optional[NDArray[np.float64]] = None,
        strategy: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> None: ...
    def commission(self, rate: float) -> "Backtest":
        """Set the commission rate (e.g., 0.001 = 0.1%)."""
        ...
    def slippage(self, rate: float) -> "Backtest":
        """Set the slippage rate (e.g., 0.001 = 0.1%)."""
        ...
    def size(self, fraction: float) -> "Backtest":
        """Set the position size as a fraction of equity (e.g., 0.10 = 10%)."""
        ...
    def cash(self, amount: float) -> "Backtest":
        """Set the initial capital."""
        ...
    def stop_loss(self, pct: float) -> "Backtest":
        """Set a stop loss percentage (e.g., 0.05 = 5%)."""
        ...
    def take_profit(self, pct: float) -> "Backtest":
        """Set a take profit percentage (e.g., 0.10 = 10%)."""
        ...
    def allow_short(self, enabled: bool = True) -> "Backtest":
        """Enable or disable short selling."""
        ...
    def borrow_cost(self, rate: float) -> "Backtest":
        """Set the annual borrow cost for short positions (e.g., 0.03 = 3%)."""
        ...
    def run(self) -> BacktestResult:
        """Execute the backtest with the configured parameters."""
        ...

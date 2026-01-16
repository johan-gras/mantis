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
    stop_loss: Optional[Union[float, str]]
    take_profit: Optional[Union[float, str]]
    borrow_cost: float
    max_position: float
    fill_price: str
    freq: Optional[str]
    """Data frequency override (e.g., "1min", "5min", "1h", "1d"). Auto-detected if None."""
    trading_hours_24: Optional[bool]
    """Whether to use 24/7 trading hours for annualization (crypto). Auto-detected if None."""
    max_volume_participation: Optional[float]
    """Maximum volume participation rate (e.g., 0.10 = 10% of bar volume). None = no limit."""
    order_type: str
    """Order type for signal-generated orders. "market" (default) or "limit"."""
    limit_offset: float
    """Limit order offset as fraction of close price (e.g., 0.01 = 1%). Only used when order_type="limit"."""
    model: Optional[str]
    """Path to ONNX model file for inference-based backtesting (requires onnx feature)."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.001,
        position_size: float = 0.10,
        allow_short: bool = True,
        fractional_shares: bool = True,
        stop_loss: Optional[Union[float, str]] = None,
        take_profit: Optional[Union[float, str]] = None,
        borrow_cost: float = 0.03,
        max_position: float = 1.0,
        fill_price: str = "next_open",
        freq: Optional[str] = None,
        trading_hours_24: Optional[bool] = None,
        max_volume_participation: Optional[float] = None,
        order_type: str = "market",
        limit_offset: float = 0.0,
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
    @property
    def alpha(self) -> Optional[float]:
        """
        Get Jensen's alpha (risk-adjusted excess return).

        Alpha represents the strategy's return that is not explained by
        exposure to the benchmark. A positive alpha indicates the strategy
        outperforms the benchmark on a risk-adjusted basis.

        Returns None if no benchmark was provided to backtest().
        """
        ...
    @property
    def beta(self) -> Optional[float]:
        """
        Get portfolio beta (sensitivity to benchmark movements).

        Beta measures how much the strategy's returns move relative to
        the benchmark. A beta of 1.0 means the strategy moves 1:1 with
        the benchmark. Beta > 1 indicates higher volatility than benchmark.

        Returns None if no benchmark was provided to backtest().
        """
        ...
    @property
    def benchmark_return(self) -> Optional[float]:
        """
        Get the benchmark's total return for the backtest period.

        Returns None if no benchmark was provided to backtest().
        """
        ...
    @property
    def excess_return(self) -> Optional[float]:
        """
        Get the excess return (strategy return minus benchmark return).

        Returns None if no benchmark was provided to backtest().
        """
        ...
    @property
    def tracking_error(self) -> Optional[float]:
        """
        Get the tracking error (annualized standard deviation of excess returns).

        Returns None if no benchmark was provided to backtest().
        """
        ...
    @property
    def information_ratio(self) -> Optional[float]:
        """
        Get the information ratio (alpha per unit of active risk).

        Returns None if no benchmark was provided to backtest().
        """
        ...
    @property
    def benchmark_correlation(self) -> Optional[float]:
        """
        Get the correlation with the benchmark (-1 to 1).

        Returns None if no benchmark was provided to backtest().
        """
        ...
    @property
    def up_capture(self) -> Optional[float]:
        """
        Get the up-capture ratio.

        Returns None if no benchmark was provided to backtest().
        """
        ...
    @property
    def down_capture(self) -> Optional[float]:
        """
        Get the down-capture ratio.

        Returns None if no benchmark was provided to backtest().
        """
        ...
    @property
    def has_benchmark(self) -> bool:
        """Whether benchmark comparison metrics are available."""
        ...
    @property
    def volatility(self) -> float:
        """
        Annualized volatility (standard deviation of returns).

        Returns a decimal value (e.g., 0.156 = 15.6% annualized volatility).
        Higher values indicate more risk/variability in returns.
        """
        ...
    @property
    def max_drawdown_duration(self) -> int:
        """
        Maximum drawdown duration in days.

        The longest period from peak equity to recovery to a new peak.
        Long drawdown durations can be psychologically challenging.
        """
        ...
    @property
    def avg_trade_duration(self) -> float:
        """
        Average trade holding period in days.

        The mean duration that positions are held before being closed.
        """
        ...
    def metrics(self) -> Dict[str, Any]: ...
    def summary(self) -> str: ...
    def warnings(self) -> List[str]: ...
    def plot(
        self,
        width: int = 40,
        show_drawdown: bool = True,
        trades: bool = False,
        benchmark: bool = False,
        save: Optional[str] = None,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """
        Display a visualization of the equity curve.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly figure with equity curve and drawdown subplots.
        In terminal or without plotly, returns an ASCII sparkline string.

        Args:
            width: Width of the visualization (characters for ASCII, ignored for Plotly)
            show_drawdown: Whether to show drawdown subplot (Plotly only, default: True)
            trades: Show trade entry/exit markers on the chart (Plotly only)
            benchmark: Show benchmark comparison overlay if available (Plotly only)
            save: Save the plot to a file. Supports .html, .png, .pdf, .svg extensions.
                  Requires kaleido for image export (pip install kaleido)
            title: Custom title for the plot
            height: Custom height in pixels (Plotly only)
            theme: Color theme - "light" or "dark" (Plotly only)

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.
            If save is provided, saves to file and returns the file path.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.plot()  # Shows interactive chart in Jupyter
            >>> results.plot(save="report.html")  # Save to HTML file
            >>> results.plot(trades=True, theme="dark")  # Show trades with dark theme
            >>> print(results.plot())  # ASCII sparkline in terminal
        """
        ...
    def plot_drawdown(
        self,
        save: Optional[str] = None,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """
        Display a dedicated drawdown visualization.

        Shows the drawdown from peak equity over time, highlighting
        periods where the strategy was underwater.

        Args:
            save: Save the plot to a file. Supports .html, .png, .pdf, .svg.
            title: Custom title for the plot.
            height: Custom height in pixels (Plotly only).
            theme: Color theme - "light" or "dark" (Plotly only).

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.plot_drawdown()  # Interactive drawdown chart
            >>> results.plot_drawdown(save="drawdown.html")  # Save to file
        """
        ...
    def plot_returns(
        self,
        period: str = "monthly",
        save: Optional[str] = None,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """
        Display a returns heatmap visualization.

        Shows returns aggregated by month/year as a heatmap, making it
        easy to identify seasonal patterns and performance trends.

        Args:
            period: Aggregation period - "monthly" (default) or "daily".
            save: Save the plot to a file. Supports .html, .png, .pdf, .svg.
            title: Custom title for the plot.
            height: Custom height in pixels (Plotly only).
            theme: Color theme - "light" or "dark" (Plotly only).

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.plot_returns()  # Monthly returns heatmap
            >>> results.plot_returns(period="daily")  # Daily returns histogram
        """
        ...
    def plot_trades(
        self,
        save: Optional[str] = None,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """
        Display a trade analysis visualization.

        Shows trade entry/exit points on the equity curve with
        P&L information for each trade.

        Args:
            save: Save the plot to a file. Supports .html, .png, .pdf, .svg.
            title: Custom title for the plot.
            height: Custom height in pixels (Plotly only).
            theme: Color theme - "light" or "dark" (Plotly only).

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.plot_trades()  # Equity curve with trade markers
        """
        ...
    def rolling_sharpe(
        self,
        window: int = 252,
        annualization_factor: float = 252.0,
    ) -> NDArray[np.float64]:
        """
        Calculate rolling Sharpe ratio over a sliding window.

        Returns annualized Sharpe ratio for each rolling window. Values before
        the window size is reached are NaN.

        Args:
            window: Number of periods for rolling calculation (default: 252 for daily data)
            annualization_factor: Factor to annualize returns (default: 252.0 for daily)

        Returns:
            Numpy array of rolling Sharpe ratios with same length as equity curve.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> rolling = results.rolling_sharpe(window=252)
            >>> rolling[-1]  # Most recent Sharpe ratio
            1.45
        """
        ...
    def rolling_drawdown(
        self,
        window: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """
        Calculate rolling drawdown from peak equity.

        Returns drawdown as a fraction (negative values) at each point in time.
        A value of -0.10 means the equity is 10% below its peak.

        Args:
            window: Optional maximum lookback window for peak (None = all history)

        Returns:
            Numpy array of drawdown values (0 at peaks, negative otherwise).

        Example:
            >>> results = mt.backtest(data, signal)
            >>> dd = results.rolling_drawdown()
            >>> dd.min()  # Maximum drawdown
            -0.15
            >>> dd_52week = results.rolling_drawdown(window=252)  # 52-week lookback
        """
        ...
    def rolling_max_drawdown(
        self,
        window: int = 252,
    ) -> NDArray[np.float64]:
        """
        Calculate the worst drawdown within each rolling window.

        Returns the maximum (worst) drawdown observed within each window.
        Useful for tracking strategy risk over time.

        Args:
            window: Rolling window size in periods (default: 252 for 1 year of daily data)

        Returns:
            Numpy array of worst drawdown values for each window.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> rolling_max_dd = results.rolling_max_drawdown(window=252)
        """
        ...
    def rolling_volatility(
        self,
        window: int = 21,
        annualization_factor: float = 252.0,
    ) -> NDArray[np.float64]:
        """
        Calculate rolling volatility from equity returns.

        Returns annualized volatility for each rolling window.
        Values before the window size is reached are NaN.

        Args:
            window: Number of periods for rolling calculation (default: 21 for monthly)
            annualization_factor: Factor to annualize volatility (default: 252.0)

        Returns:
            Numpy array of annualized volatility values.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> vol = results.rolling_volatility(window=21)
            >>> vol[-1]  # Most recent 21-day volatility
            0.18
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

# =============================================================================
# ONNX Model Support (requires onnx feature)
# =============================================================================

class ModelConfig:
    """
    Configuration for ONNX model inference.

    Specifies model parameters including input/output sizes, normalization,
    and fallback behavior.

    Example:
        >>> config = mt.ModelConfig("my_model", input_size=10)
        >>> model = mt.OnnxModel("model.onnx", config=config)
    """

    name: str
    version: str
    input_size: int
    output_size: int
    normalize_inputs: bool
    feature_means: Optional[List[float]]
    feature_stds: Optional[List[float]]
    fallback_value: float
    log_latency: bool
    batch_size: int

    def __init__(
        self,
        name: str = "unnamed_model",
        input_size: int = 0,
        version: str = "1.0.0",
        output_size: int = 1,
        normalize_inputs: bool = False,
        feature_means: Optional[List[float]] = None,
        feature_stds: Optional[List[float]] = None,
        fallback_value: float = 0.0,
        log_latency: bool = False,
        batch_size: int = 1,
    ) -> None: ...


class InferenceStats:
    """
    Statistics tracked during model inference.

    Provides performance metrics for ONNX model inference including
    total inferences, success rate, and latency measurements.
    """

    total_inferences: int
    successful_inferences: int
    failed_inferences: int
    total_inference_time_us: int
    min_inference_time_us: int
    max_inference_time_us: int

    def avg_inference_time_us(self) -> float:
        """Get average inference time in microseconds."""
        ...
    def avg_inference_time_ms(self) -> float:
        """Get average inference time in milliseconds."""
        ...
    def success_rate(self) -> float:
        """Get success rate as a percentage (0-100)."""
        ...


class OnnxModel:
    """
    ONNX model wrapper for inference.

    Loads and runs ONNX models for generating trading signals during backtests.
    Requires the onnx feature to be enabled.

    Example:
        >>> model = mt.OnnxModel("model.onnx", input_size=10)
        >>> prediction = model.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        >>> print(prediction)
        0.75
    """

    def __init__(
        self,
        path: str,
        input_size: int,
        config: Optional[ModelConfig] = None,
        name: Optional[str] = None,
        version: str = "1.0.0",
        normalize: bool = False,
        feature_means: Optional[List[float]] = None,
        feature_stds: Optional[List[float]] = None,
        fallback_value: float = 0.0,
    ) -> None:
        """
        Load an ONNX model from a file.

        Args:
            path: Path to the .onnx model file
            input_size: Number of input features (required)
            config: Optional ModelConfig for advanced settings
            name: Model name for logging (default: filename)
            version: Model version (default: "1.0.0")
            normalize: Whether to normalize inputs (requires means/stds)
            feature_means: Feature means for z-score normalization
            feature_stds: Feature standard deviations for normalization
            fallback_value: Value to return if inference fails (default: 0.0)
        """
        ...

    @property
    def input_size(self) -> int:
        """Get the model's input size (number of features)."""
        ...

    @property
    def output_size(self) -> int:
        """Get the model's output size."""
        ...

    @property
    def name(self) -> str:
        """Get the model's name."""
        ...

    @property
    def version(self) -> str:
        """Get the model's version."""
        ...

    def predict(self, features: List[float]) -> float:
        """
        Perform inference on a single feature vector.

        Args:
            features: List of input features

        Returns:
            Predicted value (float)
        """
        ...

    def predict_batch(self, batch_features: List[List[float]]) -> List[float]:
        """
        Perform batch inference on multiple feature vectors.

        Args:
            batch_features: List of feature vectors

        Returns:
            List of predictions
        """
        ...

    def stats(self) -> InferenceStats:
        """Get inference statistics."""
        ...

    def reset_stats(self) -> None:
        """Reset inference statistics."""
        ...

    def print_stats(self) -> None:
        """Print inference statistics to stdout."""
        ...


def load_model(
    path: str,
    input_size: int,
    name: Optional[str] = None,
    version: str = "1.0.0",
    fallback_value: float = 0.0,
) -> OnnxModel:
    """
    Load an ONNX model from a file.

    Convenience function that wraps OnnxModel creation.

    Args:
        path: Path to the .onnx model file
        input_size: Number of input features
        name: Model name for logging (default: filename)
        version: Model version (default: "1.0.0")
        fallback_value: Value to return if inference fails

    Returns:
        Loaded OnnxModel

    Example:
        >>> model = mt.load_model("model.onnx", input_size=10)
    """
    ...


def generate_signals(
    model: OnnxModel,
    features: Any,
    threshold: Optional[float] = None,
) -> NDArray[np.float64]:
    """
    Generate signals from an ONNX model and feature DataFrame.

    Runs the ONNX model on each row of the feature DataFrame
    and returns a numpy array of predictions that can be used as signals.

    Args:
        model: OnnxModel instance
        features: Feature array or DataFrame with features for each bar
        threshold: Optional threshold for converting predictions to signals.
                   If provided, predictions > threshold become 1,
                   < -threshold become -1, else 0.

    Returns:
        numpy array of predictions/signals

    Example:
        >>> model = mt.load_model("model.onnx", input_size=10)
        >>> signals = mt.generate_signals(model, feature_df, threshold=0.5)
        >>> results = mt.backtest(data, signals)
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
    stop_loss: Optional[Union[float, str]] = None,
    take_profit: Optional[Union[float, str]] = None,
    allow_short: bool = True,
    borrow_cost: float = 0.03,
    max_position: float = 1.0,
    fill_price: str = "next_open",
    benchmark: Optional[Union[Dict[str, Any], str, _DataFrame]] = None,
    freq: Optional[str] = None,
    trading_hours_24: Optional[bool] = None,
    max_volume_participation: Optional[float] = None,
    order_type: str = "market",
    limit_offset: float = 0.0,
    model: Optional[str] = None,
    features: Optional[Any] = None,
    model_input_size: Optional[int] = None,
    signal_threshold: Optional[float] = None,
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
        stop_loss: Optional stop loss. Can be:
            - float: percentage (e.g., 0.05 for 5% stop loss)
            - str: ATR-based (e.g., "2atr" for 2x ATR stop loss)
            - str: trailing (e.g., "5trail" for 5% trailing stop)
        take_profit: Optional take profit. Can be:
            - float: percentage (e.g., 0.10 for 10% take profit)
            - str: ATR-based (e.g., "3atr" for 3x ATR take profit)
            - str: risk-reward (e.g., "2rr" for 2:1 risk-reward ratio)
        allow_short: Whether to allow short positions
        borrow_cost: Annual borrow rate for shorts (default 3%)
        max_position: Maximum position size as fraction of equity (default 1.0 = 100%)
        fill_price: Execution price model ("next_open", "close", "vwap", "twap", "midpoint")
        benchmark: Optional benchmark data for performance comparison (data dict from load()).
                   When provided, the result will include alpha, beta, benchmark_return,
                   excess_return, and other benchmark comparison metrics.
        freq: Data frequency override ("1s", "5s", "1min", "5min", "15min", "30min",
              "1h", "4h", "1d", "1w", "1mo"). Auto-detected from bar timestamps if None.
        trading_hours_24: Whether to use 24/7 trading hours for metric annualization (crypto).
                          Auto-detected from weekend bars if None.
        max_volume_participation: Maximum volume participation rate (e.g., 0.10 = 10% of bar volume).
                                  Prevents unrealistic fills in illiquid markets. None = no limit.
        order_type: Order type for signal-generated orders. Options:
            - "market" (default): Execute at market price
            - "limit": Place limit orders at offset from close price
        limit_offset: Limit order offset as fraction of close price (e.g., 0.01 = 1%).
            For buys: limit_price = close * (1 - limit_offset) (below close)
            For sells: limit_price = close * (1 + limit_offset) (above close)
            Only used when order_type="limit".
        model: Path to ONNX model file for inference-based backtesting.
            When provided, signals are generated by running the model on the features.
            Requires the onnx feature to be enabled.
        features: Feature array or DataFrame for ONNX model inference.
            Required when model is provided. Must have one row per bar in data.
        model_input_size: Number of input features for the ONNX model.
            Auto-detected from features if not provided.
        signal_threshold: Threshold for converting model predictions to signals.
            Predictions > threshold become 1, < -threshold become -1, else 0.
            If None, raw predictions are used as signals.

    Returns:
        BacktestResult object with metrics, equity curve, and trades.
        If benchmark is provided, also includes alpha, beta, benchmark_return, excess_return.

    Example:
        >>> # Using load() dictionary
        >>> data = mt.load("AAPL.csv")
        >>> signal = np.where(data['close'] > data['close'].mean(), 1, -1)
        >>> results = mt.backtest(data, signal)
        >>> print(results.sharpe)
        1.24

        >>> # With ATR-based stop loss
        >>> results = mt.backtest(data, signal, stop_loss="2atr")

        >>> # With percentage stop loss and ATR take profit
        >>> results = mt.backtest(data, signal, stop_loss=0.05, take_profit="3atr")

        >>> # With risk-reward ratio
        >>> results = mt.backtest(data, signal, stop_loss="2atr", take_profit="2rr")

        >>> # With benchmark comparison
        >>> spy = mt.load("SPY.csv")
        >>> results = mt.backtest(data, signal, benchmark=spy)
        >>> print(results.alpha, results.beta)
        0.05 1.2

        >>> # Explicit frequency override for 5-minute data
        >>> results = mt.backtest(data, signal, freq="5min")

        >>> # 24/7 market (crypto) with proper annualization
        >>> results = mt.backtest(btc_data, signal, trading_hours_24=True)

        >>> # Using ONNX model for inference (requires onnx feature)
        >>> results = mt.backtest(data, model="model.onnx", features=feature_df)
        >>> # With signal threshold
        >>> results = mt.backtest(data, model="model.onnx", features=features, signal_threshold=0.5)
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

class CompareResult:
    """
    Result of comparing multiple backtest strategies.

    In Jupyter notebooks with plotly installed, automatically displays an
    interactive equity curve chart with all strategies overlaid. Also provides
    access to comparison metrics as a dictionary.
    """

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get comparison metrics as a dictionary."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        ...

    def plot(
        self,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
        save: Optional[str] = None,
    ) -> Any:
        """
        Display a visualization comparing all strategies.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly figure with equity curves for all strategies overlaid.

        Args:
            title: Chart title (default: "Strategy Comparison")
            height: Chart height in pixels (default: 500)
            theme: "light" or "dark" theme
            save: Optional path to save chart (HTML, PNG, PDF supported)

        Returns:
            Plotly Figure object in Jupyter with plotly, string otherwise.
        """
        ...


def compare(
    results: List[BacktestResult],
    names: Optional[List[str]] = None,
) -> CompareResult:
    """
    Compare multiple backtest results side by side.

    In Jupyter notebooks with plotly installed, automatically displays an
    interactive equity curve chart with all strategies overlaid.

    Args:
        results: List of BacktestResult objects to compare
        names: Optional list of names for each result

    Returns:
        CompareResult object with metrics dict and plot() method.
        In Jupyter, automatically renders as interactive chart.
    """
    ...

class SweepResultItem:
    """Result of a single parameter combination in a sweep."""

    @property
    def params(self) -> Dict[str, float]:
        """The parameter combination as a dictionary."""
        ...

    @property
    def result(self) -> BacktestResult:
        """The backtest result for this combination."""
        ...


class SweepResult:
    """
    Result of a parallel parameter sweep.

    Provides methods for analyzing and visualizing parameter sweep results.
    """

    @property
    def num_combinations(self) -> int:
        """Total number of parameter combinations tested."""
        ...

    @property
    def parallel(self) -> bool:
        """Whether parallel execution was used."""
        ...

    def items(self) -> List[SweepResultItem]:
        """Get all results as a list of SweepResultItem objects."""
        ...

    def to_dict(self) -> Dict[str, BacktestResult]:
        """Get all results as a dictionary mapping param strings to BacktestResult."""
        ...

    def best(
        self, metric: str = "sharpe", maximize: bool = True
    ) -> Optional[SweepResultItem]:
        """
        Get the best result by a given metric.

        Args:
            metric: Metric to optimize ("sharpe", "sortino", "return", "calmar", "profit_factor")
            maximize: Whether to maximize (default True) or minimize the metric

        Returns:
            The SweepResultItem with the best metric value.
        """
        ...

    def best_params(
        self, metric: str = "sharpe", maximize: bool = True
    ) -> Optional[Dict[str, float]]:
        """Get the best parameters by a given metric."""
        ...

    def sorted_by(
        self, metric: str = "sharpe", descending: bool = True
    ) -> List[SweepResultItem]:
        """Get results sorted by a metric."""
        ...

    def top(self, n: int = 10, metric: str = "sharpe") -> List[SweepResultItem]:
        """Get top N results by a metric."""
        ...

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics across all parameter combinations."""
        ...

    def plot(
        self,
        x_param: Optional[str] = None,
        y_param: Optional[str] = None,
        metric: str = "sharpe",
    ) -> Any:
        """
        Plot sweep results.

        In Jupyter with plotly installed, shows an interactive heatmap.
        Otherwise, prints a summary table.

        Args:
            x_param: Parameter for X axis (auto-detected if not provided)
            y_param: Parameter for Y axis (auto-detected if not provided)
            metric: Metric to visualize (default "sharpe")

        Returns:
            Plotly Figure in Jupyter, None otherwise.
        """
        ...


def sweep(
    data: Any,
    signal_fn: Callable[..., NDArray[np.float64]],
    params: Dict[str, List[Any]],
    n_jobs: int = -1,
    parallel: bool = True,
    **backtest_kwargs: Any,
) -> SweepResult:
    """
    Run a parameter sweep, testing multiple parameter combinations.

    Uses Rust's rayon for parallel execution across all CPU cores.

    Args:
        data: Data dictionary from load() or file path
        signal_fn: Function that takes params and returns a signal array
        params: Dictionary of parameter names to lists of values
        n_jobs: Number of parallel jobs (-1 for all cores). Deprecated, use `parallel`.
        parallel: Whether to use parallel execution (default True)
        **backtest_kwargs: Additional arguments passed to backtest()

    Returns:
        SweepResult object with results for each parameter combination.
        Use .best() to get the best result, .to_dict() for dict format.

    Example:
        >>> def generate_signal(threshold):
        ...     return (predictions > threshold).astype(int)
        >>>
        >>> sweep_results = mt.sweep(
        ...     data,
        ...     generate_signal,
        ...     params={"threshold": [0.1, 0.2, 0.3, 0.4, 0.5]}
        ... )
        >>> best = sweep_results.best()  # Get best by Sharpe
        >>> print(best.params, best.result.sharpe)
    """
    ...

def adjust(
    data: Dict[str, Any],
    splits: Optional[List[Dict[str, Any]]] = None,
    dividends: Optional[List[Dict[str, Any]]] = None,
    method: str = "proportional",
) -> Dict[str, Any]:
    """
    Adjust price data for stock splits and dividends.

    This function modifies OHLCV data to account for corporate actions,
    ensuring historical prices are comparable to current prices. This is
    essential for accurate backtesting, as unadjusted data will produce
    false signals at split/dividend dates.

    Args:
        data: Data dictionary from load() containing OHLCV arrays
        splits: Optional list of split dictionaries with keys:
            - date: Split date (str "YYYY-MM-DD" or int timestamp)
            - ratio: Split ratio (e.g., 2.0 for 2:1 split)
            - reverse: Optional bool, True for reverse splits (default False)
        dividends: Optional list of dividend dictionaries with keys:
            - date: Ex-dividend date (str "YYYY-MM-DD" or int timestamp)
            - amount: Dividend amount per share
            - type: Optional dividend type ("regular", "special", "qualified")
        method: Dividend adjustment method:
            - "proportional": Adjust prices proportionally (default, preserves returns)
            - "absolute": Subtract dividend from prices
            - "none": No dividend adjustment (only apply splits)

    Returns:
        New data dictionary with adjusted OHLCV arrays.

    Example:
        >>> data = mt.load("AAPL.csv")
        >>> # Adjust for a 4:1 split on 2020-08-31
        >>> adjusted = mt.adjust(
        ...     data,
        ...     splits=[{"date": "2020-08-31", "ratio": 4.0}],
        ... )
        >>> # Adjust for dividends
        >>> adjusted = mt.adjust(
        ...     data,
        ...     dividends=[
        ...         {"date": "2024-02-09", "amount": 0.24},
        ...         {"date": "2024-05-10", "amount": 0.25},
        ...     ],
        ... )
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
    def stop_loss(self, value: Union[float, str]) -> "Backtest":
        """
        Set a stop loss.

        Args:
            value: Stop loss specification. Can be:
                - float: percentage (e.g., 0.05 for 5%)
                - str: ATR-based (e.g., "2atr" for 2x ATR)
                - str: trailing (e.g., "5trail" for 5% trailing)

        Example:
            >>> mt.Backtest(data, signal).stop_loss("2atr")  # 2x ATR stop
        """
        ...
    def take_profit(self, value: Union[float, str]) -> "Backtest":
        """
        Set a take profit.

        Args:
            value: Take profit specification. Can be:
                - float: percentage (e.g., 0.10 for 10%)
                - str: ATR-based (e.g., "3atr" for 3x ATR)
                - str: risk-reward (e.g., "2rr" for 2:1 R:R ratio)

        Example:
            >>> mt.Backtest(data, signal).take_profit("2rr")  # 2:1 risk-reward
        """
        ...
    def allow_short(self, enabled: bool = True) -> "Backtest":
        """Enable or disable short selling."""
        ...
    def borrow_cost(self, rate: float) -> "Backtest":
        """Set the annual borrow cost for short positions (e.g., 0.03 = 3%)."""
        ...
    def max_position(self, fraction: float) -> "Backtest":
        """Set the maximum position size as a fraction of equity (e.g., 0.25 = 25%)."""
        ...
    def fill_price(self, model: str) -> "Backtest":
        """
        Set the execution price model.

        Options: "next_open" (default), "close", "vwap", "twap", "midpoint"
        """
        ...
    def benchmark(self, data: Union[Dict[str, Any], str, _DataFrame]) -> "Backtest":
        """
        Set benchmark data for performance comparison.

        When benchmark is provided, the result will include alpha, beta,
        benchmark_return, excess_return, and other comparison metrics.

        Example:
            >>> spy = mt.load("SPY.csv")
            >>> results = mt.Backtest(data, signal).benchmark(spy).run()
            >>> print(results.alpha, results.beta)
        """
        ...
    def freq(self, frequency: str) -> "Backtest":
        """
        Override the data frequency for metric annualization.

        Args:
            frequency: Data frequency. Options:
                - "1s", "5s", "10s", "15s", "30s": Second frequencies
                - "1min", "5min", "15min", "30min": Minute frequencies
                - "1h", "4h": Hourly frequencies
                - "1d" or "daily": Daily frequency
                - "1w" or "weekly": Weekly frequency
                - "1mo" or "monthly": Monthly frequency

        Example:
            >>> results = mt.Backtest(data, signal).freq("5min").run()
        """
        ...
    def trading_hours_24(self, enabled: bool = True) -> "Backtest":
        """
        Enable 24/7 trading hours for metric annualization (crypto markets).

        When enabled, metrics are annualized using 365 days/year instead of 252.

        Args:
            enabled: Whether to use 24/7 market annualization (default True)

        Example:
            >>> results = mt.Backtest(btc_data, signal).trading_hours_24().run()
        """
        ...
    def max_volume_participation(self, rate: float) -> "Backtest":
        """
        Set the maximum volume participation rate.

        This limits the maximum order size to a fraction of the bar's volume,
        preventing unrealistic fills in illiquid markets.

        Args:
            rate: Maximum participation rate as a fraction (e.g., 0.10 = 10%)

        Example:
            >>> results = mt.Backtest(data, signal).max_volume_participation(0.05).run()
        """
        ...
    def order_type(self, order_type: str) -> "Backtest":
        """
        Set the order type for signal-generated orders.

        By default, signals are converted to market orders. Use this method
        to place limit orders instead, which can provide more realistic
        execution simulation for passive strategies.

        Args:
            order_type: Order type. Options:
                - "market" (default): Execute at market price
                - "limit": Place limit orders at offset from close price

        Example:
            >>> results = mt.Backtest(data, signal).order_type("limit").limit_offset(0.005).run()
        """
        ...
    def limit_offset(self, offset: float) -> "Backtest":
        """
        Set the limit order offset from close price.

        Only used when order_type="limit". The offset determines how far
        from the close price the limit order is placed.

        For buy orders: limit_price = close * (1 - offset) (below close)
        For sell orders: limit_price = close * (1 + offset) (above close)

        Args:
            offset: Offset as a fraction of close price (e.g., 0.01 = 1%)

        Example:
            >>> results = mt.Backtest(data, signal).order_type("limit").limit_offset(0.01).run()
        """
        ...
    def run(self) -> BacktestResult:
        """Execute the backtest with the configured parameters."""
        ...


# =============================================================================
# Sensitivity Analysis
# =============================================================================

class ParameterRange:
    """
    A range of parameter values for sensitivity analysis.

    Use the factory functions to create ranges:
    - linear_range(start, end, steps): Evenly spaced values
    - log_range(start, end, steps): Logarithmically spaced values
    - discrete_range(values): Explicit list of values
    - centered_range(center, variation, steps): Range around a center value
    """

    def values(self) -> NDArray[np.float64]:
        """Get all values in this range as a numpy array."""
        ...
    def __len__(self) -> int: ...


class Cliff:
    """Detected cliff (sharp performance drop) in parameter space."""

    parameter: str
    value_before: float
    value_after: float
    metric_before: float
    metric_after: float
    drop_pct: float


class Plateau:
    """Detected plateau (stable performance region) in parameter space."""

    parameter: str
    start_value: float
    end_value: float
    avg_metric: float
    std_metric: float


class HeatmapData:
    """2D heatmap data for parameter sensitivity visualization."""

    x_param: str
    y_param: str

    def x_values(self) -> NDArray[np.float64]:
        """Get X-axis values."""
        ...
    def y_values(self) -> NDArray[np.float64]:
        """Get Y-axis values."""
        ...
    def values(self) -> NDArray[np.float64]:
        """Get 2D grid of metric values (NaN for missing)."""
        ...
    def to_csv(self) -> str:
        """Export to CSV format."""
        ...
    def best(self) -> Optional[tuple]:
        """Find best value: (x_value, y_value, metric_value)."""
        ...


class SensitivitySummary:
    """Summary statistics from sensitivity analysis."""

    num_combinations: int
    metric: str
    mean_metric: float
    std_metric: float
    min_metric: float
    max_metric: float
    stability_score: float
    num_cliffs: int
    num_plateaus: int
    is_fragile: bool


class SensitivityResult:
    """Results from parameter sensitivity analysis."""

    strategy_name: str
    symbol: str
    num_combinations: int

    def best_params(self) -> Optional[Dict[str, float]]:
        """Get the best performing parameter set."""
        ...
    def stability_score(self) -> float:
        """Get the overall stability score (0-1, higher is more stable)."""
        ...
    def parameter_stability(self, param_name: str) -> Optional[float]:
        """Get stability score for a specific parameter."""
        ...
    def is_fragile(self, threshold: float = 0.5) -> bool:
        """Check if the strategy is fragile (high sensitivity to parameters)."""
        ...
    def heatmap(self, x_param: str, y_param: str) -> Optional[HeatmapData]:
        """Get 2D heatmap data for two parameters."""
        ...
    def parameter_importance(self) -> List[tuple]:
        """Get parameter importance ranking as (name, importance) tuples."""
        ...
    def cliffs(self) -> List[Cliff]:
        """Get detected cliffs (sharp performance drops)."""
        ...
    def plateaus(self) -> List[Plateau]:
        """Get detected plateaus (stable performance regions)."""
        ...
    def summary(self) -> SensitivitySummary:
        """Get summary statistics."""
        ...
    def to_csv(self) -> str:
        """Export results to CSV format."""
        ...
    def plot_heatmap(self, x_param: str, y_param: str) -> Any:
        """
        Display a heatmap visualization for two parameters.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly heatmap. Otherwise returns CSV representation.
        """
        ...


class CostScenario:
    """Results for a single cost multiplier level."""

    multiplier: float
    total_return: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    total_costs: float
    avg_cost_per_trade: float

    def is_zero_cost(self) -> bool:
        """Whether this is the zero-cost baseline."""
        ...
    def is_baseline(self) -> bool:
        """Whether this is the baseline (1x) scenario."""
        ...


class CostSensitivityResult:
    """Results from cost sensitivity analysis."""

    symbol: str
    strategy_name: str

    def scenarios(self) -> List[CostScenario]:
        """Get all scenarios."""
        ...
    def scenario_at(self, multiplier: float) -> Optional[CostScenario]:
        """Get scenario at specific multiplier."""
        ...
    def baseline(self) -> Optional[CostScenario]:
        """Get baseline (1x) scenario."""
        ...
    def zero_cost(self) -> Optional[CostScenario]:
        """Get zero-cost scenario (theoretical upper bound)."""
        ...
    def sharpe_degradation_at(self, multiplier: float) -> Optional[float]:
        """Calculate Sharpe ratio degradation percentage at given multiplier."""
        ...
    def return_degradation_at(self, multiplier: float) -> Optional[float]:
        """Calculate return degradation percentage at given multiplier."""
        ...
    def is_robust(self, threshold_sharpe: float = 0.5) -> bool:
        """Check if strategy passes robustness threshold at 5x costs."""
        ...
    def cost_elasticity(self) -> Optional[float]:
        """Calculate cost elasticity (% change in return per % change in costs)."""
        ...
    def breakeven_multiplier(self) -> Optional[float]:
        """Calculate breakeven cost multiplier (where returns become zero/negative)."""
        ...
    def report(self) -> str:
        """Generate formatted summary report."""
        ...
    def plot(self) -> Any:
        """
        Display a visualization of cost sensitivity.

        In Jupyter with plotly, returns interactive figure. Otherwise returns text report.
        """
        ...


def linear_range(start: float, end: float, steps: int) -> ParameterRange:
    """
    Create a linear parameter range.

    Generates evenly spaced values from start to end.

    Args:
        start: Starting value
        end: Ending value
        steps: Number of values to generate

    Example:
        >>> fast_range = mt.linear_range(5.0, 20.0, 4)
        >>> # Generates: [5.0, 10.0, 15.0, 20.0]
    """
    ...


def log_range(start: float, end: float, steps: int) -> ParameterRange:
    """
    Create a logarithmic parameter range.

    Generates logarithmically spaced values from start to end.
    Useful for parameters spanning multiple orders of magnitude.

    Args:
        start: Starting value (must be > 0)
        end: Ending value (must be > 0)
        steps: Number of values to generate

    Example:
        >>> rate_range = mt.log_range(0.001, 0.1, 3)
        >>> # Generates: [0.001, 0.01, 0.1]
    """
    ...


def discrete_range(values: List[float]) -> ParameterRange:
    """
    Create a discrete parameter range from explicit values.

    Args:
        values: List of parameter values to test

    Example:
        >>> periods = mt.discrete_range([5, 10, 20, 50])
    """
    ...


def centered_range(center: float, variation: float, steps: int) -> ParameterRange:
    """
    Create a centered parameter range around a base value.

    Args:
        center: Center value
        variation: Plus/minus variation amount
        steps: Number of values to generate

    Example:
        >>> threshold_range = mt.centered_range(0.5, 0.1, 5)
        >>> # Generates: [0.4, 0.45, 0.5, 0.55, 0.6]
    """
    ...


def sensitivity(
    data: Union[Dict[str, Any], str, _DataFrame],
    strategy: str,
    params: Dict[str, ParameterRange],
    metric: str = "sharpe",
    commission: float = 0.001,
    slippage: float = 0.001,
    cash: float = 100_000.0,
    parallel: bool = True,
) -> SensitivityResult:
    """
    Run parameter sensitivity analysis on a built-in strategy.

    Tests how strategy performance varies across different parameter values.
    This helps identify:
    - Fragile strategies that only work with specific parameters
    - Robust strategies that perform well across parameter ranges
    - Cliffs where performance drops sharply
    - Plateaus where performance is stable

    Args:
        data: Data dictionary from load() or path to CSV/Parquet file
        strategy: Name of built-in strategy ("sma-crossover", "momentum",
                  "mean-reversion", "rsi", "macd", "breakout")
        params: Dictionary mapping parameter names to ParameterRange objects
        metric: Metric to analyze ("sharpe", "sortino", "return", "calmar",
                "profit_factor", "win_rate", "max_drawdown")
        commission: Commission rate (default 0.001 = 0.1%)
        slippage: Slippage rate (default 0.001 = 0.1%)
        cash: Initial capital (default 100,000)
        parallel: Run parameter combinations in parallel (default True)

    Returns:
        SensitivityResult with analysis results.

    Example:
        >>> data = mt.load_sample("AAPL")
        >>> result = mt.sensitivity(
        ...     data,
        ...     strategy="sma-crossover",
        ...     params={
        ...         "fast_period": mt.linear_range(5, 20, 4),
        ...         "slow_period": mt.linear_range(20, 60, 5),
        ...     },
        ...     metric="sharpe"
        ... )
        >>> print(result.stability_score())
        0.72
        >>> print(result.best_params())
        {'fast_period': 10.0, 'slow_period': 40.0}
    """
    ...


def cost_sensitivity(
    data: Union[Dict[str, Any], str, _DataFrame],
    signal: Optional[NDArray[np.float64]] = None,
    strategy: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    multipliers: Optional[List[float]] = None,
    commission: float = 0.001,
    slippage: float = 0.001,
    cash: float = 100_000.0,
    include_zero_cost: bool = True,
) -> CostSensitivityResult:
    """
    Run cost sensitivity analysis to test strategy robustness to transaction costs.

    Tests the same strategy at multiple cost levels (e.g., 1x, 2x, 5x, 10x) to see
    how performance degrades. A robust strategy should maintain acceptable returns
    even with significantly higher costs.

    Args:
        data: Data dictionary from load() or path to CSV/Parquet file
        signal: Signal array (1=long, -1=short, 0=flat), or None for built-in strategy
        strategy: Name of built-in strategy if signal is None
        strategy_params: Parameters for built-in strategy
        multipliers: List of cost multipliers to test (default [0.0, 1.0, 2.0, 5.0, 10.0])
        commission: Base commission rate (default 0.001 = 0.1%)
        slippage: Base slippage rate (default 0.001 = 0.1%)
        cash: Initial capital (default 100,000)
        include_zero_cost: Include zero-cost scenario (default True)

    Returns:
        CostSensitivityResult with analysis results.

    Example:
        >>> data = mt.load_sample("AAPL")
        >>> signal = (data['close'] > data['close'].mean()).astype(int)
        >>> result = mt.cost_sensitivity(data, signal)
        >>> print(result.sharpe_degradation_at(5.0))
        45.2
        >>> print(result.is_robust())
        True
    """
    ...

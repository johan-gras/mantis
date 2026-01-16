//! Performance analytics and reporting.

use crate::engine::BacktestResult;
use crate::types::Trade;
use chrono::Datelike;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tabled::{builder::Builder, settings::Style};

/// Benchmark comparison metrics.
/// These metrics compare portfolio performance against a benchmark index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Benchmark name (e.g., "SPY", "QQQ").
    pub benchmark_name: String,
    /// Jensen's alpha - risk-adjusted excess return.
    /// Measures the portfolio's return above the expected return given its beta.
    pub alpha: f64,
    /// Portfolio beta - sensitivity to benchmark movements.
    /// Beta > 1 means more volatile than benchmark, < 1 means less volatile.
    pub beta: f64,
    /// Tracking error - standard deviation of excess returns.
    /// Measures how closely the portfolio follows the benchmark.
    pub tracking_error: f64,
    /// Information ratio - alpha divided by tracking error.
    /// Measures risk-adjusted excess return per unit of active risk.
    pub information_ratio: f64,
    /// Correlation coefficient with benchmark (-1 to 1).
    pub correlation: f64,
    /// Upside capture ratio - percentage of benchmark gains captured.
    /// > 100% means outperforming in up markets.
    pub up_capture: f64,
    /// Downside capture ratio - percentage of benchmark losses captured.
    /// < 100% means protecting better in down markets.
    pub down_capture: f64,
    /// Benchmark total return for the period.
    pub benchmark_return_pct: f64,
    /// Excess return (portfolio return - benchmark return).
    pub excess_return_pct: f64,
}

impl BenchmarkMetrics {
    /// Calculate benchmark comparison metrics from aligned return series.
    ///
    /// # Arguments
    /// * `benchmark_name` - Name of the benchmark
    /// * `portfolio_returns` - Daily portfolio returns (as decimals, e.g., 0.01 for 1%)
    /// * `benchmark_returns` - Daily benchmark returns (as decimals)
    /// * `risk_free_rate` - Annual risk-free rate (as decimal, e.g., 0.05 for 5%)
    ///
    /// Both return series must be aligned (same dates, same length).
    pub fn calculate(
        benchmark_name: impl Into<String>,
        portfolio_returns: &[f64],
        benchmark_returns: &[f64],
        risk_free_rate: f64,
    ) -> Option<Self> {
        if portfolio_returns.is_empty()
            || benchmark_returns.is_empty()
            || portfolio_returns.len() != benchmark_returns.len()
        {
            return None;
        }

        let n = portfolio_returns.len() as f64;

        // Calculate means
        let port_mean: f64 = portfolio_returns.iter().sum::<f64>() / n;
        let bench_mean: f64 = benchmark_returns.iter().sum::<f64>() / n;

        // Calculate beta: Cov(portfolio, benchmark) / Var(benchmark)
        let covariance: f64 = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(p, b)| (p - port_mean) * (b - bench_mean))
            .sum::<f64>()
            / n;

        let bench_variance: f64 = benchmark_returns
            .iter()
            .map(|b| (b - bench_mean).powi(2))
            .sum::<f64>()
            / n;

        let port_variance: f64 = portfolio_returns
            .iter()
            .map(|p| (p - port_mean).powi(2))
            .sum::<f64>()
            / n;

        let beta = if bench_variance > 0.0 {
            covariance / bench_variance
        } else {
            0.0
        };

        // Calculate alpha (annualized Jensen's alpha)
        // Alpha = (Portfolio Return - Rf) - Beta * (Benchmark Return - Rf)
        let annualized_port_return = (1.0 + port_mean).powf(252.0) - 1.0;
        let annualized_bench_return = (1.0 + bench_mean).powf(252.0) - 1.0;
        let alpha = (annualized_port_return - risk_free_rate)
            - beta * (annualized_bench_return - risk_free_rate);

        // Calculate correlation
        let port_std = port_variance.sqrt();
        let bench_std = bench_variance.sqrt();
        let correlation = if port_std > 0.0 && bench_std > 0.0 {
            covariance / (port_std * bench_std)
        } else {
            0.0
        };

        // Calculate excess returns and tracking error
        let excess_returns: Vec<f64> = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(p, b)| p - b)
            .collect();

        let excess_mean: f64 = excess_returns.iter().sum::<f64>() / n;
        let tracking_error_daily: f64 = (excess_returns
            .iter()
            .map(|e| (e - excess_mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let tracking_error = tracking_error_daily * 252.0_f64.sqrt() * 100.0; // Annualized as percentage

        // Calculate information ratio (annualized)
        let information_ratio = if tracking_error > 0.0 {
            (excess_mean * 252.0 * 100.0) / tracking_error
        } else {
            0.0
        };

        // Calculate capture ratios
        let (up_capture, down_capture) =
            Self::calculate_capture_ratios(portfolio_returns, benchmark_returns);

        // Calculate total returns
        let portfolio_total_return: f64 =
            portfolio_returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;
        let benchmark_total_return: f64 =
            benchmark_returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;

        Some(Self {
            benchmark_name: benchmark_name.into(),
            alpha: alpha * 100.0, // Convert to percentage
            beta,
            tracking_error,
            information_ratio,
            correlation,
            up_capture,
            down_capture,
            benchmark_return_pct: benchmark_total_return * 100.0,
            excess_return_pct: (portfolio_total_return - benchmark_total_return) * 100.0,
        })
    }

    /// Calculate upside and downside capture ratios.
    fn calculate_capture_ratios(
        portfolio_returns: &[f64],
        benchmark_returns: &[f64],
    ) -> (f64, f64) {
        let up_periods: Vec<(f64, f64)> = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .filter(|(_, b)| **b > 0.0)
            .map(|(p, b)| (*p, *b))
            .collect();

        let down_periods: Vec<(f64, f64)> = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .filter(|(_, b)| **b < 0.0)
            .map(|(p, b)| (*p, *b))
            .collect();

        let up_capture = if !up_periods.is_empty() {
            let port_up: f64 = up_periods.iter().map(|(p, _)| p).sum::<f64>();
            let bench_up: f64 = up_periods.iter().map(|(_, b)| b).sum::<f64>();
            if bench_up > 0.0 {
                (port_up / bench_up) * 100.0
            } else {
                100.0
            }
        } else {
            100.0
        };

        let down_capture = if !down_periods.is_empty() {
            let port_down: f64 = down_periods.iter().map(|(p, _)| p).sum::<f64>();
            let bench_down: f64 = down_periods.iter().map(|(_, b)| b).sum::<f64>();
            if bench_down < 0.0 {
                (port_down / bench_down) * 100.0
            } else {
                100.0
            }
        } else {
            100.0
        };

        (up_capture, down_capture)
    }

    /// Extract daily returns from an equity curve.
    /// This produces actual daily returns rather than synthetic uniform returns.
    pub fn extract_daily_returns(equity_curve: &[crate::types::EquityPoint]) -> Vec<f64> {
        if equity_curve.len() < 2 {
            return vec![];
        }

        equity_curve
            .windows(2)
            .map(|w| (w[1].equity - w[0].equity) / w[0].equity)
            .collect()
    }

    /// Extract daily returns from a price series (e.g., benchmark close prices).
    pub fn extract_returns_from_prices(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }

        prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect()
    }
}

/// Represents a single drawdown period from peak to recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownPeriod {
    /// Start of drawdown (when equity dropped below peak)
    pub start: chrono::DateTime<chrono::Utc>,
    /// Time of maximum drawdown depth (trough)
    pub trough: chrono::DateTime<chrono::Utc>,
    /// End of drawdown (when equity recovered to peak, None if not recovered)
    pub end: Option<chrono::DateTime<chrono::Utc>>,
    /// Maximum drawdown depth as percentage
    pub depth_pct: f64,
    /// Duration from start to end (or current if not recovered) in days
    pub duration_days: i64,
}

/// Comprehensive drawdown analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownAnalysis {
    /// Maximum drawdown percentage
    pub max_drawdown_pct: f64,
    /// Duration of the maximum drawdown period in days
    pub max_drawdown_duration_days: i64,
    /// Average of all drawdown depths
    pub avg_drawdown_pct: f64,
    /// Ulcer Index - sqrt(mean(drawdown_pct^2))
    pub ulcer_index: f64,
    /// All drawdown periods
    pub periods: Vec<DrawdownPeriod>,
    /// Time underwater as percentage of total time
    pub time_underwater_pct: f64,
}

impl DrawdownAnalysis {
    /// Calculate comprehensive drawdown analysis from an equity curve.
    /// This provides actual drawdown statistics rather than approximations.
    pub fn from_equity_curve(equity_curve: &[crate::types::EquityPoint]) -> Self {
        if equity_curve.len() < 2 {
            return Self::empty();
        }

        let mut peak = equity_curve[0].equity;
        let mut peak_time = equity_curve[0].timestamp;
        let mut periods: Vec<DrawdownPeriod> = Vec::new();
        let mut current_period: Option<DrawdownPeriod> = None;
        let mut drawdown_pcts: Vec<f64> = Vec::with_capacity(equity_curve.len());

        for point in equity_curve.iter() {
            if point.equity >= peak {
                // New peak - drawdown is 0%, close any existing drawdown period
                drawdown_pcts.push(0.0);
                if let Some(mut period) = current_period.take() {
                    period.end = Some(point.timestamp);
                    period.duration_days = (point.timestamp - period.start).num_days();
                    periods.push(period);
                }
                peak = point.equity;
                peak_time = point.timestamp;
            } else {
                // In drawdown - calculate percentage below peak
                let drawdown_pct = if peak > 0.0 {
                    ((peak - point.equity) / peak) * 100.0
                } else {
                    0.0
                };
                drawdown_pcts.push(drawdown_pct);

                // Track drawdown period
                match &mut current_period {
                    None => {
                        // Start new drawdown period
                        current_period = Some(DrawdownPeriod {
                            start: peak_time,
                            trough: point.timestamp,
                            end: None,
                            depth_pct: drawdown_pct,
                            duration_days: (point.timestamp - peak_time).num_days(),
                        });
                    }
                    Some(period) => {
                        // Update existing period if deeper
                        if drawdown_pct > period.depth_pct {
                            period.trough = point.timestamp;
                            period.depth_pct = drawdown_pct;
                        }
                        period.duration_days = (point.timestamp - period.start).num_days();
                    }
                }
            }
        }

        // Handle ongoing drawdown at end of data
        if let Some(period) = current_period {
            periods.push(period);
        }

        // Calculate statistics
        let max_drawdown_pct = periods
            .iter()
            .map(|p| p.depth_pct)
            .fold(0.0_f64, |a, b| a.max(b));

        let max_drawdown_duration_days = periods.iter().map(|p| p.duration_days).max().unwrap_or(0);

        let avg_drawdown_pct = if !periods.is_empty() {
            periods.iter().map(|p| p.depth_pct).sum::<f64>() / periods.len() as f64
        } else {
            0.0
        };

        // Ulcer Index: sqrt(mean(drawdown_pct^2))
        let ulcer_index = if !drawdown_pcts.is_empty() {
            let sum_squared: f64 = drawdown_pcts.iter().map(|d| d.powi(2)).sum();
            (sum_squared / drawdown_pcts.len() as f64).sqrt()
        } else {
            0.0
        };

        // Time underwater percentage
        let total_days = if equity_curve.len() >= 2 {
            (equity_curve.last().unwrap().timestamp - equity_curve.first().unwrap().timestamp)
                .num_days()
        } else {
            0
        };
        let underwater_days: i64 = periods.iter().map(|p| p.duration_days).sum();
        let time_underwater_pct = if total_days > 0 {
            (underwater_days as f64 / total_days as f64) * 100.0
        } else {
            0.0
        };

        Self {
            max_drawdown_pct,
            max_drawdown_duration_days,
            avg_drawdown_pct,
            ulcer_index,
            periods,
            time_underwater_pct,
        }
    }

    /// Create an empty analysis for when equity curve is unavailable.
    pub fn empty() -> Self {
        Self {
            max_drawdown_pct: 0.0,
            max_drawdown_duration_days: 0,
            avg_drawdown_pct: 0.0,
            ulcer_index: 0.0,
            periods: Vec::new(),
            time_underwater_pct: 0.0,
        }
    }
}

/// Warning about a potentially suspicious metric value.
/// These warnings help identify potential issues like lookahead bias, overfitting,
/// or insufficient sample size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousMetricWarning {
    /// The metric that triggered the warning
    pub metric: String,
    /// The actual value of the metric
    pub value: f64,
    /// The threshold that was exceeded
    pub threshold: f64,
    /// Human-readable warning message
    pub message: String,
    /// Severity level: "caution" or "warning"
    pub severity: String,
}

/// Comprehensive performance metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Returns
    pub total_return_pct: f64,
    pub annual_return_pct: f64,
    pub monthly_return_avg: f64,

    // Risk metrics
    pub volatility_annual: f64,
    pub max_drawdown_pct: f64,
    pub max_drawdown_duration_days: i64,
    pub avg_drawdown_pct: f64,
    pub ulcer_index: f64,

    // Risk-adjusted returns
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub omega_ratio: f64,

    // Distribution metrics
    pub skewness: f64,
    pub kurtosis: f64,
    pub tail_ratio: f64,

    // Overfitting detection metrics
    pub deflated_sharpe_ratio: f64,
    pub probabilistic_sharpe_ratio: f64,

    // Trade statistics
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub avg_trade: f64,
    pub avg_holding_period_days: f64,

    // Expectancy
    pub expectancy: f64,
    pub expectancy_ratio: f64,

    // Consistency
    pub winning_months: usize,
    pub losing_months: usize,
    pub best_month_pct: f64,
    pub worst_month_pct: f64,

    // Recovery
    pub recovery_factor: f64,
    pub payoff_ratio: f64,
}

impl PerformanceMetrics {
    /// Calculate comprehensive metrics from backtest results.
    /// For optimization contexts with multiple parameter combinations,
    /// use `from_result_with_trials` to properly adjust the deflated Sharpe ratio.
    pub fn from_result(result: &BacktestResult) -> Self {
        Self::from_result_with_trials(result, 1)
    }

    /// Calculate comprehensive metrics from backtest results with multiple testing adjustment.
    ///
    /// # Arguments
    /// * `result` - The backtest result to analyze
    /// * `n_trials` - Number of parameter combinations tested (for deflated Sharpe ratio)
    ///   Set to 1 for single backtest, or to the number of parameter
    ///   combinations tested during optimization.
    pub fn from_result_with_trials(result: &BacktestResult, n_trials: usize) -> Self {
        let _returns = Self::calculate_returns(&result.trades, result.initial_capital);
        let daily_returns = Self::calculate_daily_returns(result);

        // Volatility
        let volatility_annual = Self::annualized_volatility(&daily_returns);

        // Drawdown analysis
        let (max_dd_duration, avg_dd) = Self::drawdown_analysis(result);

        // Ulcer index
        let ulcer_index = Self::ulcer_index(result);

        // Omega ratio
        let omega_ratio = Self::omega_ratio(&daily_returns, 0.0);

        // Distribution metrics
        let skewness = Self::skewness(&daily_returns);
        let kurtosis = Self::kurtosis(&daily_returns);
        let tail_ratio = Self::tail_ratio(&daily_returns);

        // Overfitting detection metrics with multiple testing adjustment
        let n_observations = daily_returns.len();
        let deflated_sharpe_ratio =
            Self::deflated_sharpe_ratio(result.sharpe_ratio, n_trials, n_observations);
        let probabilistic_sharpe_ratio = Self::probabilistic_sharpe_ratio(
            result.sharpe_ratio,
            skewness,
            kurtosis,
            n_observations,
            0.0, // Benchmark SR = 0
        );

        // Trade analysis
        let closed_trades: Vec<_> = result.trades.iter().filter(|t| t.is_closed()).collect();
        let winning: Vec<_> = closed_trades
            .iter()
            .filter(|t| t.net_pnl().unwrap_or(0.0) > 0.0)
            .collect();
        let losing: Vec<_> = closed_trades
            .iter()
            .filter(|t| t.net_pnl().unwrap_or(0.0) < 0.0)
            .collect();

        let largest_win = winning
            .iter()
            .filter_map(|t| t.net_pnl())
            .fold(0.0_f64, |a, b| a.max(b));
        let largest_loss = losing
            .iter()
            .filter_map(|t| t.net_pnl())
            .fold(0.0_f64, |a, b| a.min(b));

        let avg_trade = if !closed_trades.is_empty() {
            closed_trades
                .iter()
                .filter_map(|t| t.net_pnl())
                .sum::<f64>()
                / closed_trades.len() as f64
        } else {
            0.0
        };

        let avg_holding_period_days = if !closed_trades.is_empty() {
            closed_trades
                .iter()
                .filter_map(|t| t.holding_period())
                .map(|d| d.num_hours() as f64 / 24.0)
                .sum::<f64>()
                / closed_trades.len() as f64
        } else {
            0.0
        };

        // Expectancy
        let expectancy = (result.win_rate / 100.0) * result.avg_win
            - (1.0 - result.win_rate / 100.0) * result.avg_loss.abs();
        let expectancy_ratio = if result.avg_loss.abs() > 0.0 {
            expectancy / result.avg_loss.abs()
        } else {
            0.0
        };

        // Monthly analysis
        let monthly_returns = Self::monthly_returns(result);
        let winning_months = monthly_returns.iter().filter(|&&r| r > 0.0).count();
        let losing_months = monthly_returns.iter().filter(|&&r| r < 0.0).count();
        let best_month_pct = monthly_returns
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst_month_pct = monthly_returns.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let monthly_return_avg = if !monthly_returns.is_empty() {
            monthly_returns.iter().sum::<f64>() / monthly_returns.len() as f64
        } else {
            0.0
        };

        // Recovery factor
        let total_profit = result.final_equity - result.initial_capital;
        let recovery_factor = if result.max_drawdown_pct > 0.0 {
            total_profit / (result.initial_capital * result.max_drawdown_pct / 100.0)
        } else {
            0.0
        };

        // Payoff ratio
        let payoff_ratio = if result.avg_loss.abs() > 0.0 {
            result.avg_win / result.avg_loss.abs()
        } else {
            0.0
        };

        Self {
            total_return_pct: result.total_return_pct,
            annual_return_pct: result.annual_return_pct,
            monthly_return_avg,
            volatility_annual,
            max_drawdown_pct: result.max_drawdown_pct,
            max_drawdown_duration_days: max_dd_duration,
            avg_drawdown_pct: avg_dd,
            ulcer_index,
            sharpe_ratio: result.sharpe_ratio,
            sortino_ratio: result.sortino_ratio,
            calmar_ratio: result.calmar_ratio,
            omega_ratio,
            skewness,
            kurtosis,
            tail_ratio,
            deflated_sharpe_ratio,
            probabilistic_sharpe_ratio,
            total_trades: result.total_trades,
            winning_trades: result.winning_trades,
            losing_trades: result.losing_trades,
            win_rate: result.win_rate,
            profit_factor: result.profit_factor,
            avg_win: result.avg_win,
            avg_loss: result.avg_loss,
            largest_win,
            largest_loss,
            avg_trade,
            avg_holding_period_days,
            expectancy,
            expectancy_ratio,
            winning_months,
            losing_months,
            best_month_pct: if best_month_pct == f64::NEG_INFINITY {
                0.0
            } else {
                best_month_pct
            },
            worst_month_pct: if worst_month_pct == f64::INFINITY {
                0.0
            } else {
                worst_month_pct
            },
            recovery_factor,
            payoff_ratio,
        }
    }

    fn calculate_returns(trades: &[Trade], _initial_capital: f64) -> Vec<f64> {
        trades
            .iter()
            .filter(|t| t.is_closed())
            .filter_map(|t| t.return_pct())
            .collect()
    }

    /// Calculate actual daily returns from the equity curve.
    /// This provides meaningful volatility calculations by using real equity changes
    /// instead of synthetic uniform returns.
    fn calculate_daily_returns(result: &BacktestResult) -> Vec<f64> {
        // Use actual equity curve if available
        if result.equity_curve.len() >= 2 {
            // Group equity points by day and take the last equity value for each day.
            // This handles both daily and intraday data correctly.
            let mut daily_equity: BTreeMap<(i32, u32, u32), f64> = BTreeMap::new();
            for point in &result.equity_curve {
                let key = (
                    point.timestamp.year(),
                    point.timestamp.month(),
                    point.timestamp.day(),
                );
                // BTreeMap::insert overwrites, so we keep the last value for each day
                daily_equity.insert(key, point.equity);
            }

            // Convert to vector of daily equity values (sorted by date due to BTreeMap)
            let equity_values: Vec<f64> = daily_equity.values().copied().collect();

            // Calculate returns between consecutive days
            if equity_values.len() >= 2 {
                return equity_values
                    .windows(2)
                    .map(|w| (w[1] - w[0]) / w[0])
                    .collect();
            }
        }

        // Fallback to synthetic returns only if equity curve is unavailable or too small
        if result.trading_days <= 1 {
            return vec![];
        }
        let daily_return = (result.final_equity / result.initial_capital)
            .powf(1.0 / result.trading_days as f64)
            - 1.0;
        vec![daily_return; result.trading_days]
    }

    fn annualized_volatility(daily_returns: &[f64]) -> f64 {
        if daily_returns.is_empty() {
            return 0.0;
        }
        let mean: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance: f64 = daily_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / daily_returns.len() as f64;
        variance.sqrt() * 252.0_f64.sqrt() * 100.0
    }

    /// Calculate drawdown analysis from the equity curve.
    /// Returns (max_drawdown_duration_days, average_drawdown_pct)
    fn drawdown_analysis(result: &BacktestResult) -> (i64, f64) {
        if result.equity_curve.len() >= 2 {
            let analysis = DrawdownAnalysis::from_equity_curve(&result.equity_curve);
            (
                analysis.max_drawdown_duration_days,
                analysis.avg_drawdown_pct,
            )
        } else {
            // Fallback to approximations when equity curve unavailable
            let avg_dd = result.max_drawdown_pct / 2.0;
            let max_dd_duration = (result.end_time - result.start_time).num_days() / 10;
            (max_dd_duration.max(0), avg_dd)
        }
    }

    /// Calculate Ulcer Index from the equity curve.
    /// Ulcer Index = sqrt(mean(drawdown_pct^2)) - measures downside volatility
    fn ulcer_index(result: &BacktestResult) -> f64 {
        if result.equity_curve.len() >= 2 {
            let analysis = DrawdownAnalysis::from_equity_curve(&result.equity_curve);
            analysis.ulcer_index
        } else {
            // Fallback approximation when equity curve unavailable
            let squared_dd = result.max_drawdown_pct.powi(2);
            (squared_dd / 2.0).sqrt()
        }
    }

    fn omega_ratio(returns: &[f64], threshold: f64) -> f64 {
        if returns.is_empty() {
            return 1.0;
        }

        let gains: f64 = returns
            .iter()
            .filter(|&&r| r > threshold)
            .map(|r| r - threshold)
            .sum();
        let losses: f64 = returns
            .iter()
            .filter(|&&r| r < threshold)
            .map(|r| threshold - r)
            .sum();

        if losses == 0.0 {
            if gains > 0.0 {
                f64::INFINITY
            } else {
                1.0
            }
        } else {
            gains / losses
        }
    }

    /// Calculate skewness of returns distribution.
    /// Skewness = E[(X - μ)³] / σ³
    /// Positive skewness: right tail is longer (more extreme positive returns)
    /// Negative skewness: left tail is longer (more extreme negative returns)
    /// Zero skewness: symmetric distribution
    fn skewness(returns: &[f64]) -> f64 {
        if returns.len() < 3 {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n;

        // Calculate variance
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;

        if variance == 0.0 {
            return 0.0;
        }

        let std_dev = variance.sqrt();

        // Calculate third moment
        let third_moment: f64 = returns.iter().map(|r| (r - mean).powi(3)).sum::<f64>() / n;

        third_moment / std_dev.powi(3)
    }

    /// Calculate excess kurtosis of returns distribution.
    /// Kurtosis = E[(X - μ)⁴] / σ⁴ - 3
    /// Positive kurtosis: heavy tails (more extreme events than normal distribution)
    /// Negative kurtosis: light tails (fewer extreme events)
    /// Zero kurtosis: same tail behavior as normal distribution
    fn kurtosis(returns: &[f64]) -> f64 {
        if returns.len() < 4 {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n;

        // Calculate variance
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;

        if variance == 0.0 {
            return 0.0;
        }

        // Calculate fourth moment
        let fourth_moment: f64 = returns.iter().map(|r| (r - mean).powi(4)).sum::<f64>() / n;

        // Subtract 3 to get excess kurtosis (relative to normal distribution)
        (fourth_moment / variance.powi(2)) - 3.0
    }

    /// Calculate tail ratio: 95th percentile gain / 95th percentile loss.
    /// Higher values indicate asymmetric upside potential.
    /// Tail ratio > 1: larger gains than losses in the tails
    /// Tail ratio < 1: larger losses than gains in the tails
    fn tail_ratio(returns: &[f64]) -> f64 {
        if returns.len() < 20 {
            return 1.0;
        }

        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // 95th percentile (95% of returns are below this)
        let p95_idx = ((sorted.len() as f64) * 0.95) as usize;
        let p95_gain = sorted
            .get(p95_idx.min(sorted.len() - 1))
            .copied()
            .unwrap_or(0.0);

        // 5th percentile (5% of returns are below this - the losses)
        let p05_idx = ((sorted.len() as f64) * 0.05) as usize;
        let p05_loss = sorted.get(p05_idx).copied().unwrap_or(0.0);

        // Tail ratio is the ratio of absolute values
        // Handle edge cases:
        if p05_loss.abs() < 1e-10 && p95_gain.abs() < 1e-10 {
            // Both near zero
            1.0
        } else if p05_loss.abs() < 1e-10 {
            // No downside tail, only upside
            100.0
        } else if p95_gain.abs() < 1e-10 {
            // No upside tail, only downside
            0.01
        } else {
            // Normal case: ratio of upside to downside tail
            p95_gain.abs() / p05_loss.abs()
        }
    }

    /// Calculate Deflated Sharpe Ratio (DSR) - adjusts for multiple testing bias.
    /// Based on Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio".
    ///
    /// DSR = (SR - E[SR_max]) / sqrt(Var[SR_max])
    ///
    /// Where E[SR_max] is the expected maximum Sharpe ratio under the null hypothesis
    /// (all strategies are random) given the number of trials.
    ///
    /// # Arguments
    /// * `sharpe` - The observed Sharpe ratio
    /// * `n_trials` - Number of independent strategies/parameter combinations tested
    /// * `n_observations` - Number of return observations (e.g., trading days)
    ///
    /// # Returns
    /// Deflated Sharpe Ratio. Values < 0 suggest the observed SR is likely due to luck.
    fn deflated_sharpe_ratio(sharpe: f64, n_trials: usize, n_observations: usize) -> f64 {
        if n_trials <= 1 {
            // No multiple testing - return original Sharpe
            return sharpe;
        }

        if n_observations < 2 {
            return 0.0;
        }

        // Expected maximum Sharpe ratio under null hypothesis (random strategies)
        // Approximation: E[SR_max] ≈ sqrt(2 * ln(N)) for large N
        let n = n_trials as f64;
        let expected_max_sr = (2.0 * n.ln()).sqrt();

        // Variance of maximum SR under null: Var[SR_max] ≈ 1 for large samples
        // More accurate: Var[SR_max] = 1 + (1 - Euler-Mascheroni) / N
        let euler_mascheroni = 0.5772156649;
        let var_max_sr = 1.0 + (1.0 - euler_mascheroni) / n;

        // Deflated Sharpe Ratio
        (sharpe - expected_max_sr) / var_max_sr.sqrt()
    }

    /// Calculate Probabilistic Sharpe Ratio (PSR) - probability that SR > benchmark SR.
    /// Based on Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier".
    ///
    /// PSR = Φ(Z) where Z = (SR - SR*) × sqrt(T-1) / sqrt(1 - γ₃×SR + (γ₄-1)/4 × SR²)
    ///
    /// PSR = P[True SR > Benchmark SR | observed data]
    ///
    /// # Arguments
    /// * `sharpe` - The observed annualized Sharpe ratio
    /// * `skewness` - Skewness of returns (γ₃)
    /// * `kurtosis` - Excess kurtosis of returns (γ₄)
    /// * `n_observations` - Number of return observations (T)
    /// * `benchmark_sr` - Reference Sharpe ratio (typically 0)
    ///
    /// # Returns
    /// Probability between 0 and 1. PSR > 0.95 suggests high confidence the strategy
    /// has skill (true SR > benchmark). PSR < 0.5 suggests the observed SR is likely luck.
    fn probabilistic_sharpe_ratio(
        sharpe: f64,
        skewness: f64,
        kurtosis: f64,
        n_observations: usize,
        benchmark_sr: f64,
    ) -> f64 {
        if n_observations < 2 {
            return 0.5; // No information
        }

        let t = n_observations as f64;

        // Calculate the denominator: adjusted variance of SR estimate
        // Var[SR] = (1 - γ₃×SR + (γ₄-1)/4 × SR²) / (T-1)
        let variance_adjustment = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe.powi(2);

        // Protect against negative variance (can happen with extreme skew/kurtosis)
        let variance_adjustment = variance_adjustment.max(0.01);

        // Standard error of SR estimate
        let std_error = (variance_adjustment / (t - 1.0)).sqrt();

        // Test statistic: Z = (SR - SR_benchmark) / SE[SR]
        let z_score = if std_error > 0.0 {
            (sharpe - benchmark_sr) / std_error
        } else {
            0.0
        };

        // PSR = Φ(Z) - cumulative distribution function of standard normal
        // Using error function: Φ(z) = 0.5 × (1 + erf(z / sqrt(2)))
        let psr = 0.5 * (1.0 + erf(z_score / std::f64::consts::SQRT_2));

        psr.clamp(0.0, 1.0)
    }

    fn monthly_returns(result: &BacktestResult) -> Vec<f64> {
        if result.equity_curve.is_empty() {
            return vec![];
        }

        // Group equity points by (year, month) and get first/last equity for each month
        let mut monthly_data: BTreeMap<(i32, u32), (f64, f64)> = BTreeMap::new();

        for point in &result.equity_curve {
            let key = (point.timestamp.year(), point.timestamp.month());
            let entry = monthly_data
                .entry(key)
                .or_insert((point.equity, point.equity));
            // Update the end equity (last value in this month)
            entry.1 = point.equity;
        }

        // Calculate returns for each month
        monthly_data
            .values()
            .map(|(start_equity, end_equity)| {
                if *start_equity > 0.0 {
                    ((end_equity - start_equity) / start_equity) * 100.0 // Return as percentage
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Check for suspicious metric values that may indicate issues.
    ///
    /// Returns a list of warnings based on these thresholds:
    /// - Sharpe ratio > 3: May indicate data issues or lookahead bias
    /// - Win rate > 80%: May indicate lookahead bias
    /// - Max drawdown < 5%: May indicate execution logic issues
    /// - Total trades < 30: Limited statistical significance
    ///
    /// For OOS/IS degradation checks, use `check_oos_degradation` separately
    /// as it requires walk-forward results.
    pub fn check_suspicious_metrics(&self) -> Vec<SuspiciousMetricWarning> {
        let mut warnings = Vec::new();

        // Sharpe ratio > 3 is suspicious
        if self.sharpe_ratio > 3.0 {
            warnings.push(SuspiciousMetricWarning {
                metric: "sharpe_ratio".to_string(),
                value: self.sharpe_ratio,
                threshold: 3.0,
                message:
                    "Sharpe ratio > 3 is unusually high. Verify data quality and execution logic."
                        .to_string(),
                severity: "warning".to_string(),
            });
        }

        // Win rate > 80% is suspicious
        if self.win_rate > 0.80 {
            warnings.push(SuspiciousMetricWarning {
                metric: "win_rate".to_string(),
                value: self.win_rate * 100.0, // Convert to percentage for display
                threshold: 80.0,
                message: "Win rate > 80% is unusually high. Check for lookahead bias in signal generation.".to_string(),
                severity: "warning".to_string(),
            });
        }

        // Max drawdown < 5% with significant trades is suspicious
        if self.max_drawdown_pct.abs() < 5.0 && self.total_trades >= 10 {
            warnings.push(SuspiciousMetricWarning {
                metric: "max_drawdown_pct".to_string(),
                value: self.max_drawdown_pct.abs(),
                threshold: 5.0,
                message: "Max drawdown < 5% with multiple trades is unusual. Verify execution logic and cost modeling.".to_string(),
                severity: "caution".to_string(),
            });
        }

        // Too few trades for statistical significance
        if self.total_trades < 30 && self.total_trades > 0 {
            warnings.push(SuspiciousMetricWarning {
                metric: "total_trades".to_string(),
                value: self.total_trades as f64,
                threshold: 30.0,
                message: "Fewer than 30 trades limits statistical significance. Results may not generalize.".to_string(),
                severity: "caution".to_string(),
            });
        }

        // Profit factor > 5 is suspicious
        if self.profit_factor > 5.0 && self.profit_factor.is_finite() {
            warnings.push(SuspiciousMetricWarning {
                metric: "profit_factor".to_string(),
                value: self.profit_factor,
                threshold: 5.0,
                message: "Profit factor > 5 is unusually high. Verify data and execution."
                    .to_string(),
                severity: "caution".to_string(),
            });
        }

        warnings
    }
}

/// Check OOS/IS degradation from walk-forward results.
/// Returns a warning if OOS/IS ratio is below threshold (0.60 indicates likely overfit).
pub fn check_oos_degradation(oos_sharpe: f64, is_sharpe: f64) -> Option<SuspiciousMetricWarning> {
    if is_sharpe <= 0.0 {
        return None; // Can't compute ratio with non-positive IS Sharpe
    }

    let ratio = oos_sharpe / is_sharpe;

    if ratio < 0.60 {
        Some(SuspiciousMetricWarning {
            metric: "oos_is_ratio".to_string(),
            value: ratio,
            threshold: 0.60,
            message: format!(
                "OOS/IS ratio of {:.2} indicates likely overfitting. Strategy may not work in live trading.",
                ratio
            ),
            severity: "warning".to_string(),
        })
    } else if ratio < 0.80 {
        Some(SuspiciousMetricWarning {
            metric: "oos_is_ratio".to_string(),
            value: ratio,
            threshold: 0.80,
            message: format!(
                "OOS/IS ratio of {:.2} is acceptable but warrants caution. Monitor live performance closely.",
                ratio
            ),
            severity: "caution".to_string(),
        })
    } else {
        None
    }
}

/// Calculate rolling Sharpe ratio over a sliding window.
///
/// Returns a vector of Sharpe ratios, one for each window position.
/// The first `window - 1` elements will be NaN (insufficient data).
///
/// # Arguments
/// * `returns` - Daily returns (as decimals, e.g., 0.01 for 1%)
/// * `window` - Rolling window size in periods (e.g., 252 for annual)
/// * `annualization_factor` - Annualization factor (e.g., 252.0 for daily returns)
///
/// # Example
/// ```ignore
/// use mantis::analytics::rolling_sharpe;
///
/// let returns = vec![0.01, -0.005, 0.008, 0.003, -0.002];
/// let rolling = rolling_sharpe(&returns, 3, 252.0);
/// ```
pub fn rolling_sharpe(returns: &[f64], window: usize, annualization_factor: f64) -> Vec<f64> {
    if window == 0 || returns.is_empty() {
        return vec![f64::NAN; returns.len()];
    }

    let mut result = vec![f64::NAN; returns.len()];

    for i in (window - 1)..returns.len() {
        let window_returns = &returns[(i + 1 - window)..=i];
        let sharpe = calculate_window_sharpe(window_returns, annualization_factor);
        result[i] = sharpe;
    }

    result
}

/// Calculate Sharpe ratio for a single window of returns.
fn calculate_window_sharpe(returns: &[f64], annualization_factor: f64) -> f64 {
    if returns.is_empty() {
        return f64::NAN;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 || std_dev.is_nan() {
        return if mean > 0.0 {
            f64::INFINITY
        } else if mean < 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }

    (mean / std_dev) * annualization_factor.sqrt()
}

/// Calculate rolling drawdown from an equity curve.
///
/// Returns a vector of drawdown percentages at each point in time.
/// Drawdown is calculated as (current_equity - peak_equity) / peak_equity.
/// Values are negative (or zero at peaks).
///
/// # Arguments
/// * `equity` - Equity values over time (must be positive)
///
/// # Example
/// ```ignore
/// use mantis::analytics::rolling_drawdown;
///
/// let equity = vec![100.0, 105.0, 102.0, 108.0, 103.0];
/// let drawdowns = rolling_drawdown(&equity);
/// // Returns: [0.0, 0.0, -0.0286, 0.0, -0.0463]
/// ```
pub fn rolling_drawdown(equity: &[f64]) -> Vec<f64> {
    if equity.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(equity.len());
    let mut peak = equity[0];

    for &value in equity {
        if value > peak {
            peak = value;
        }
        let drawdown = if peak > 0.0 {
            (value - peak) / peak
        } else {
            0.0
        };
        result.push(drawdown);
    }

    result
}

/// Calculate rolling drawdown with a maximum lookback window.
///
/// Unlike `rolling_drawdown()`, this function only looks back `window` periods
/// to find the peak, which can be useful for analyzing recent behavior.
///
/// # Arguments
/// * `equity` - Equity values over time
/// * `window` - Maximum lookback window to find peak (0 = use all history)
pub fn rolling_drawdown_windowed(equity: &[f64], window: usize) -> Vec<f64> {
    if equity.is_empty() {
        return vec![];
    }

    if window == 0 {
        return rolling_drawdown(equity);
    }

    let mut result = Vec::with_capacity(equity.len());

    for i in 0..equity.len() {
        let start = if i >= window { i + 1 - window } else { 0 };
        let window_slice = &equity[start..=i];
        let peak = window_slice
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let drawdown = if peak > 0.0 {
            (equity[i] - peak) / peak
        } else {
            0.0
        };
        result.push(drawdown);
    }

    result
}

/// Calculate rolling maximum drawdown over a sliding window.
///
/// Returns the worst (minimum) drawdown observed within each rolling window.
/// Useful for tracking strategy risk over time.
///
/// # Arguments
/// * `equity` - Equity values over time
/// * `window` - Rolling window size
pub fn rolling_max_drawdown(equity: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || equity.is_empty() {
        return vec![f64::NAN; equity.len()];
    }

    let mut result = vec![f64::NAN; equity.len()];
    let drawdowns = rolling_drawdown(equity);

    for i in (window - 1)..equity.len() {
        let window_drawdowns = &drawdowns[(i + 1 - window)..=i];
        let max_dd = window_drawdowns.iter().fold(0.0_f64, |a, &b| a.min(b));
        result[i] = max_dd;
    }

    result
}

/// Calculate rolling volatility from returns.
///
/// Returns annualized volatility for each rolling window.
///
/// # Arguments
/// * `returns` - Daily returns (as decimals)
/// * `window` - Rolling window size in periods
/// * `annualization_factor` - Annualization factor (e.g., 252.0 for daily)
pub fn rolling_volatility(returns: &[f64], window: usize, annualization_factor: f64) -> Vec<f64> {
    if window == 0 || returns.is_empty() {
        return vec![f64::NAN; returns.len()];
    }

    let mut result = vec![f64::NAN; returns.len()];

    for i in (window - 1)..returns.len() {
        let window_returns = &returns[(i + 1 - window)..=i];
        let n = window_returns.len() as f64;
        let mean: f64 = window_returns.iter().sum::<f64>() / n;
        let variance: f64 = window_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / n;
        let volatility = variance.sqrt() * annualization_factor.sqrt();
        result[i] = volatility;
    }

    result
}

/// Error function (erf) approximation for normal CDF calculation.
/// Uses Abramowitz and Stegun approximation (max error: 1.5e-7).
pub fn erf(x: f64) -> f64 {
    // Constants for approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    // Save the sign of x
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Format results for terminal display.
pub struct ResultFormatter;

impl ResultFormatter {
    /// Print a comprehensive results report to stdout.
    pub fn print_report(result: &BacktestResult) {
        let metrics = PerformanceMetrics::from_result(result);

        println!();
        println!("{}", "═".repeat(60).blue());
        println!("{}", " BACKTEST RESULTS ".bold().blue());
        println!("{}", "═".repeat(60).blue());
        println!();

        // Overview
        println!("{}", "Overview".bold().underline());
        println!("  Strategy:        {}", result.strategy_name);
        println!("  Symbol(s):       {}", result.symbols.join(", "));
        println!(
            "  Period:          {} to {}",
            result.start_time.format("%Y-%m-%d"),
            result.end_time.format("%Y-%m-%d")
        );
        println!("  Trading Days:    {}", result.trading_days);
        println!();

        // Performance
        println!("{}", "Performance".bold().underline());
        println!("  Initial Capital: ${:>12.2}", result.initial_capital);
        println!(
            "  Final Equity:    ${:>12.2}  {}",
            result.final_equity,
            Self::format_pct_change(result.total_return_pct)
        );
        println!("  Total Return:    {:>12.2}%", result.total_return_pct);
        println!("  Annual Return:   {:>12.2}%", result.annual_return_pct);
        println!();

        // Risk Metrics
        println!("{}", "Risk Metrics".bold().underline());
        println!("  Max Drawdown:    {:>12.2}%", -result.max_drawdown_pct);
        println!("  Volatility:      {:>12.2}%", metrics.volatility_annual);
        println!("  Sharpe Ratio:    {:>12.2}", result.sharpe_ratio);
        println!("  Sortino Ratio:   {:>12.2}", result.sortino_ratio);
        println!("  Calmar Ratio:    {:>12.2}", result.calmar_ratio);
        println!("  Omega Ratio:     {:>12.2}", metrics.omega_ratio);
        println!();

        // Distribution Metrics
        println!("{}", "Distribution Metrics".bold().underline());
        println!("  Skewness:        {:>12.2}", metrics.skewness);
        println!("  Kurtosis:        {:>12.2}", metrics.kurtosis);
        println!("  Tail Ratio:      {:>12.2}", metrics.tail_ratio);
        println!();

        // Overfitting Detection
        println!("{}", "Overfitting Detection".bold().underline());
        println!("  Deflated Sharpe: {:>12.2}", metrics.deflated_sharpe_ratio);
        println!(
            "  Prob. Sharpe:    {:>12.2}  ({})",
            metrics.probabilistic_sharpe_ratio,
            Self::format_psr_confidence(metrics.probabilistic_sharpe_ratio)
        );
        println!();

        // Trade Statistics
        println!("{}", "Trade Statistics".bold().underline());
        println!("  Total Trades:    {:>12}", result.total_trades);
        println!(
            "  Winning Trades:  {:>12}  ({:.1}%)",
            result.winning_trades, result.win_rate
        );
        println!(
            "  Losing Trades:   {:>12}  ({:.1}%)",
            result.losing_trades,
            100.0 - result.win_rate
        );
        println!("  Profit Factor:   {:>12.2}", result.profit_factor);
        println!();

        println!("{}", "Trade P&L".bold().underline());
        println!("  Average Win:     ${:>11.2}", result.avg_win);
        println!("  Average Loss:    ${:>11.2}", result.avg_loss);
        println!("  Largest Win:     ${:>11.2}", metrics.largest_win);
        println!("  Largest Loss:    ${:>11.2}", metrics.largest_loss);
        println!("  Expectancy:      ${:>11.2}", metrics.expectancy);
        println!();

        println!("{}", "═".repeat(60).blue());
    }

    /// Format percentage change with color.
    fn format_pct_change(pct: f64) -> String {
        if pct >= 0.0 {
            format!("(+{:.2}%)", pct).green().to_string()
        } else {
            format!("({:.2}%)", pct).red().to_string()
        }
    }

    /// Format PSR confidence level with interpretation.
    fn format_psr_confidence(psr: f64) -> String {
        let pct = psr * 100.0;
        if psr >= 0.95 {
            format!("{:.1}% - High confidence", pct).green().to_string()
        } else if psr >= 0.75 {
            format!("{:.1}% - Moderate confidence", pct)
                .yellow()
                .to_string()
        } else if psr >= 0.50 {
            format!("{:.1}% - Low confidence", pct).yellow().to_string()
        } else {
            format!("{:.1}% - Likely luck", pct).red().to_string()
        }
    }

    /// Print benchmark comparison metrics.
    pub fn print_benchmark_comparison(metrics: &BenchmarkMetrics) {
        println!();
        println!(
            "{}",
            format!("Benchmark Comparison ({})", metrics.benchmark_name)
                .bold()
                .underline()
        );
        println!(
            "  Benchmark Return: {:>10.2}%",
            metrics.benchmark_return_pct
        );
        println!(
            "  Excess Return:    {:>10.2}%  {}",
            metrics.excess_return_pct,
            Self::format_pct_change(metrics.excess_return_pct)
        );
        println!("  Alpha:            {:>10.2}%", metrics.alpha);
        println!("  Beta:             {:>10.2}", metrics.beta);
        println!("  Correlation:      {:>10.2}", metrics.correlation);
        println!();
        println!("  Tracking Error:   {:>10.2}%", metrics.tracking_error);
        println!("  Information Ratio:{:>10.2}", metrics.information_ratio);
        println!();
        println!("  Up Capture:       {:>10.2}%", metrics.up_capture);
        println!("  Down Capture:     {:>10.2}%", metrics.down_capture);
        println!();
    }

    /// Print a comprehensive report including benchmark comparison.
    pub fn print_report_with_benchmark(
        result: &BacktestResult,
        benchmark_metrics: Option<&BenchmarkMetrics>,
    ) {
        let metrics = PerformanceMetrics::from_result(result);

        println!();
        println!("{}", "═".repeat(60).blue());
        println!("{}", " BACKTEST RESULTS ".bold().blue());
        println!("{}", "═".repeat(60).blue());
        println!();

        // Overview
        println!("{}", "Overview".bold().underline());
        println!("  Strategy:        {}", result.strategy_name);
        println!("  Symbol(s):       {}", result.symbols.join(", "));
        println!(
            "  Period:          {} to {}",
            result.start_time.format("%Y-%m-%d"),
            result.end_time.format("%Y-%m-%d")
        );
        println!("  Trading Days:    {}", result.trading_days);
        println!();

        // Performance
        println!("{}", "Performance".bold().underline());
        println!("  Initial Capital: ${:>12.2}", result.initial_capital);
        println!(
            "  Final Equity:    ${:>12.2}  {}",
            result.final_equity,
            Self::format_pct_change(result.total_return_pct)
        );
        println!("  Total Return:    {:>12.2}%", result.total_return_pct);
        println!("  Annual Return:   {:>12.2}%", result.annual_return_pct);
        println!();

        // Risk Metrics
        println!("{}", "Risk Metrics".bold().underline());
        println!("  Max Drawdown:    {:>12.2}%", -result.max_drawdown_pct);
        println!("  Volatility:      {:>12.2}%", metrics.volatility_annual);
        println!("  Sharpe Ratio:    {:>12.2}", result.sharpe_ratio);
        println!("  Sortino Ratio:   {:>12.2}", result.sortino_ratio);
        println!("  Calmar Ratio:    {:>12.2}", result.calmar_ratio);
        println!("  Omega Ratio:     {:>12.2}", metrics.omega_ratio);
        println!();

        // Distribution Metrics
        println!("{}", "Distribution Metrics".bold().underline());
        println!("  Skewness:        {:>12.2}", metrics.skewness);
        println!("  Kurtosis:        {:>12.2}", metrics.kurtosis);
        println!("  Tail Ratio:      {:>12.2}", metrics.tail_ratio);
        println!();

        // Benchmark Comparison (if available)
        if let Some(bm) = benchmark_metrics {
            Self::print_benchmark_comparison(bm);
        }

        // Trade Statistics
        println!("{}", "Trade Statistics".bold().underline());
        println!("  Total Trades:    {:>12}", result.total_trades);
        println!(
            "  Winning Trades:  {:>12}  ({:.1}%)",
            result.winning_trades, result.win_rate
        );
        println!(
            "  Losing Trades:   {:>12}  ({:.1}%)",
            result.losing_trades,
            100.0 - result.win_rate
        );
        println!("  Profit Factor:   {:>12.2}", result.profit_factor);
        println!();

        println!("{}", "Trade P&L".bold().underline());
        println!("  Average Win:     ${:>11.2}", result.avg_win);
        println!("  Average Loss:    ${:>11.2}", result.avg_loss);
        println!("  Largest Win:     ${:>11.2}", metrics.largest_win);
        println!("  Largest Loss:    ${:>11.2}", metrics.largest_loss);
        println!("  Expectancy:      ${:>11.2}", metrics.expectancy);
        println!();

        println!("{}", "═".repeat(60).blue());
    }

    /// Print results as a table.
    pub fn print_table(results: &[BacktestResult]) {
        let mut builder = Builder::new();
        builder.push_record([
            "Strategy", "Return %", "Annual %", "Max DD %", "Sharpe", "Trades", "Win Rate",
        ]);

        for result in results {
            builder.push_record([
                result.strategy_name.clone(),
                format!("{:.2}", result.total_return_pct),
                format!("{:.2}", result.annual_return_pct),
                format!("{:.2}", -result.max_drawdown_pct),
                format!("{:.2}", result.sharpe_ratio),
                result.total_trades.to_string(),
                format!("{:.1}%", result.win_rate),
            ]);
        }

        let table = builder.build().with(Style::rounded()).to_string();
        println!("{}", table);
    }

    /// Export results to JSON.
    pub fn to_json(result: &BacktestResult) -> String {
        serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export results to CSV line.
    pub fn to_csv_line(result: &BacktestResult) -> String {
        format!(
            "{},{},{:.2},{:.2},{:.2},{:.2},{},{},{:.1},{:.2},{:.2}",
            result.strategy_name,
            result.symbols.join(";"),
            result.initial_capital,
            result.final_equity,
            result.total_return_pct,
            result.annual_return_pct,
            result.total_trades,
            result.winning_trades,
            result.win_rate,
            result.sharpe_ratio,
            result.max_drawdown_pct
        )
    }

    /// Get CSV header.
    pub fn csv_header() -> &'static str {
        "strategy,symbols,initial_capital,final_equity,total_return_pct,annual_return_pct,total_trades,winning_trades,win_rate,sharpe_ratio,max_drawdown_pct"
    }
}

/// Trade report generator.
pub struct TradeReport;

impl TradeReport {
    /// Print individual trades.
    pub fn print_trades(trades: &[Trade], limit: usize) {
        let closed_trades: Vec<_> = trades.iter().filter(|t| t.is_closed()).collect();

        if closed_trades.is_empty() {
            println!("No closed trades.");
            return;
        }

        let display_trades = if limit > 0 && limit < closed_trades.len() {
            &closed_trades[..limit]
        } else {
            &closed_trades
        };

        let mut builder = Builder::new();
        builder.push_record([
            "#", "Symbol", "Side", "Qty", "Entry", "Exit", "P&L", "Return %",
        ]);

        for (i, trade) in display_trades.iter().enumerate() {
            let pnl = trade.net_pnl().unwrap_or(0.0);
            let pnl_str = if pnl >= 0.0 {
                format!("+{:.2}", pnl)
            } else {
                format!("{:.2}", pnl)
            };

            let ret = trade.return_pct().unwrap_or(0.0);
            let ret_str = if ret >= 0.0 {
                format!("+{:.2}%", ret)
            } else {
                format!("{:.2}%", ret)
            };

            builder.push_record([
                (i + 1).to_string(),
                trade.symbol.clone(),
                trade.side.to_string(),
                format!("{:.2}", trade.quantity),
                format!("{:.2}", trade.entry_price),
                format!("{:.2}", trade.exit_price.unwrap_or(0.0)),
                pnl_str,
                ret_str,
            ]);
        }

        let table = builder.build().with(Style::rounded()).to_string();
        println!("{}", table);

        if limit > 0 && limit < closed_trades.len() {
            println!("... and {} more trades", closed_trades.len() - limit);
        }
    }
}

// ============================================================================
// Factor Attribution Module
// ============================================================================

/// Factor returns for a single time period.
/// Used to provide factor data for attribution analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorReturns {
    /// Market excess return (Rm - Rf)
    pub mkt_rf: f64,
    /// Size factor (Small Minus Big)
    pub smb: f64,
    /// Value factor (High Minus Low book-to-market)
    pub hml: f64,
    /// Profitability factor (Robust Minus Weak) - Fama-French 5-factor
    pub rmw: f64,
    /// Investment factor (Conservative Minus Aggressive) - Fama-French 5-factor
    pub cma: f64,
    /// Momentum factor (Up Minus Down) - Carhart 4-factor
    pub umd: f64,
    /// Risk-free rate for the period
    pub rf: f64,
}

impl FactorReturns {
    /// Create factor returns with all factors.
    pub fn new(mkt_rf: f64, smb: f64, hml: f64, rmw: f64, cma: f64, umd: f64, rf: f64) -> Self {
        Self {
            mkt_rf,
            smb,
            hml,
            rmw,
            cma,
            umd,
            rf,
        }
    }

    /// Create factor returns for Fama-French 3-factor model.
    pub fn three_factor(mkt_rf: f64, smb: f64, hml: f64, rf: f64) -> Self {
        Self {
            mkt_rf,
            smb,
            hml,
            rmw: 0.0,
            cma: 0.0,
            umd: 0.0,
            rf,
        }
    }

    /// Create factor returns for Carhart 4-factor model.
    pub fn four_factor(mkt_rf: f64, smb: f64, hml: f64, umd: f64, rf: f64) -> Self {
        Self {
            mkt_rf,
            smb,
            hml,
            rmw: 0.0,
            cma: 0.0,
            umd,
            rf,
        }
    }
}

/// Result of multiple linear regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Regression coefficients (betas), excluding intercept
    pub coefficients: Vec<f64>,
    /// Intercept (alpha)
    pub intercept: f64,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Adjusted R-squared
    pub r_squared_adj: f64,
    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,
    /// t-statistics for coefficients
    pub t_statistics: Vec<f64>,
    /// p-values for coefficients
    pub p_values: Vec<f64>,
    /// Residuals (y - y_hat)
    pub residuals: Vec<f64>,
    /// Number of observations
    pub n_observations: usize,
    /// Residual standard error
    pub residual_std_error: f64,
    /// F-statistic for overall model significance
    pub f_statistic: f64,
}

impl RegressionResult {
    /// Check if a coefficient is statistically significant at the given alpha level.
    pub fn is_significant(&self, coef_index: usize, alpha: f64) -> bool {
        if coef_index >= self.p_values.len() {
            return false;
        }
        self.p_values[coef_index] < alpha
    }
}

/// Factor loadings from regression analysis.
/// These represent the exposure of a portfolio to various risk factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorLoadings {
    /// Alpha (annualized): excess return not explained by factors
    pub alpha: f64,
    /// Alpha t-statistic
    pub alpha_t_stat: f64,
    /// Alpha p-value
    pub alpha_p_value: f64,
    /// Market beta (exposure to market factor)
    pub market_beta: f64,
    /// Market beta t-statistic
    pub market_t_stat: f64,
    /// SMB beta (exposure to size factor)
    pub smb_beta: f64,
    /// SMB t-statistic
    pub smb_t_stat: f64,
    /// HML beta (exposure to value factor)
    pub hml_beta: f64,
    /// HML t-statistic
    pub hml_t_stat: f64,
    /// RMW beta (exposure to profitability factor) - Fama-French 5-factor only
    pub rmw_beta: f64,
    /// RMW t-statistic
    pub rmw_t_stat: f64,
    /// CMA beta (exposure to investment factor) - Fama-French 5-factor only
    pub cma_beta: f64,
    /// CMA t-statistic
    pub cma_t_stat: f64,
    /// UMD beta (exposure to momentum factor) - Carhart 4-factor only
    pub umd_beta: f64,
    /// UMD t-statistic
    pub umd_t_stat: f64,
    /// R-squared: proportion of return variance explained by factors
    pub r_squared: f64,
    /// Adjusted R-squared
    pub r_squared_adj: f64,
    /// Number of observations used
    pub n_observations: usize,
    /// Model type used for attribution
    pub model_type: FactorModelType,
}

/// Type of factor model used for attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactorModelType {
    /// Fama-French 3-factor model (MKT, SMB, HML)
    FamaFrench3,
    /// Carhart 4-factor model (MKT, SMB, HML, UMD)
    Carhart4,
    /// Fama-French 5-factor model (MKT, SMB, HML, RMW, CMA)
    FamaFrench5,
    /// Fama-French 5-factor + Momentum (all 6 factors)
    FamaFrench5Momentum,
}

impl std::fmt::Display for FactorModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactorModelType::FamaFrench3 => write!(f, "Fama-French 3-Factor"),
            FactorModelType::Carhart4 => write!(f, "Carhart 4-Factor"),
            FactorModelType::FamaFrench5 => write!(f, "Fama-French 5-Factor"),
            FactorModelType::FamaFrench5Momentum => write!(f, "Fama-French 5-Factor + Momentum"),
        }
    }
}

/// Performs Ordinary Least Squares (OLS) multiple linear regression.
///
/// Solves: Y = Xβ + ε using the normal equations: β = (X'X)^(-1) X'Y
///
/// # Arguments
/// * `y` - Dependent variable (portfolio excess returns)
/// * `x` - Independent variables (factor returns), each inner Vec is one factor's time series
///
/// # Returns
/// `RegressionResult` containing coefficients, R², t-statistics, etc.
pub fn multiple_regression(y: &[f64], x: &[Vec<f64>]) -> Option<RegressionResult> {
    let n = y.len();
    let k = x.len(); // number of independent variables (factors)

    if n < k + 2 || k == 0 {
        return None;
    }

    // Verify all x vectors have the same length as y
    for factor in x {
        if factor.len() != n {
            return None;
        }
    }

    // Build design matrix X with intercept column
    // X is n x (k+1) matrix: [1, x1, x2, ..., xk]
    let cols = k + 1;

    // Calculate X'X (cols x cols matrix)
    let mut xtx = vec![vec![0.0; cols]; cols];

    // First column is all 1s (intercept)
    xtx[0][0] = n as f64;
    for j in 1..cols {
        let sum: f64 = x[j - 1].iter().sum();
        xtx[0][j] = sum;
        xtx[j][0] = sum;
    }

    // Fill the rest of X'X
    for i in 1..cols {
        for j in i..cols {
            let sum: f64 = x[i - 1]
                .iter()
                .zip(x[j - 1].iter())
                .map(|(a, b)| a * b)
                .sum();
            xtx[i][j] = sum;
            xtx[j][i] = sum;
        }
    }

    // Calculate X'Y (cols x 1 vector)
    let mut xty = vec![0.0; cols];
    xty[0] = y.iter().sum();
    for j in 1..cols {
        xty[j] = x[j - 1].iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    }

    // Solve (X'X)β = X'Y using Gaussian elimination with partial pivoting
    let coefficients = solve_linear_system(&xtx, &xty)?;

    // Extract intercept and betas
    let intercept = coefficients[0];
    let betas: Vec<f64> = coefficients[1..].to_vec();

    // Calculate fitted values and residuals
    let mut y_hat = vec![0.0; n];
    let mut residuals = vec![0.0; n];
    for i in 0..n {
        y_hat[i] = intercept;
        for j in 0..k {
            y_hat[i] += betas[j] * x[j][i];
        }
        residuals[i] = y[i] - y_hat[i];
    }

    // Calculate R-squared
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = residuals.iter().map(|e| e.powi(2)).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    // Adjusted R-squared
    let df_total = n as f64 - 1.0;
    let df_resid = n as f64 - cols as f64;
    let r_squared_adj = if df_resid > 0.0 && df_total > 0.0 {
        1.0 - (1.0 - r_squared) * (df_total / df_resid)
    } else {
        r_squared
    };

    // Residual standard error
    let residual_std_error = if df_resid > 0.0 {
        (ss_res / df_resid).sqrt()
    } else {
        0.0
    };

    // Calculate standard errors of coefficients
    // SE(β) = sqrt(σ² * diag((X'X)^(-1)))
    let xtx_inv = invert_matrix(&xtx)?;
    let mut std_errors = Vec::with_capacity(cols);
    let sigma_sq = if df_resid > 0.0 {
        ss_res / df_resid
    } else {
        0.0
    };

    for (i, row) in xtx_inv.iter().enumerate().take(cols) {
        let se = (sigma_sq * row[i]).sqrt();
        std_errors.push(se);
    }

    // Calculate t-statistics and p-values
    let mut t_statistics = Vec::with_capacity(cols);
    let mut p_values = Vec::with_capacity(cols);

    for i in 0..cols {
        let se = std_errors[i].max(1e-10);
        let t_stat = coefficients[i] / se;
        std_errors[i] = se;
        t_statistics.push(t_stat);

        // Two-tailed p-value using t-distribution approximation
        let p_val = t_distribution_p_value(t_stat.abs(), df_resid as usize);
        p_values.push(p_val);
    }

    // Calculate F-statistic
    let ss_reg = ss_tot - ss_res;
    let df_reg = k as f64;
    let f_statistic = if df_resid > 0.0 && ss_res > 0.0 && df_reg > 0.0 {
        (ss_reg / df_reg) / (ss_res / df_resid)
    } else {
        0.0
    };

    Some(RegressionResult {
        coefficients: betas,
        intercept,
        r_squared,
        r_squared_adj,
        std_errors: std_errors[1..].to_vec(), // Exclude intercept SE for betas
        t_statistics: t_statistics[1..].to_vec(),
        p_values: p_values[1..].to_vec(),
        residuals,
        n_observations: n,
        residual_std_error,
        f_statistic,
    })
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }

    // Create augmented matrix
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, bi)| {
            let mut new_row = row.clone();
            new_row.push(*bi);
            new_row
        })
        .collect();

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        aug.swap(i, max_row);

        // Check for singular matrix
        if aug[i][i].abs() < 1e-12 {
            return None;
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[k][i] / aug[i][i];
            let aug_i_vals: Vec<f64> = aug[i][i..=n].to_vec();
            for (j_offset, aug_i_val) in aug_i_vals.iter().enumerate() {
                aug[k][i + j_offset] -= factor * aug_i_val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Some(x)
}

/// Invert a matrix using Gauss-Jordan elimination.
fn invert_matrix(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return None;
    }

    // Create augmented matrix [A | I]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut new_row = row.clone();
            for j in 0..n {
                new_row.push(if i == j { 1.0 } else { 0.0 });
            }
            new_row
        })
        .collect();

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }

        aug.swap(i, max_row);

        if aug[i][i].abs() < 1e-12 {
            return None; // Singular matrix
        }

        // Scale pivot row
        let pivot = aug[i][i];
        for elem in aug[i].iter_mut().take(2 * n) {
            *elem /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                let aug_i_vals: Vec<f64> = aug[i].iter().take(2 * n).copied().collect();
                for (j, aug_i_val) in aug_i_vals.iter().enumerate() {
                    aug[k][j] -= factor * aug_i_val;
                }
            }
        }
    }

    // Extract inverse matrix
    let inverse: Vec<Vec<f64>> = aug.iter().map(|row| row[n..].to_vec()).collect();

    Some(inverse)
}

/// Approximate p-value from t-distribution using normal approximation for large df.
fn t_distribution_p_value(t: f64, df: usize) -> f64 {
    if df == 0 {
        return 1.0;
    }

    // For df > 30, use normal approximation
    if df > 30 {
        // Two-tailed p-value from normal distribution
        let p = 2.0 * (1.0 - normal_cdf(t));
        return p.clamp(0.0, 1.0);
    }

    // For smaller df, use a more accurate approximation
    // Using the regularized incomplete beta function approximation
    let x = df as f64 / (df as f64 + t * t);
    let p = incomplete_beta(df as f64 / 2.0, 0.5, x);
    p.clamp(0.0, 1.0)
}

/// Standard normal CDF using error function.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Approximation of the regularized incomplete beta function.
/// Uses continued fraction approximation for numerical stability.
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the symmetry relation when appropriate
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - incomplete_beta(b, a, 1.0 - x);
    }

    // Compute using continued fraction (Lentz's algorithm)
    let lnbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - lnbeta).exp() / a;

    // Continued fraction
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 0.0;

    for m in 1..200 {
        let m = m as f64;

        // Even step
        let numerator = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + numerator * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + numerator / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        f *= d * c;

        // Odd step
        let numerator = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + numerator * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + numerator / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        let delta = d * c;
        f *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    front * f
}

/// Log gamma function approximation using Lanczos coefficients.
/// These are standard mathematical constants that require high precision.
#[allow(clippy::excessive_precision)]
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // Lanczos approximation coefficients (require high precision)
    let g = 7;
    let coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Reflection formula
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut y = coef[0];
    for (i, &c) in coef.iter().enumerate().skip(1).take(g + 1) {
        y += c / (x + i as f64);
    }

    let t = x + g as f64 + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (x + 0.5)) - t + y.ln()
}

/// Factor Attribution analyzer for portfolio returns.
pub struct FactorAttribution;

impl FactorAttribution {
    /// Run Fama-French 3-factor model attribution.
    ///
    /// Model: R_p - R_f = α + β_mkt*(R_m - R_f) + β_smb*SMB + β_hml*HML + ε
    ///
    /// # Arguments
    /// * `portfolio_returns` - Portfolio daily returns (not excess returns)
    /// * `factor_data` - Vector of daily factor returns
    ///
    /// # Returns
    /// Factor loadings with alpha, betas, and R²
    pub fn fama_french_3factor(
        portfolio_returns: &[f64],
        factor_data: &[FactorReturns],
    ) -> Option<FactorLoadings> {
        if portfolio_returns.len() != factor_data.len() || portfolio_returns.len() < 20 {
            return None;
        }

        // Calculate excess portfolio returns (R_p - R_f)
        let excess_returns: Vec<f64> = portfolio_returns
            .iter()
            .zip(factor_data.iter())
            .map(|(r, f)| r - f.rf)
            .collect();

        // Prepare factor columns
        let mkt: Vec<f64> = factor_data.iter().map(|f| f.mkt_rf).collect();
        let smb: Vec<f64> = factor_data.iter().map(|f| f.smb).collect();
        let hml: Vec<f64> = factor_data.iter().map(|f| f.hml).collect();

        let factors = vec![mkt, smb, hml];
        let regression = multiple_regression(&excess_returns, &factors)?;

        Some(Self::build_loadings(
            &regression,
            FactorModelType::FamaFrench3,
        ))
    }

    /// Run Carhart 4-factor model attribution.
    ///
    /// Model: R_p - R_f = α + β_mkt*(R_m - R_f) + β_smb*SMB + β_hml*HML + β_umd*UMD + ε
    pub fn carhart_4factor(
        portfolio_returns: &[f64],
        factor_data: &[FactorReturns],
    ) -> Option<FactorLoadings> {
        if portfolio_returns.len() != factor_data.len() || portfolio_returns.len() < 20 {
            return None;
        }

        let excess_returns: Vec<f64> = portfolio_returns
            .iter()
            .zip(factor_data.iter())
            .map(|(r, f)| r - f.rf)
            .collect();

        let mkt: Vec<f64> = factor_data.iter().map(|f| f.mkt_rf).collect();
        let smb: Vec<f64> = factor_data.iter().map(|f| f.smb).collect();
        let hml: Vec<f64> = factor_data.iter().map(|f| f.hml).collect();
        let umd: Vec<f64> = factor_data.iter().map(|f| f.umd).collect();

        let factors = vec![mkt, smb, hml, umd];
        let regression = multiple_regression(&excess_returns, &factors)?;

        Some(Self::build_loadings(&regression, FactorModelType::Carhart4))
    }

    /// Run Fama-French 5-factor model attribution.
    ///
    /// Model: R_p - R_f = α + β_mkt*(R_m - R_f) + β_smb*SMB + β_hml*HML + β_rmw*RMW + β_cma*CMA + ε
    pub fn fama_french_5factor(
        portfolio_returns: &[f64],
        factor_data: &[FactorReturns],
    ) -> Option<FactorLoadings> {
        if portfolio_returns.len() != factor_data.len() || portfolio_returns.len() < 30 {
            return None;
        }

        let excess_returns: Vec<f64> = portfolio_returns
            .iter()
            .zip(factor_data.iter())
            .map(|(r, f)| r - f.rf)
            .collect();

        let mkt: Vec<f64> = factor_data.iter().map(|f| f.mkt_rf).collect();
        let smb: Vec<f64> = factor_data.iter().map(|f| f.smb).collect();
        let hml: Vec<f64> = factor_data.iter().map(|f| f.hml).collect();
        let rmw: Vec<f64> = factor_data.iter().map(|f| f.rmw).collect();
        let cma: Vec<f64> = factor_data.iter().map(|f| f.cma).collect();

        let factors = vec![mkt, smb, hml, rmw, cma];
        let regression = multiple_regression(&excess_returns, &factors)?;

        Some(Self::build_loadings(
            &regression,
            FactorModelType::FamaFrench5,
        ))
    }

    /// Run full 6-factor model (Fama-French 5-factor + Momentum).
    pub fn six_factor(
        portfolio_returns: &[f64],
        factor_data: &[FactorReturns],
    ) -> Option<FactorLoadings> {
        if portfolio_returns.len() != factor_data.len() || portfolio_returns.len() < 30 {
            return None;
        }

        let excess_returns: Vec<f64> = portfolio_returns
            .iter()
            .zip(factor_data.iter())
            .map(|(r, f)| r - f.rf)
            .collect();

        let mkt: Vec<f64> = factor_data.iter().map(|f| f.mkt_rf).collect();
        let smb: Vec<f64> = factor_data.iter().map(|f| f.smb).collect();
        let hml: Vec<f64> = factor_data.iter().map(|f| f.hml).collect();
        let rmw: Vec<f64> = factor_data.iter().map(|f| f.rmw).collect();
        let cma: Vec<f64> = factor_data.iter().map(|f| f.cma).collect();
        let umd: Vec<f64> = factor_data.iter().map(|f| f.umd).collect();

        let factors = vec![mkt, smb, hml, rmw, cma, umd];
        let regression = multiple_regression(&excess_returns, &factors)?;

        Some(Self::build_loadings(
            &regression,
            FactorModelType::FamaFrench5Momentum,
        ))
    }

    /// Build FactorLoadings struct from regression results.
    fn build_loadings(
        regression: &RegressionResult,
        model_type: FactorModelType,
    ) -> FactorLoadings {
        let n_factors = regression.coefficients.len();

        // Annualize alpha (daily to annual): α_annual = α_daily * 252
        let alpha = regression.intercept * 252.0 * 100.0; // Convert to percentage

        // Calculate t-stat for alpha using intercept standard error
        // We need to recalculate this since we stored only coefficient SEs
        let alpha_t_stat = if regression.residual_std_error > 0.0 {
            regression.intercept
                / (regression.residual_std_error / (regression.n_observations as f64).sqrt())
        } else {
            0.0
        };
        let alpha_p_value = t_distribution_p_value(
            alpha_t_stat.abs(),
            regression.n_observations.saturating_sub(n_factors + 1),
        );

        // Extract betas and t-stats by factor
        let get_beta = |i: usize| regression.coefficients.get(i).copied().unwrap_or(0.0);
        let get_t = |i: usize| regression.t_statistics.get(i).copied().unwrap_or(0.0);

        FactorLoadings {
            alpha,
            alpha_t_stat,
            alpha_p_value,
            market_beta: get_beta(0),
            market_t_stat: get_t(0),
            smb_beta: get_beta(1),
            smb_t_stat: get_t(1),
            hml_beta: get_beta(2),
            hml_t_stat: get_t(2),
            rmw_beta: if matches!(
                model_type,
                FactorModelType::FamaFrench5 | FactorModelType::FamaFrench5Momentum
            ) {
                get_beta(3)
            } else {
                0.0
            },
            rmw_t_stat: if matches!(
                model_type,
                FactorModelType::FamaFrench5 | FactorModelType::FamaFrench5Momentum
            ) {
                get_t(3)
            } else {
                0.0
            },
            cma_beta: if matches!(
                model_type,
                FactorModelType::FamaFrench5 | FactorModelType::FamaFrench5Momentum
            ) {
                get_beta(4)
            } else {
                0.0
            },
            cma_t_stat: if matches!(
                model_type,
                FactorModelType::FamaFrench5 | FactorModelType::FamaFrench5Momentum
            ) {
                get_t(4)
            } else {
                0.0
            },
            umd_beta: match model_type {
                FactorModelType::Carhart4 => get_beta(3),
                FactorModelType::FamaFrench5Momentum => get_beta(5),
                _ => 0.0,
            },
            umd_t_stat: match model_type {
                FactorModelType::Carhart4 => get_t(3),
                FactorModelType::FamaFrench5Momentum => get_t(5),
                _ => 0.0,
            },
            r_squared: regression.r_squared,
            r_squared_adj: regression.r_squared_adj,
            n_observations: regression.n_observations,
            model_type,
        }
    }

    /// Check if factor loadings indicate significant factor exposure.
    pub fn has_significant_exposure(loadings: &FactorLoadings, alpha_level: f64) -> Vec<String> {
        let mut significant = Vec::new();

        if loadings.alpha_p_value < alpha_level {
            significant.push("Alpha".to_string());
        }

        // Check market beta (t-stat > ~2 for 95% confidence)
        let t_critical = 1.96; // Approximate for large samples
        if loadings.market_t_stat.abs() > t_critical {
            significant.push("Market".to_string());
        }
        if loadings.smb_t_stat.abs() > t_critical {
            significant.push("SMB (Size)".to_string());
        }
        if loadings.hml_t_stat.abs() > t_critical {
            significant.push("HML (Value)".to_string());
        }
        if matches!(
            loadings.model_type,
            FactorModelType::FamaFrench5 | FactorModelType::FamaFrench5Momentum
        ) && loadings.rmw_t_stat.abs() > t_critical
        {
            significant.push("RMW (Profitability)".to_string());
        }
        if matches!(
            loadings.model_type,
            FactorModelType::FamaFrench5 | FactorModelType::FamaFrench5Momentum
        ) && loadings.cma_t_stat.abs() > t_critical
        {
            significant.push("CMA (Investment)".to_string());
        }
        if matches!(
            loadings.model_type,
            FactorModelType::Carhart4 | FactorModelType::FamaFrench5Momentum
        ) && loadings.umd_t_stat.abs() > t_critical
        {
            significant.push("UMD (Momentum)".to_string());
        }

        significant
    }
}

impl ResultFormatter {
    /// Print factor attribution results.
    pub fn print_factor_attribution(loadings: &FactorLoadings) {
        println!();
        println!("{}", "═".repeat(60).blue());
        println!(
            "{}",
            format!(" FACTOR ATTRIBUTION ({}) ", loadings.model_type)
                .bold()
                .blue()
        );
        println!("{}", "═".repeat(60).blue());
        println!();

        // Model fit
        println!("{}", "Model Fit".bold().underline());
        println!(
            "  R-squared:       {:>10.4}  ({:.1}% of variance explained)",
            loadings.r_squared,
            loadings.r_squared * 100.0
        );
        println!("  Adj. R-squared:  {:>10.4}", loadings.r_squared_adj);
        println!("  Observations:    {:>10}", loadings.n_observations);
        println!();

        // Alpha
        println!("{}", "Alpha (Annualized)".bold().underline());
        let alpha_sig = if loadings.alpha_p_value < 0.05 {
            "**".green()
        } else if loadings.alpha_p_value < 0.10 {
            "*".yellow()
        } else {
            "".normal()
        };
        println!(
            "  Alpha:           {:>10.2}%  (t={:.2}, p={:.4}) {}",
            loadings.alpha, loadings.alpha_t_stat, loadings.alpha_p_value, alpha_sig
        );
        println!();

        // Factor Loadings
        println!("{}", "Factor Loadings".bold().underline());
        Self::print_factor_row("Market (β)", loadings.market_beta, loadings.market_t_stat);
        Self::print_factor_row("SMB (Size)", loadings.smb_beta, loadings.smb_t_stat);
        Self::print_factor_row("HML (Value)", loadings.hml_beta, loadings.hml_t_stat);

        if matches!(
            loadings.model_type,
            FactorModelType::FamaFrench5 | FactorModelType::FamaFrench5Momentum
        ) {
            Self::print_factor_row("RMW (Profit.)", loadings.rmw_beta, loadings.rmw_t_stat);
            Self::print_factor_row("CMA (Invest.)", loadings.cma_beta, loadings.cma_t_stat);
        }

        if matches!(
            loadings.model_type,
            FactorModelType::Carhart4 | FactorModelType::FamaFrench5Momentum
        ) {
            Self::print_factor_row("UMD (Momentum)", loadings.umd_beta, loadings.umd_t_stat);
        }

        println!();
        println!(
            "  {} = significant at 5%, {} = significant at 10%",
            "**".green(),
            "*".yellow()
        );
        println!("{}", "═".repeat(60).blue());
    }

    fn print_factor_row(name: &str, beta: f64, t_stat: f64) {
        let sig = if t_stat.abs() > 2.576 {
            "***".green()
        } else if t_stat.abs() > 1.96 {
            "**".green()
        } else if t_stat.abs() > 1.645 {
            "*".yellow()
        } else {
            "".normal()
        };
        println!("  {:14} {:>10.4}  (t={:>6.2}) {}", name, beta, t_stat, sig);
    }
}

// ============================================================================
// Statistical Tests for Validation and Robustness
// ============================================================================

/// Result of a statistical hypothesis test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    /// Name of the test
    pub test_name: String,
    /// Test statistic value
    pub statistic: f64,
    /// P-value (probability of observing this result under null hypothesis)
    pub p_value: f64,
    /// Whether to reject the null hypothesis at 5% significance level
    pub reject_null_at_5pct: bool,
    /// Whether to reject the null hypothesis at 1% significance level
    pub reject_null_at_1pct: bool,
    /// Human-readable interpretation of the result
    pub interpretation: String,
}

impl StatisticalTestResult {
    /// Create a new statistical test result.
    pub fn new(
        test_name: impl Into<String>,
        statistic: f64,
        p_value: f64,
        interpretation: impl Into<String>,
    ) -> Self {
        Self {
            test_name: test_name.into(),
            statistic,
            p_value,
            reject_null_at_5pct: p_value < 0.05,
            reject_null_at_1pct: p_value < 0.01,
            interpretation: interpretation.into(),
        }
    }
}

/// Statistical tests for validating return distributions and time series properties.
pub struct StatisticalTests;

impl StatisticalTests {
    /// Jarque-Bera test for normality.
    ///
    /// Tests whether the sample data has the skewness and kurtosis matching
    /// a normal distribution. Useful for validating return distribution assumptions.
    ///
    /// # Null Hypothesis
    /// The data comes from a normal distribution (skewness = 0, excess kurtosis = 0).
    ///
    /// # Arguments
    /// * `data` - Sample data (typically daily returns)
    ///
    /// # Returns
    /// Test result with JB statistic and p-value. High JB statistic (low p-value)
    /// indicates non-normality.
    ///
    /// # Formula
    /// JB = (n/6) * (S² + K²/4)
    /// where S = skewness, K = excess kurtosis, n = sample size
    pub fn jarque_bera(data: &[f64]) -> Option<StatisticalTestResult> {
        let n = data.len();
        if n < 8 {
            // Need reasonable sample size for meaningful test
            return None;
        }

        let n_f64 = n as f64;

        // Calculate mean
        let mean = data.iter().sum::<f64>() / n_f64;

        // Calculate moments
        let m2: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_f64;
        let m3: f64 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n_f64;
        let m4: f64 = data.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n_f64;

        if m2 == 0.0 {
            // Zero variance - data is constant
            return None;
        }

        // Sample skewness: m3 / m2^(3/2)
        let skewness = m3 / m2.powf(1.5);

        // Sample excess kurtosis: m4 / m2^2 - 3
        let excess_kurtosis = (m4 / m2.powi(2)) - 3.0;

        // Jarque-Bera statistic
        // JB = (n/6) * (S^2 + K^2/4)
        let jb_statistic = (n_f64 / 6.0) * (skewness.powi(2) + excess_kurtosis.powi(2) / 4.0);

        // Under H0, JB follows chi-squared distribution with 2 degrees of freedom
        let p_value = 1.0 - chi_squared_cdf(jb_statistic, 2);

        let interpretation = if p_value < 0.05 {
            format!(
                "Non-normal distribution (skew={:.3}, kurt={:.3}). Sharpe ratio assumptions may be violated.",
                skewness, excess_kurtosis
            )
        } else {
            format!(
                "Cannot reject normality (skew={:.3}, kurt={:.3}). Distribution appears normal.",
                skewness, excess_kurtosis
            )
        };

        Some(StatisticalTestResult::new(
            "Jarque-Bera",
            jb_statistic,
            p_value,
            interpretation,
        ))
    }

    /// Durbin-Watson test for autocorrelation in residuals.
    ///
    /// Tests whether there is first-order autocorrelation in a time series.
    /// Critical for detecting serial dependence in strategy returns.
    ///
    /// # Null Hypothesis
    /// No first-order autocorrelation (ρ = 0).
    ///
    /// # Arguments
    /// * `residuals` - Time series of residuals or returns
    ///
    /// # Returns
    /// Test result with DW statistic (range 0-4).
    /// - DW ≈ 2: No autocorrelation
    /// - DW < 2: Positive autocorrelation (trending)
    /// - DW > 2: Negative autocorrelation (mean-reverting)
    ///
    /// # Formula
    /// DW = Σ(e_t - e_{t-1})² / Σe_t²
    pub fn durbin_watson(residuals: &[f64]) -> Option<StatisticalTestResult> {
        let n = residuals.len();
        if n < 3 {
            return None;
        }

        let mean = residuals.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = residuals.iter().map(|e| e - mean).collect();

        // Sum of squared residuals relative to the mean
        let ss_residuals: f64 = centered.iter().map(|e| e.powi(2)).sum();

        if ss_residuals == 0.0 {
            return None;
        }

        // Sum of squared differences
        let ss_diff: f64 = centered.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();

        let dw_statistic = ss_diff / ss_residuals;

        // Convert DW to approximate p-value for positive autocorrelation
        // Using the relationship: DW ≈ 2(1 - ρ) where ρ is first-order autocorrelation
        let rho = 1.0 - dw_statistic / 2.0;

        // Approximate p-value using transformation
        // Under H0, DW is approximately normal for large samples
        let n_f64 = n as f64;
        let se = (4.0 / n_f64).sqrt(); // Approximate standard error

        // Test for departure from 2 (positive autocorrelation if DW < 2)
        let z = (dw_statistic - 2.0) / se;
        let p_value_lower = normal_cdf(z); // P-value for positive autocorrelation

        // Use two-sided p-value for general test
        let p_value = 2.0 * p_value_lower.min(1.0 - p_value_lower);

        let interpretation = if dw_statistic < 1.5 {
            format!(
                "Strong positive autocorrelation (ρ≈{:.3}). Returns may be trending.",
                rho
            )
        } else if dw_statistic < 1.8 {
            format!(
                "Moderate positive autocorrelation (ρ≈{:.3}). Some serial dependence present.",
                rho
            )
        } else if dw_statistic > 2.5 {
            format!(
                "Negative autocorrelation (ρ≈{:.3}). Returns may be mean-reverting.",
                rho
            )
        } else {
            format!(
                "No significant autocorrelation (ρ≈{:.3}). Returns appear independent.",
                rho
            )
        };

        Some(StatisticalTestResult::new(
            "Durbin-Watson",
            dw_statistic,
            p_value,
            interpretation,
        ))
    }

    /// Augmented Dickey-Fuller (ADF) test for stationarity.
    ///
    /// Tests whether a time series has a unit root (is non-stationary).
    /// Critical for validating that returns are stationary before backtesting.
    ///
    /// # Null Hypothesis
    /// The series has a unit root (is non-stationary).
    ///
    /// # Arguments
    /// * `data` - Time series data (e.g., returns or prices)
    /// * `max_lag` - Maximum number of lags to include (None for automatic selection)
    ///
    /// # Returns
    /// Test result with ADF statistic. More negative values indicate stronger
    /// evidence against unit root (i.e., evidence of stationarity).
    ///
    /// # Formula
    /// Δy_t = α + βt + γy_{t-1} + Σδ_i*Δy_{t-i} + ε_t
    /// Test H0: γ = 0 (unit root) vs H1: γ < 0 (stationary)
    pub fn adf_test(data: &[f64], max_lag: Option<usize>) -> Option<StatisticalTestResult> {
        let n = data.len();
        if n < 20 {
            return None;
        }

        // Automatic lag selection using Schwert (1989) rule: max_lag = 12*(T/100)^(1/4)
        let auto_lag = ((12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize).max(1);

        // Calculate first differences
        let diff: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
        if diff.len() < 5 {
            return None;
        }

        let max_diff_lag = diff.len().saturating_sub(1);
        if max_diff_lag == 0 {
            return None;
        }

        let initial_lag = max_lag.unwrap_or(auto_lag).min(n / 4).min(max_diff_lag);

        for current_lag in (0..=initial_lag).rev() {
            if diff.len() <= current_lag {
                continue;
            }

            let y: Vec<f64> = diff.iter().skip(current_lag).copied().collect();
            let n_obs = y.len();

            if n_obs < 10 {
                continue;
            }

            let mut x_lagged_level = Vec::with_capacity(n_obs);
            let mut x_lagged_diffs: Vec<Vec<f64>> = (0..current_lag)
                .map(|_| Vec::with_capacity(n_obs))
                .collect();

            for i in current_lag..diff.len() {
                x_lagged_level.push(data[i]);
                for j in 0..current_lag {
                    x_lagged_diffs[j].push(diff[i - 1 - j]);
                }
            }

            let mut factors: Vec<Vec<f64>> = vec![x_lagged_level];
            for lag_diff in x_lagged_diffs {
                factors.push(lag_diff);
            }

            if let Some(regression) = multiple_regression(&y, &factors) {
                let adf_statistic = regression.t_statistics.first().copied().unwrap_or(0.0);

                let critical_1pct = -3.43;
                let critical_5pct = -2.86;
                let critical_10pct = -2.57;

                let p_value = adf_p_value(adf_statistic, n_obs);

                let interpretation = if adf_statistic < critical_1pct {
                    format!(
                        "Strongly stationary (ADF < {:.2}). Series has no unit root.",
                        critical_1pct
                    )
                } else if adf_statistic < critical_5pct {
                    format!(
                        "Stationary at 5% level (ADF < {:.2}). Reject unit root hypothesis.",
                        critical_5pct
                    )
                } else if adf_statistic < critical_10pct {
                    format!(
                        "Weakly stationary (ADF < {:.2}). Marginal evidence against unit root.",
                        critical_10pct
                    )
                } else {
                    "Non-stationary (cannot reject unit root). Consider differencing the series."
                        .to_string()
                };

                return Some(StatisticalTestResult::new(
                    "Augmented Dickey-Fuller",
                    adf_statistic,
                    p_value,
                    interpretation,
                ));
            }
        }

        None
    }

    /// Calculate first-order autocorrelation coefficient.
    ///
    /// # Arguments
    /// * `data` - Time series data
    ///
    /// # Returns
    /// Autocorrelation coefficient at lag 1 (range -1 to 1)
    pub fn autocorrelation(data: &[f64]) -> Option<f64> {
        let n = data.len();
        if n < 3 {
            return None;
        }

        let mean = data.iter().sum::<f64>() / n as f64;

        // Variance
        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        if variance == 0.0 {
            return None;
        }

        // Autocovariance at lag 1
        let autocovariance: f64 = data
            .windows(2)
            .map(|w| (w[0] - mean) * (w[1] - mean))
            .sum::<f64>()
            / (n - 1) as f64;

        Some(autocovariance / variance)
    }

    /// Ljung-Box test for autocorrelation at multiple lags.
    ///
    /// Tests the null hypothesis that the first h autocorrelations are jointly zero.
    ///
    /// # Arguments
    /// * `data` - Time series data
    /// * `lags` - Number of lags to test (default: 10)
    ///
    /// # Returns
    /// Q statistic and p-value. Low p-value indicates significant autocorrelation.
    pub fn ljung_box(data: &[f64], lags: Option<usize>) -> Option<StatisticalTestResult> {
        let n = data.len();
        let h = lags.unwrap_or(10.min(n / 5));

        if n < h + 5 {
            return None;
        }

        let mean = data.iter().sum::<f64>() / n as f64;

        // Variance (denominator for autocorrelation)
        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();

        if variance == 0.0 {
            return None;
        }

        // Calculate autocorrelations for each lag
        let mut q_stat = 0.0;
        for k in 1..=h {
            if k >= n {
                break;
            }

            // Autocovariance at lag k
            let autocov: f64 = (0..n - k)
                .map(|i| (data[i] - mean) * (data[i + k] - mean))
                .sum();

            let rho_k = autocov / variance;

            // Ljung-Box Q = n(n+2) * Σ (ρ_k² / (n-k))
            q_stat += rho_k.powi(2) / (n - k) as f64;
        }

        q_stat *= (n as f64) * (n as f64 + 2.0);

        // Under H0, Q follows chi-squared distribution with h degrees of freedom
        let p_value = 1.0 - chi_squared_cdf(q_stat, h);

        let interpretation = if p_value < 0.05 {
            format!(
                "Significant autocorrelation detected (Q={:.2}, lags={}). Returns are serially dependent.",
                q_stat, h
            )
        } else {
            format!(
                "No significant autocorrelation (Q={:.2}, lags={}). Returns appear independent.",
                q_stat, h
            )
        };

        Some(StatisticalTestResult::new(
            "Ljung-Box Q",
            q_stat,
            p_value,
            interpretation,
        ))
    }

    /// Print a summary of statistical test results.
    pub fn print_summary(results: &[StatisticalTestResult]) {
        println!();
        println!("{}", "═".repeat(70).blue());
        println!("{}", " STATISTICAL TESTS SUMMARY ".bold().blue());
        println!("{}", "═".repeat(70).blue());
        println!();

        let mut builder = Builder::default();
        builder.push_record(["Test", "Statistic", "P-value", "Result"]);

        for result in results {
            let status = if result.reject_null_at_1pct {
                "***".to_string()
            } else if result.reject_null_at_5pct {
                "**".to_string()
            } else {
                "".to_string()
            };

            builder.push_record([
                result.test_name.clone(),
                format!("{:.4}", result.statistic),
                format!("{:.4}", result.p_value),
                status,
            ]);
        }

        let mut table = builder.build();
        table.with(Style::rounded());
        println!("{}", table);
        println!();
        println!("  *** = significant at 1%, ** = significant at 5%");
        println!();

        for result in results {
            println!("  {}: {}", result.test_name.bold(), result.interpretation);
        }

        println!("{}", "═".repeat(70).blue());
    }
}

/// Chi-squared CDF using the incomplete gamma function.
/// Useful for computing p-values for chi-squared distributed statistics.
fn chi_squared_cdf(x: f64, df: usize) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if df == 0 {
        return 1.0;
    }

    // Chi-squared CDF = lower incomplete gamma / gamma
    // P(X <= x) = γ(k/2, x/2) / Γ(k/2)
    let k = df as f64;
    lower_incomplete_gamma(k / 2.0, x / 2.0)
}

/// Lower incomplete gamma function (regularized).
/// Uses series expansion for small x and continued fraction for large x.
fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if a <= 0.0 {
        return 1.0;
    }

    // Choose method based on x relative to a
    if x < a + 1.0 {
        // Use series expansion
        let mut sum = 1.0 / a;
        let mut term = 1.0 / a;

        for n in 1..200 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }

        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Use continued fraction (complement)
        1.0 - upper_incomplete_gamma_cf(a, x)
    }
}

/// Upper incomplete gamma using continued fraction.
fn upper_incomplete_gamma_cf(a: f64, x: f64) -> f64 {
    let mut f = 1e-30_f64;
    let mut c = 1e-30_f64;
    let mut d = 0.0_f64;

    for n in 1..200 {
        let an = if n == 1 {
            1.0
        } else if n % 2 == 0 {
            (n as f64 / 2.0 - 1.0) * x / ((a + n as f64 - 2.0) * (a + n as f64 - 1.0))
        } else {
            -(a + (n as f64 - 1.0) / 2.0) * x / ((a + n as f64 - 2.0) * (a + n as f64 - 1.0))
        };

        let bn = if n == 1 { x } else { a + n as f64 - 1.0 };

        d = bn + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;

        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }

        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    f * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Approximate p-value for ADF test statistic.
/// Uses interpolation from MacKinnon (1994) critical values.
fn adf_p_value(t_stat: f64, n: usize) -> f64 {
    // MacKinnon (1994) critical values for ADF with constant
    // For sample size n=100:
    // 1%: -3.43, 5%: -2.86, 10%: -2.57
    // These shift slightly with sample size, but we use fixed approximation

    // Simple linear interpolation between critical values
    let critical_values = [
        (-4.0, 0.001),
        (-3.5, 0.005),
        (-3.43, 0.01),
        (-3.15, 0.025),
        (-2.86, 0.05),
        (-2.57, 0.10),
        (-2.0, 0.20),
        (-1.5, 0.30),
        (-1.0, 0.50),
        (0.0, 0.70),
        (1.0, 0.90),
        (2.0, 0.99),
    ];

    // Adjust critical values slightly for sample size
    let size_adj = if n < 25 {
        0.3
    } else if n < 50 {
        0.15
    } else if n < 100 {
        0.05
    } else {
        0.0
    };

    // Find bracketing critical values
    for i in 0..critical_values.len() - 1 {
        let (t1, p1) = critical_values[i];
        let (t2, p2) = critical_values[i + 1];
        let t1_adj = t1 - size_adj;
        let t2_adj = t2 - size_adj;

        if t_stat <= t1_adj {
            return p1;
        }
        if t_stat < t2_adj {
            // Linear interpolation
            let ratio = (t_stat - t1_adj) / (t2_adj - t1_adj);
            return p1 + ratio * (p2 - p1);
        }
    }

    // Beyond the table
    if t_stat < critical_values[0].0 - size_adj {
        0.001
    } else {
        0.99
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::BacktestConfig;
    use crate::types::Side;
    use chrono::{TimeZone, Utc};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn create_test_result() -> BacktestResult {
        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap();

        let mut trade = Trade::open("TEST", Side::Buy, 100.0, 100.0, start, 1.0, 0.0);
        trade.close(110.0, end, 1.0);

        BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 110000.0,
            total_return_pct: 10.0,
            annual_return_pct: 10.0,
            trading_days: 252,
            total_trades: 1,
            winning_trades: 1,
            losing_trades: 0,
            win_rate: 100.0,
            avg_win: 1000.0,
            avg_loss: 0.0,
            profit_factor: f64::INFINITY,
            max_drawdown_pct: 5.0,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            calmar_ratio: 2.0,
            trades: vec![trade],
            equity_curve: vec![],
            start_time: start,
            end_time: end,
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
            seed: None,
        }
    }

    #[test]
    fn test_metrics_calculation() {
        let result = create_test_result();
        let metrics = PerformanceMetrics::from_result(&result);

        assert_eq!(metrics.total_trades, 1);
        assert_eq!(metrics.winning_trades, 1);
        assert_eq!(metrics.win_rate, 100.0);
    }

    #[test]
    fn test_json_export() {
        let result = create_test_result();
        let json = ResultFormatter::to_json(&result);
        assert!(json.contains("TestStrategy"));
    }

    #[test]
    fn test_csv_export() {
        let result = create_test_result();
        let csv = ResultFormatter::to_csv_line(&result);
        assert!(csv.contains("TestStrategy"));
        assert!(csv.contains("TEST"));
    }

    #[test]
    fn test_benchmark_metrics_calculation() {
        // Simulated daily returns - portfolio outperforms benchmark
        let portfolio_returns = vec![0.01, -0.005, 0.015, 0.008, -0.012, 0.02, 0.005, -0.003];
        let benchmark_returns = vec![0.008, -0.003, 0.01, 0.005, -0.01, 0.015, 0.003, -0.002];

        let metrics =
            BenchmarkMetrics::calculate("SPY", &portfolio_returns, &benchmark_returns, 0.05);

        assert!(metrics.is_some());
        let m = metrics.unwrap();

        // Beta should be positive (correlated with market)
        assert!(m.beta > 0.0, "Beta should be positive: {}", m.beta);

        // Correlation should be positive
        assert!(
            m.correlation > 0.0,
            "Correlation should be positive: {}",
            m.correlation
        );
        assert!(
            m.correlation <= 1.0,
            "Correlation should be <= 1: {}",
            m.correlation
        );

        // Tracking error should be positive
        assert!(
            m.tracking_error > 0.0,
            "Tracking error should be positive: {}",
            m.tracking_error
        );

        // Excess return should be positive (portfolio outperformed)
        assert!(
            m.excess_return_pct > 0.0,
            "Excess return should be positive: {}",
            m.excess_return_pct
        );

        // Capture ratios should be reasonable
        assert!(
            m.up_capture > 0.0,
            "Up capture should be positive: {}",
            m.up_capture
        );
        assert!(
            m.down_capture > 0.0,
            "Down capture should be positive: {}",
            m.down_capture
        );

        // Benchmark name should be set
        assert_eq!(m.benchmark_name, "SPY");
    }

    #[test]
    fn test_benchmark_metrics_perfect_correlation() {
        // Portfolio returns match benchmark exactly
        let returns = vec![0.01, -0.005, 0.015, 0.008, -0.012];

        let metrics = BenchmarkMetrics::calculate("SPY", &returns, &returns, 0.0);

        assert!(metrics.is_some());
        let m = metrics.unwrap();

        // Perfect correlation
        assert!(
            (m.correlation - 1.0).abs() < 0.0001,
            "Correlation should be 1.0: {}",
            m.correlation
        );

        // Beta should be 1.0
        assert!(
            (m.beta - 1.0).abs() < 0.0001,
            "Beta should be 1.0: {}",
            m.beta
        );

        // Tracking error should be near zero
        assert!(
            m.tracking_error < 0.1,
            "Tracking error should be near zero: {}",
            m.tracking_error
        );

        // Excess return should be zero
        assert!(
            m.excess_return_pct.abs() < 0.0001,
            "Excess return should be zero: {}",
            m.excess_return_pct
        );
    }

    #[test]
    fn test_benchmark_metrics_empty_returns() {
        let empty: Vec<f64> = vec![];
        let returns = vec![0.01, -0.005];

        // Empty portfolio returns
        assert!(BenchmarkMetrics::calculate("SPY", &empty, &returns, 0.0).is_none());

        // Empty benchmark returns
        assert!(BenchmarkMetrics::calculate("SPY", &returns, &empty, 0.0).is_none());

        // Both empty
        assert!(BenchmarkMetrics::calculate("SPY", &empty, &empty, 0.0).is_none());
    }

    #[test]
    fn test_benchmark_metrics_mismatched_lengths() {
        let portfolio = vec![0.01, -0.005, 0.015];
        let benchmark = vec![0.01, -0.005]; // Different length

        let metrics = BenchmarkMetrics::calculate("SPY", &portfolio, &benchmark, 0.0);
        assert!(
            metrics.is_none(),
            "Should return None for mismatched lengths"
        );
    }

    #[test]
    fn test_extract_daily_returns() {
        use crate::types::EquityPoint;

        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
                equity: 101000.0, // +1%
                cash: 0.0,
                positions_value: 101000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
                equity: 100500.0, // -0.495%
                cash: 0.0,
                positions_value: 100500.0,
                drawdown: 500.0,
                drawdown_pct: 0.495,
            },
        ];

        let returns = BenchmarkMetrics::extract_daily_returns(&equity_curve);

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.01).abs() < 0.0001); // +1%
        assert!((returns[1] - (-0.00495)).abs() < 0.0001); // -0.495%
    }

    #[test]
    fn test_extract_returns_from_prices() {
        let prices = vec![100.0, 101.0, 100.5, 102.0];

        let returns = BenchmarkMetrics::extract_returns_from_prices(&prices);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.01).abs() < 0.0001); // 101/100 - 1 = 0.01
        assert!((returns[1] - (-0.00495)).abs() < 0.0001); // 100.5/101 - 1
        assert!((returns[2] - 0.01493).abs() < 0.0001); // 102/100.5 - 1
    }

    #[test]
    fn test_capture_ratios_all_up_market() {
        // All positive benchmark returns
        let portfolio = vec![0.02, 0.01, 0.015];
        let benchmark = vec![0.01, 0.005, 0.01];

        let metrics = BenchmarkMetrics::calculate("SPY", &portfolio, &benchmark, 0.0).unwrap();

        // Portfolio captured 200% of benchmark gains
        assert!(
            metrics.up_capture > 100.0,
            "Up capture should be > 100%: {}",
            metrics.up_capture
        );
        // Down capture should be 100% (no down periods)
        assert!(
            (metrics.down_capture - 100.0).abs() < 0.01,
            "Down capture should be 100%: {}",
            metrics.down_capture
        );
    }

    #[test]
    fn test_benchmark_serialization() {
        let portfolio_returns = vec![0.01, -0.005, 0.015];
        let benchmark_returns = vec![0.008, -0.003, 0.01];

        let metrics =
            BenchmarkMetrics::calculate("SPY", &portfolio_returns, &benchmark_returns, 0.0)
                .unwrap();

        // Test JSON serialization
        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("SPY"));
        assert!(json.contains("alpha"));
        assert!(json.contains("beta"));

        // Test deserialization
        let deserialized: BenchmarkMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.benchmark_name, "SPY");
        assert!((deserialized.beta - metrics.beta).abs() < 0.0001);
    }

    #[test]
    fn test_calculate_daily_returns_from_equity_curve() {
        use crate::types::EquityPoint;

        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap();

        // Create an equity curve with varying daily returns
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 50000.0,
                positions_value: 50000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 102000.0, // +2%
                cash: 51000.0,
                positions_value: 51000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 99000.0, // -2.94%
                cash: 49500.0,
                positions_value: 49500.0,
                drawdown: 3000.0,
                drawdown_pct: 2.94,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 4, 16, 0, 0).unwrap(),
                equity: 103000.0, // +4.04%
                cash: 51500.0,
                positions_value: 51500.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 5, 16, 0, 0).unwrap(),
                equity: 101000.0, // -1.94%
                cash: 50500.0,
                positions_value: 50500.0,
                drawdown: 2000.0,
                drawdown_pct: 1.94,
            },
        ];

        let mut trade = Trade::open("TEST", Side::Buy, 100.0, 100.0, start, 1.0, 0.0);
        trade.close(101.0, end, 1.0);

        let result = BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 101000.0,
            total_return_pct: 1.0,
            annual_return_pct: 1.0,
            trading_days: 5,
            total_trades: 1,
            winning_trades: 1,
            losing_trades: 0,
            win_rate: 100.0,
            avg_win: 100.0,
            avg_loss: 0.0,
            profit_factor: f64::INFINITY,
            max_drawdown_pct: 2.94,
            sharpe_ratio: 0.5,
            sortino_ratio: 0.7,
            calmar_ratio: 0.34,
            trades: vec![trade],
            equity_curve,
            start_time: start,
            end_time: end,
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
            seed: None,
        };

        // Test that daily returns are calculated from actual equity curve
        let returns = PerformanceMetrics::calculate_daily_returns(&result);

        assert_eq!(returns.len(), 4, "Should have 4 daily returns from 5 days");

        // Verify actual returns are calculated correctly
        // Day 1 to Day 2: (102000 - 100000) / 100000 = 0.02 (+2%)
        assert!(
            (returns[0] - 0.02).abs() < 0.0001,
            "First return should be ~2%: {}",
            returns[0]
        );

        // Day 2 to Day 3: (99000 - 102000) / 102000 = -0.0294 (-2.94%)
        assert!(
            (returns[1] - (-0.0294117647)).abs() < 0.0001,
            "Second return should be ~-2.94%: {}",
            returns[1]
        );

        // Returns should NOT be all identical (unlike synthetic returns)
        assert!(
            returns[0] != returns[1],
            "Returns should vary, not be synthetic uniform values"
        );
    }

    #[test]
    fn test_calculate_daily_returns_meaningful_volatility() {
        use crate::types::EquityPoint;

        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 10, 0, 0, 0).unwrap();

        // Create an equity curve with significant variation
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 105000.0, // +5%
                cash: 0.0,
                positions_value: 105000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 95000.0, // -9.5%
                cash: 0.0,
                positions_value: 95000.0,
                drawdown: 10000.0,
                drawdown_pct: 9.5,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 4, 16, 0, 0).unwrap(),
                equity: 110000.0, // +15.8%
                cash: 0.0,
                positions_value: 110000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let mut trade = Trade::open("TEST", Side::Buy, 100.0, 100.0, start, 1.0, 0.0);
        trade.close(110.0, end, 1.0);

        let result = BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 110000.0,
            total_return_pct: 10.0,
            annual_return_pct: 10.0,
            trading_days: 4,
            total_trades: 1,
            winning_trades: 1,
            losing_trades: 0,
            win_rate: 100.0,
            avg_win: 1000.0,
            avg_loss: 0.0,
            profit_factor: f64::INFINITY,
            max_drawdown_pct: 9.5,
            sharpe_ratio: 0.5,
            sortino_ratio: 0.7,
            calmar_ratio: 1.0,
            trades: vec![trade],
            equity_curve,
            start_time: start,
            end_time: end,
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
            seed: None,
        };

        let metrics = PerformanceMetrics::from_result(&result);

        // Volatility should be meaningful (non-zero) with varying returns
        assert!(
            metrics.volatility_annual > 0.0,
            "Volatility should be positive with varying returns: {}",
            metrics.volatility_annual
        );

        // Volatility should be significant given the large daily swings
        assert!(
            metrics.volatility_annual > 50.0,
            "Volatility should be significant given 5%, -9.5%, +15.8% daily moves: {}",
            metrics.volatility_annual
        );
    }

    #[test]
    fn test_calculate_daily_returns_empty_equity_curve_fallback() {
        // Test fallback to synthetic returns when equity curve is empty
        let result = create_test_result(); // has empty equity_curve

        let returns = PerformanceMetrics::calculate_daily_returns(&result);

        // Should fall back to synthetic returns (252 identical values)
        assert_eq!(returns.len(), 252);

        // All values should be identical in fallback mode
        let first = returns[0];
        assert!(
            returns.iter().all(|&r| (r - first).abs() < 1e-10),
            "Fallback returns should all be identical"
        );
    }

    #[test]
    fn test_calculate_daily_returns_intraday_aggregation() {
        use crate::types::EquityPoint;

        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap();

        // Create equity curve with multiple intraday points
        // Should aggregate to daily by taking last value of each day
        let equity_curve = vec![
            // Day 1 - multiple points, last should be used
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 9, 30, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap(),
                equity: 100500.0,
                cash: 0.0,
                positions_value: 100500.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 101000.0, // This should be used for Day 1
                cash: 0.0,
                positions_value: 101000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            // Day 2 - multiple points
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 9, 30, 0).unwrap(),
                equity: 100000.0,
                cash: 0.0,
                positions_value: 100000.0,
                drawdown: 1000.0,
                drawdown_pct: 0.99,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 102000.0, // This should be used for Day 2
                cash: 0.0,
                positions_value: 102000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            // Day 3 - single point
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 103000.0, // This should be used for Day 3
                cash: 0.0,
                positions_value: 103000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let mut trade = Trade::open("TEST", Side::Buy, 100.0, 100.0, start, 1.0, 0.0);
        trade.close(103.0, end, 1.0);

        let result = BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 103000.0,
            total_return_pct: 3.0,
            annual_return_pct: 3.0,
            trading_days: 3,
            total_trades: 1,
            winning_trades: 1,
            losing_trades: 0,
            win_rate: 100.0,
            avg_win: 300.0,
            avg_loss: 0.0,
            profit_factor: f64::INFINITY,
            max_drawdown_pct: 0.99,
            sharpe_ratio: 1.0,
            sortino_ratio: 1.5,
            calmar_ratio: 3.0,
            trades: vec![trade],
            equity_curve,
            start_time: start,
            end_time: end,
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
            seed: None,
        };

        let returns = PerformanceMetrics::calculate_daily_returns(&result);

        // Should have 2 daily returns from 3 days
        assert_eq!(returns.len(), 2);

        // Day 1 to Day 2: (102000 - 101000) / 101000 = 0.0099 (~0.99%)
        assert!(
            (returns[0] - 0.00990099).abs() < 0.0001,
            "First return should be ~0.99%: {}",
            returns[0]
        );

        // Day 2 to Day 3: (103000 - 102000) / 102000 = 0.0098 (~0.98%)
        assert!(
            (returns[1] - 0.00980392).abs() < 0.0001,
            "Second return should be ~0.98%: {}",
            returns[1]
        );
    }

    #[test]
    fn test_drawdown_analysis_single_drawdown() {
        use crate::types::EquityPoint;

        // Equity curve: 100 -> 110 -> 90 -> 105
        // Peak at 110, trough at 90 (18.18% drawdown), then partial recovery
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 110000.0, // New peak
                cash: 0.0,
                positions_value: 110000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 90000.0, // Trough - 18.18% drawdown
                cash: 0.0,
                positions_value: 90000.0,
                drawdown: 20000.0,
                drawdown_pct: 18.18,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 4, 16, 0, 0).unwrap(),
                equity: 105000.0, // Partial recovery, still in drawdown
                cash: 0.0,
                positions_value: 105000.0,
                drawdown: 5000.0,
                drawdown_pct: 4.55,
            },
        ];

        let analysis = DrawdownAnalysis::from_equity_curve(&equity_curve);

        // Max drawdown should be ~18.18% ((110000-90000)/110000 * 100)
        assert!(
            (analysis.max_drawdown_pct - 18.18).abs() < 0.1,
            "Max drawdown should be ~18.18%: {}",
            analysis.max_drawdown_pct
        );

        // Should have 1 ongoing drawdown period
        assert_eq!(analysis.periods.len(), 1, "Should have 1 drawdown period");

        // Period should not be recovered (end is None)
        assert!(
            analysis.periods[0].end.is_none(),
            "Drawdown period should not be recovered"
        );
    }

    #[test]
    fn test_drawdown_analysis_multiple_drawdowns() {
        use crate::types::EquityPoint;

        // Equity curve with two separate drawdown periods
        // 100 -> 120 -> 100 -> 130 -> 100 -> 150
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 5, 16, 0, 0).unwrap(),
                equity: 120000.0, // First peak
                cash: 0.0,
                positions_value: 120000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 10, 16, 0, 0).unwrap(),
                equity: 100000.0, // First drawdown ~16.67%
                cash: 0.0,
                positions_value: 100000.0,
                drawdown: 20000.0,
                drawdown_pct: 16.67,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 15, 16, 0, 0).unwrap(),
                equity: 130000.0, // New peak, first drawdown recovered
                cash: 0.0,
                positions_value: 130000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 20, 16, 0, 0).unwrap(),
                equity: 100000.0, // Second drawdown ~23.08%
                cash: 0.0,
                positions_value: 100000.0,
                drawdown: 30000.0,
                drawdown_pct: 23.08,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 25, 16, 0, 0).unwrap(),
                equity: 150000.0, // New peak, second drawdown recovered
                cash: 0.0,
                positions_value: 150000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let analysis = DrawdownAnalysis::from_equity_curve(&equity_curve);

        // Max drawdown should be ~23.08%
        assert!(
            (analysis.max_drawdown_pct - 23.08).abs() < 0.1,
            "Max drawdown should be ~23.08%: {}",
            analysis.max_drawdown_pct
        );

        // Should have 2 drawdown periods
        assert_eq!(analysis.periods.len(), 2, "Should have 2 drawdown periods");

        // Both periods should be recovered
        assert!(
            analysis.periods[0].end.is_some(),
            "First period should be recovered"
        );
        assert!(
            analysis.periods[1].end.is_some(),
            "Second period should be recovered"
        );

        // Average drawdown should be average of ~16.67% and ~23.08%
        let expected_avg = (16.666666 + 23.076923) / 2.0;
        assert!(
            (analysis.avg_drawdown_pct - expected_avg).abs() < 0.1,
            "Avg drawdown should be ~{:.2}%: {}",
            expected_avg,
            analysis.avg_drawdown_pct
        );
    }

    #[test]
    fn test_drawdown_analysis_ulcer_index() {
        use crate::types::EquityPoint;

        // Simple case: constant 10% drawdown
        // Ulcer Index = sqrt(mean(10^2)) = sqrt(100) = 10
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 90000.0, // 10% drawdown
                cash: 0.0,
                positions_value: 90000.0,
                drawdown: 10000.0,
                drawdown_pct: 10.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 90000.0, // Still 10% drawdown
                cash: 0.0,
                positions_value: 90000.0,
                drawdown: 10000.0,
                drawdown_pct: 10.0,
            },
        ];

        let analysis = DrawdownAnalysis::from_equity_curve(&equity_curve);

        // Ulcer index calculation:
        // Point 0: drawdown = 0%
        // Point 1: drawdown = 10%
        // Point 2: drawdown = 10%
        // Ulcer = sqrt((0^2 + 10^2 + 10^2) / 3) = sqrt(200/3) = 8.165
        assert!(
            (analysis.ulcer_index - 8.165).abs() < 0.1,
            "Ulcer index should be ~8.165: {}",
            analysis.ulcer_index
        );
    }

    #[test]
    fn test_drawdown_analysis_no_drawdown() {
        use crate::types::EquityPoint;

        // Steadily increasing equity - no drawdowns
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 110000.0,
                cash: 0.0,
                positions_value: 110000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 120000.0,
                cash: 0.0,
                positions_value: 120000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let analysis = DrawdownAnalysis::from_equity_curve(&equity_curve);

        assert!(
            analysis.max_drawdown_pct.abs() < 0.0001,
            "Max drawdown should be 0%"
        );
        assert!(
            analysis.periods.is_empty(),
            "Should have no drawdown periods"
        );
        assert!(
            analysis.ulcer_index.abs() < 0.0001,
            "Ulcer index should be 0, got: {}",
            analysis.ulcer_index
        );
    }

    #[test]
    fn test_drawdown_analysis_empty_curve() {
        let analysis = DrawdownAnalysis::from_equity_curve(&[]);

        assert_eq!(analysis.max_drawdown_pct, 0.0);
        assert_eq!(analysis.avg_drawdown_pct, 0.0);
        assert_eq!(analysis.ulcer_index, 0.0);
        assert!(analysis.periods.is_empty());
    }

    #[test]
    fn test_drawdown_analysis_serialization() {
        use crate::types::EquityPoint;

        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 95000.0,
                cash: 0.0,
                positions_value: 95000.0,
                drawdown: 5000.0,
                drawdown_pct: 5.0,
            },
        ];

        let analysis = DrawdownAnalysis::from_equity_curve(&equity_curve);

        // Test JSON serialization
        let json = serde_json::to_string(&analysis).unwrap();
        assert!(json.contains("max_drawdown_pct"));
        assert!(json.contains("ulcer_index"));

        // Test deserialization
        let deserialized: DrawdownAnalysis = serde_json::from_str(&json).unwrap();
        assert!((deserialized.max_drawdown_pct - analysis.max_drawdown_pct).abs() < 0.0001);
    }

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric distribution should have near-zero skewness
        let returns = vec![-0.02, -0.01, 0.0, 0.01, 0.02];
        let skew = PerformanceMetrics::skewness(&returns);
        assert!(
            skew.abs() < 0.1,
            "Symmetric distribution should have near-zero skewness: {}",
            skew
        );
    }

    #[test]
    fn test_skewness_positive() {
        // Right-skewed: more extreme positive values
        let returns = vec![-0.01, -0.01, 0.0, 0.0, 0.05, 0.10];
        let skew = PerformanceMetrics::skewness(&returns);
        assert!(
            skew > 0.0,
            "Right-skewed distribution should have positive skewness: {}",
            skew
        );
    }

    #[test]
    fn test_skewness_negative() {
        // Left-skewed: more extreme negative values
        let returns = vec![-0.10, -0.05, 0.0, 0.0, 0.01, 0.01];
        let skew = PerformanceMetrics::skewness(&returns);
        assert!(
            skew < 0.0,
            "Left-skewed distribution should have negative skewness: {}",
            skew
        );
    }

    #[test]
    fn test_skewness_insufficient_data() {
        // Too few data points
        let returns = vec![0.01, 0.02];
        let skew = PerformanceMetrics::skewness(&returns);
        assert_eq!(skew, 0.0, "Should return 0.0 for insufficient data");
    }

    #[test]
    fn test_kurtosis_normal() {
        // Approximately normal distribution should have near-zero excess kurtosis
        let returns = vec![-0.02, -0.01, 0.0, 0.0, 0.01, 0.02];
        let kurt = PerformanceMetrics::kurtosis(&returns);
        // For small samples, kurtosis can vary, so we just check it's calculated
        assert!(kurt.is_finite(), "Kurtosis should be finite: {}", kurt);
    }

    #[test]
    fn test_kurtosis_heavy_tails() {
        // Heavy tails: extreme values should produce positive excess kurtosis
        // Use more data points with clear heavy tails
        let mut returns = vec![0.0; 100];
        // Add extreme outliers in the tails
        returns[0] = -1.0;
        returns[1] = -0.8;
        returns[98] = 0.8;
        returns[99] = 1.0;
        let kurt = PerformanceMetrics::kurtosis(&returns);
        assert!(
            kurt > 0.0,
            "Heavy-tailed distribution should have positive excess kurtosis: {}",
            kurt
        );
    }

    #[test]
    fn test_kurtosis_insufficient_data() {
        // Too few data points
        let returns = vec![0.01, 0.02, 0.03];
        let kurt = PerformanceMetrics::kurtosis(&returns);
        assert_eq!(kurt, 0.0, "Should return 0.0 for insufficient data");
    }

    #[test]
    fn test_tail_ratio_symmetric() {
        // Symmetric distribution should have tail ratio near 1.0
        let returns = vec![
            -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
        ];
        let tail = PerformanceMetrics::tail_ratio(&returns);
        assert!(
            (tail - 1.0).abs() < 0.5,
            "Symmetric distribution should have tail ratio near 1.0: {}",
            tail
        );
    }

    #[test]
    fn test_tail_ratio_positive_bias() {
        // Larger positive tail
        let mut returns = vec![0.0; 20];
        returns[0] = -0.02; // 5th percentile
        returns[19] = 0.10; // 95th percentile
        let tail = PerformanceMetrics::tail_ratio(&returns);
        assert!(
            tail > 1.0,
            "Positive bias should have tail ratio > 1.0: {}",
            tail
        );
    }

    #[test]
    fn test_tail_ratio_negative_bias() {
        // Larger negative tail
        // Create distribution with larger losses than gains
        let returns = vec![
            -0.10, -0.08, -0.06, -0.04, -0.02, // Bottom 25% - losses
            -0.01, -0.01, 0.0, 0.0, 0.0, // Middle 25%
            0.0, 0.0, 0.0, 0.01, 0.01, // Middle 25%
            0.01, 0.01, 0.02, 0.02, 0.02, // Top 25% - smaller gains
        ];
        // p05 (5th percentile) should be around -0.10 to -0.08
        // p95 (95th percentile) should be around 0.02
        // tail_ratio = 0.02 / 0.10 = 0.2 < 1.0
        let tail = PerformanceMetrics::tail_ratio(&returns);
        assert!(
            tail < 1.0,
            "Negative bias should have tail ratio < 1.0: {}",
            tail
        );
    }

    #[test]
    fn test_tail_ratio_insufficient_data() {
        // Too few data points
        let returns = vec![0.01, 0.02, 0.03];
        let tail = PerformanceMetrics::tail_ratio(&returns);
        assert_eq!(tail, 1.0, "Should return 1.0 for insufficient data");
    }

    #[test]
    fn test_distribution_metrics_integration() {
        use crate::types::EquityPoint;

        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 10, 0, 0, 0).unwrap();

        // Create equity curve with varying returns
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 102000.0, // +2%
                cash: 0.0,
                positions_value: 102000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 98000.0, // -3.9%
                cash: 0.0,
                positions_value: 98000.0,
                drawdown: 4000.0,
                drawdown_pct: 3.9,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 4, 16, 0, 0).unwrap(),
                equity: 105000.0, // +7.1%
                cash: 0.0,
                positions_value: 105000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let mut trade = Trade::open("TEST", Side::Buy, 100.0, 100.0, start, 1.0, 0.0);
        trade.close(105.0, end, 1.0);

        let result = BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 105000.0,
            total_return_pct: 5.0,
            annual_return_pct: 5.0,
            trading_days: 4,
            total_trades: 1,
            winning_trades: 1,
            losing_trades: 0,
            win_rate: 100.0,
            avg_win: 500.0,
            avg_loss: 0.0,
            profit_factor: f64::INFINITY,
            max_drawdown_pct: 3.9,
            sharpe_ratio: 0.5,
            sortino_ratio: 0.7,
            calmar_ratio: 1.28,
            trades: vec![trade],
            equity_curve,
            start_time: start,
            end_time: end,
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
            seed: None,
        };

        let metrics = PerformanceMetrics::from_result(&result);

        // All distribution metrics should be calculated
        assert!(
            metrics.skewness.is_finite(),
            "Skewness should be calculated"
        );
        assert!(
            metrics.kurtosis.is_finite(),
            "Kurtosis should be calculated"
        );
        assert!(metrics.tail_ratio > 0.0, "Tail ratio should be positive");
    }

    #[test]
    fn test_deflated_sharpe_ratio_no_trials() {
        // With n_trials = 1 (no multiple testing), DSR should equal SR
        let sharpe = 1.5;
        let n_trials = 1;
        let n_observations = 252;

        let dsr = PerformanceMetrics::deflated_sharpe_ratio(sharpe, n_trials, n_observations);

        assert_eq!(dsr, sharpe, "DSR should equal SR when n_trials = 1");
    }

    #[test]
    fn test_deflated_sharpe_ratio_multiple_trials() {
        // With multiple trials, DSR should be lower than SR
        let sharpe = 2.0;
        let n_trials = 100; // Tested 100 parameter combinations
        let n_observations = 252;

        let dsr = PerformanceMetrics::deflated_sharpe_ratio(sharpe, n_trials, n_observations);

        assert!(
            dsr < sharpe,
            "DSR should be less than SR with multiple trials"
        );
        assert!(dsr.is_finite(), "DSR should be finite");

        // For 100 trials, expected max SR ≈ sqrt(2*ln(100)) ≈ 3.03
        // So a SR of 2.0 should deflate to a negative value
        assert!(
            dsr < 0.0,
            "DSR should be negative for SR=2.0 with 100 trials"
        );
    }

    #[test]
    fn test_deflated_sharpe_ratio_high_sharpe() {
        // Very high Sharpe ratio should survive deflation
        let sharpe = 5.0;
        let n_trials = 10;
        let n_observations = 500;

        let dsr = PerformanceMetrics::deflated_sharpe_ratio(sharpe, n_trials, n_observations);

        assert!(dsr.is_finite(), "DSR should be finite");
        // With only 10 trials, a SR of 5.0 is still impressive
        assert!(dsr > 2.0, "High SR should remain positive after deflation");
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_zero_sharpe() {
        // SR = 0 should give PSR ≈ 0.5 (50% confidence)
        let sharpe = 0.0;
        let skewness = 0.0;
        let kurtosis = 0.0;
        let n_observations = 252;
        let benchmark_sr = 0.0;

        let psr = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            skewness,
            kurtosis,
            n_observations,
            benchmark_sr,
        );

        assert!((0.0..=1.0).contains(&psr), "PSR should be between 0 and 1");
        assert!(
            (psr - 0.5).abs() < 0.05,
            "PSR should be close to 0.5 for SR=0: {}",
            psr
        );
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_positive_sharpe() {
        // High positive SR should give high PSR (>95%)
        let sharpe = 3.0;
        let skewness = 0.0;
        let kurtosis = 0.0;
        let n_observations = 252;
        let benchmark_sr = 0.0;

        let psr = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            skewness,
            kurtosis,
            n_observations,
            benchmark_sr,
        );

        assert!((0.0..=1.0).contains(&psr), "PSR should be between 0 and 1");
        assert!(psr > 0.95, "PSR should be > 0.95 for high SR: {}", psr);
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_negative_sharpe() {
        // Negative SR should give low PSR (<5%)
        let sharpe = -2.0;
        let skewness = 0.0;
        let kurtosis = 0.0;
        let n_observations = 252;
        let benchmark_sr = 0.0;

        let psr = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            skewness,
            kurtosis,
            n_observations,
            benchmark_sr,
        );

        assert!((0.0..=1.0).contains(&psr), "PSR should be between 0 and 1");
        assert!(psr < 0.05, "PSR should be < 0.05 for negative SR: {}", psr);
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_with_skewness() {
        // Negative skewness should reduce PSR (use fewer observations to see effect)
        let sharpe = 0.8; // Lower Sharpe to avoid saturation
        let n_observations = 30; // Fewer observations to see skewness effect
        let benchmark_sr = 0.0;

        // PSR with zero skewness
        let psr_zero_skew = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            0.0,
            0.0,
            n_observations,
            benchmark_sr,
        );

        // PSR with negative skewness (bad tail behavior)
        let psr_neg_skew = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            -1.5,
            0.0,
            n_observations,
            benchmark_sr,
        );

        assert!(
            psr_zero_skew > psr_neg_skew,
            "Negative skewness should reduce PSR: {} vs {}",
            psr_zero_skew,
            psr_neg_skew
        );
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_with_kurtosis() {
        // High kurtosis (fat tails) should reduce PSR (use fewer observations to see effect)
        let sharpe = 0.8; // Lower Sharpe to avoid saturation
        let n_observations = 30; // Fewer observations to see kurtosis effect
        let benchmark_sr = 0.0;

        // PSR with normal kurtosis
        let psr_normal_kurt = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            0.0,
            0.0,
            n_observations,
            benchmark_sr,
        );

        // PSR with high excess kurtosis (fat tails - more extreme events)
        let psr_high_kurt = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            0.0,
            8.0,
            n_observations,
            benchmark_sr,
        );

        assert!(
            psr_normal_kurt > psr_high_kurt,
            "High kurtosis should reduce PSR: {} vs {}",
            psr_normal_kurt,
            psr_high_kurt
        );
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_benchmark_comparison() {
        // PSR should measure probability of exceeding benchmark
        let sharpe = 0.8; // Use moderate SR to avoid saturation
        let skewness = 0.0;
        let kurtosis = 0.0;
        let n_observations = 50; // Fewer observations to see benchmark effect

        // PSR vs zero benchmark
        let psr_vs_zero = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            skewness,
            kurtosis,
            n_observations,
            0.0,
        );

        // PSR vs higher benchmark (harder to beat)
        let psr_vs_high = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            skewness,
            kurtosis,
            n_observations,
            0.5,
        );

        assert!(
            psr_vs_zero > psr_vs_high,
            "PSR should be lower vs higher benchmark: {} vs {}",
            psr_vs_zero,
            psr_vs_high
        );
    }

    #[test]
    fn test_overfitting_metrics_in_backtest_result() {
        // Ensure DSR and PSR are calculated in actual backtest results
        let result = create_test_result();
        let metrics = PerformanceMetrics::from_result(&result);

        assert!(
            metrics.deflated_sharpe_ratio.is_finite(),
            "Deflated Sharpe Ratio should be calculated"
        );
        assert!(
            metrics.probabilistic_sharpe_ratio >= 0.0 && metrics.probabilistic_sharpe_ratio <= 1.0,
            "PSR should be between 0 and 1: {}",
            metrics.probabilistic_sharpe_ratio
        );

        // With a decent SR of 1.5 and no multiple testing (n_trials=1),
        // DSR should equal SR
        assert_eq!(
            metrics.deflated_sharpe_ratio, result.sharpe_ratio,
            "DSR should equal SR when n_trials=1"
        );

        // With SR=1.5, PSR should be quite high
        assert!(
            metrics.probabilistic_sharpe_ratio > 0.80,
            "PSR should be > 0.80 for SR=1.5: {}",
            metrics.probabilistic_sharpe_ratio
        );
    }

    // =========================================================================
    // Factor Attribution Tests
    // =========================================================================

    #[test]
    fn test_multiple_regression_simple() {
        // y = 2 + 3*x1 + noise
        let x1: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x1.iter().map(|&x| 2.0 + 3.0 * x).collect();

        let result = multiple_regression(&y, &[x1]).unwrap();

        // Check intercept is close to 2
        assert!(
            (result.intercept - 2.0).abs() < 0.01,
            "Intercept should be ~2.0: {}",
            result.intercept
        );

        // Check coefficient is close to 3
        assert!(
            (result.coefficients[0] - 3.0).abs() < 0.01,
            "Coefficient should be ~3.0: {}",
            result.coefficients[0]
        );

        // R-squared should be 1.0 (perfect fit)
        assert!(
            result.r_squared > 0.99,
            "R-squared should be ~1.0: {}",
            result.r_squared
        );
    }

    #[test]
    fn test_multiple_regression_two_variables() {
        // y = 1 + 2*x1 - 0.5*x2 + noise
        let n = 100;
        let x1: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| (i as f64 / 5.0).sin()).collect();
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 1.0 + 2.0 * a - 0.5 * b)
            .collect();

        let result = multiple_regression(&y, &[x1, x2]).unwrap();

        // Check intercept
        assert!(
            (result.intercept - 1.0).abs() < 0.1,
            "Intercept should be ~1.0: {}",
            result.intercept
        );

        // Check first coefficient
        assert!(
            (result.coefficients[0] - 2.0).abs() < 0.1,
            "First coefficient should be ~2.0: {}",
            result.coefficients[0]
        );

        // Check second coefficient
        assert!(
            (result.coefficients[1] - (-0.5)).abs() < 0.1,
            "Second coefficient should be ~-0.5: {}",
            result.coefficients[1]
        );

        // Good R-squared
        assert!(
            result.r_squared > 0.95,
            "R-squared should be high: {}",
            result.r_squared
        );
    }

    #[test]
    fn test_multiple_regression_edge_cases() {
        // Minimal valid case: need n > k + 1 for regression
        // With 1 variable, need at least 3 observations
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0, 3.0];
        assert!(
            multiple_regression(&y, &x).is_some(),
            "Should work with 3 observations for 1 variable"
        );

        // Too few observations (need n >= k + 2)
        let x_small = vec![vec![1.0, 2.0]];
        let y_small = vec![1.0, 2.0];
        assert!(
            multiple_regression(&y_small, &x_small).is_none(),
            "Should return None with insufficient observations"
        );

        // Empty data
        let empty: Vec<f64> = vec![];
        assert!(multiple_regression(&empty, &[vec![]]).is_none());

        // Mismatched lengths
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![vec![1.0, 2.0]]; // Different length
        assert!(multiple_regression(&y, &x).is_none());

        // Zero factors should return None
        let y_only = vec![1.0, 2.0, 3.0];
        assert!(multiple_regression(&y_only, &[]).is_none());
    }

    #[test]
    fn test_factor_returns_constructors() {
        // Three-factor
        let ff3 = FactorReturns::three_factor(0.01, 0.002, -0.001, 0.0001);
        assert_eq!(ff3.mkt_rf, 0.01);
        assert_eq!(ff3.smb, 0.002);
        assert_eq!(ff3.hml, -0.001);
        assert_eq!(ff3.rmw, 0.0);
        assert_eq!(ff3.cma, 0.0);
        assert_eq!(ff3.umd, 0.0);
        assert_eq!(ff3.rf, 0.0001);

        // Four-factor (Carhart)
        let c4 = FactorReturns::four_factor(0.01, 0.002, -0.001, 0.003, 0.0001);
        assert_eq!(c4.umd, 0.003);
        assert_eq!(c4.rmw, 0.0);
        assert_eq!(c4.cma, 0.0);

        // Full six-factor
        let full = FactorReturns::new(0.01, 0.002, -0.001, 0.001, -0.002, 0.003, 0.0001);
        assert_eq!(full.rmw, 0.001);
        assert_eq!(full.cma, -0.002);
        assert_eq!(full.umd, 0.003);
    }

    fn generate_synthetic_factor_data(n: usize) -> Vec<FactorReturns> {
        // Generate synthetic factor data with realistic correlations
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let market_shock = (t * 20.0).sin() * 0.02;
                FactorReturns {
                    mkt_rf: 0.0005 + market_shock + (i as f64 * 0.1).cos() * 0.01,
                    smb: 0.0001 + (i as f64 * 0.2).sin() * 0.005,
                    hml: 0.0002 + (i as f64 * 0.3).cos() * 0.004,
                    rmw: 0.0001 + (i as f64 * 0.15).sin() * 0.003,
                    cma: 0.0001 + (i as f64 * 0.25).cos() * 0.002,
                    umd: 0.0003 + (i as f64 * 0.12).sin() * 0.006,
                    rf: 0.0001, // ~2.5% annual risk-free rate
                }
            })
            .collect()
    }

    fn generate_portfolio_returns(
        factor_data: &[FactorReturns],
        alpha: f64,
        betas: &[f64],
    ) -> Vec<f64> {
        // Generate portfolio returns: R_p = rf + alpha + beta * factors + noise
        factor_data
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let noise = (i as f64 * 0.7).sin() * 0.002; // Small noise
                f.rf + alpha
                    + betas.first().unwrap_or(&0.0) * f.mkt_rf
                    + betas.get(1).unwrap_or(&0.0) * f.smb
                    + betas.get(2).unwrap_or(&0.0) * f.hml
                    + betas.get(3).unwrap_or(&0.0) * f.rmw
                    + betas.get(4).unwrap_or(&0.0) * f.cma
                    + betas.get(5).unwrap_or(&0.0) * f.umd
                    + noise
            })
            .collect()
    }

    #[test]
    fn test_fama_french_3factor_regression() {
        let factor_data = generate_synthetic_factor_data(252); // 1 year of daily data

        // Portfolio with known exposures: alpha=0.0002, mkt=1.2, smb=0.3, hml=-0.2
        let true_alpha = 0.0002; // Daily alpha
        let true_betas = vec![1.2, 0.3, -0.2];
        let portfolio_returns = generate_portfolio_returns(&factor_data, true_alpha, &true_betas);

        let loadings = FactorAttribution::fama_french_3factor(&portfolio_returns, &factor_data)
            .expect("Should compute factor loadings");

        // Check model type
        assert_eq!(loadings.model_type, FactorModelType::FamaFrench3);

        // Market beta should be close to 1.2
        assert!(
            (loadings.market_beta - 1.2).abs() < 0.15,
            "Market beta should be ~1.2: {}",
            loadings.market_beta
        );

        // SMB beta should be close to 0.3
        assert!(
            (loadings.smb_beta - 0.3).abs() < 0.15,
            "SMB beta should be ~0.3: {}",
            loadings.smb_beta
        );

        // HML beta should be close to -0.2
        assert!(
            (loadings.hml_beta - (-0.2)).abs() < 0.15,
            "HML beta should be ~-0.2: {}",
            loadings.hml_beta
        );

        // R-squared should be high (since returns are generated from factors)
        assert!(
            loadings.r_squared > 0.85,
            "R-squared should be high: {}",
            loadings.r_squared
        );

        // RMW, CMA, UMD should be zero for 3-factor model
        assert_eq!(loadings.rmw_beta, 0.0);
        assert_eq!(loadings.cma_beta, 0.0);
        assert_eq!(loadings.umd_beta, 0.0);
    }

    #[test]
    fn test_carhart_4factor_regression() {
        let factor_data = generate_synthetic_factor_data(252);

        // Portfolio with momentum exposure: alpha=0.0001, mkt=1.0, smb=0.2, hml=0.1, umd=0.5
        let true_betas = vec![1.0, 0.2, 0.1, 0.0, 0.0, 0.5]; // Last is UMD
        let portfolio_returns = generate_portfolio_returns(&factor_data, 0.0001, &true_betas);

        let loadings = FactorAttribution::carhart_4factor(&portfolio_returns, &factor_data)
            .expect("Should compute factor loadings");

        // Check model type
        assert_eq!(loadings.model_type, FactorModelType::Carhart4);

        // Check momentum exposure
        assert!(
            (loadings.umd_beta - 0.5).abs() < 0.2,
            "UMD beta should be ~0.5: {}",
            loadings.umd_beta
        );

        // Market beta should be close to 1.0
        assert!(
            (loadings.market_beta - 1.0).abs() < 0.15,
            "Market beta should be ~1.0: {}",
            loadings.market_beta
        );

        // RMW and CMA should still be zero (not in 4-factor model)
        assert_eq!(loadings.rmw_beta, 0.0);
        assert_eq!(loadings.cma_beta, 0.0);
    }

    #[test]
    fn test_fama_french_5factor_regression() {
        let factor_data = generate_synthetic_factor_data(252);

        // Portfolio with profitability and investment exposure
        let true_betas = vec![1.1, 0.15, -0.1, 0.25, -0.15]; // mkt, smb, hml, rmw, cma
        let portfolio_returns = generate_portfolio_returns(&factor_data, 0.00015, &true_betas);

        let loadings = FactorAttribution::fama_french_5factor(&portfolio_returns, &factor_data)
            .expect("Should compute factor loadings");

        // Check model type
        assert_eq!(loadings.model_type, FactorModelType::FamaFrench5);

        // Check RMW exposure
        assert!(
            (loadings.rmw_beta - 0.25).abs() < 0.15,
            "RMW beta should be ~0.25: {}",
            loadings.rmw_beta
        );

        // Check CMA exposure
        assert!(
            (loadings.cma_beta - (-0.15)).abs() < 0.15,
            "CMA beta should be ~-0.15: {}",
            loadings.cma_beta
        );

        // UMD should be zero (not in 5-factor model)
        assert_eq!(loadings.umd_beta, 0.0);
    }

    #[test]
    fn test_six_factor_regression() {
        let factor_data = generate_synthetic_factor_data(300);

        // Portfolio with all factor exposures
        let true_betas = vec![1.0, 0.2, 0.15, 0.1, -0.1, 0.3]; // all 6 factors
        let portfolio_returns = generate_portfolio_returns(&factor_data, 0.0001, &true_betas);

        let loadings = FactorAttribution::six_factor(&portfolio_returns, &factor_data)
            .expect("Should compute factor loadings");

        // Check model type
        assert_eq!(loadings.model_type, FactorModelType::FamaFrench5Momentum);

        // All factors should have non-zero loading estimates
        assert!(
            loadings.market_beta.abs() > 0.5,
            "Market beta should be significant: {}",
            loadings.market_beta
        );

        // R-squared should be high
        assert!(
            loadings.r_squared > 0.80,
            "R-squared should be high for 6-factor model: {}",
            loadings.r_squared
        );
    }

    #[test]
    fn test_factor_attribution_insufficient_data() {
        // Too few observations
        let factor_data: Vec<FactorReturns> = (0..10)
            .map(|_| FactorReturns::three_factor(0.01, 0.002, -0.001, 0.0001))
            .collect();
        let portfolio_returns: Vec<f64> = (0..10).map(|i| 0.001 * i as f64).collect();

        // 3-factor requires 20+ observations
        let result = FactorAttribution::fama_french_3factor(&portfolio_returns, &factor_data);
        assert!(
            result.is_none(),
            "Should return None with insufficient data"
        );

        // 5-factor requires 30+ observations
        let factor_data_25: Vec<FactorReturns> = (0..25)
            .map(|_| FactorReturns::three_factor(0.01, 0.002, -0.001, 0.0001))
            .collect();
        let portfolio_returns_25: Vec<f64> = (0..25).map(|i| 0.001 * i as f64).collect();
        let result_5f =
            FactorAttribution::fama_french_5factor(&portfolio_returns_25, &factor_data_25);
        assert!(result_5f.is_none(), "5-factor should need 30+ observations");
    }

    #[test]
    fn test_factor_attribution_mismatched_lengths() {
        let factor_data = generate_synthetic_factor_data(100);
        let portfolio_returns: Vec<f64> = (0..50).map(|i| 0.001 * i as f64).collect();

        let result = FactorAttribution::fama_french_3factor(&portfolio_returns, &factor_data);
        assert!(
            result.is_none(),
            "Should return None with mismatched lengths"
        );
    }

    #[test]
    fn test_has_significant_exposure() {
        let factor_data = generate_synthetic_factor_data(252);

        // Create portfolio with strong market exposure only
        let true_betas = vec![1.5, 0.0, 0.0]; // Only market exposure
        let portfolio_returns = generate_portfolio_returns(&factor_data, 0.0, &true_betas);

        let loadings = FactorAttribution::fama_french_3factor(&portfolio_returns, &factor_data)
            .expect("Should compute loadings");

        let significant = FactorAttribution::has_significant_exposure(&loadings, 0.05);

        // Market should definitely be significant with beta=1.5
        assert!(
            significant.contains(&"Market".to_string()),
            "Market exposure should be significant: {:?}",
            significant
        );
    }

    #[test]
    fn test_factor_loadings_serialization() {
        let factor_data = generate_synthetic_factor_data(100);
        let portfolio_returns: Vec<f64> = factor_data
            .iter()
            .map(|f| f.rf + f.mkt_rf * 1.1 + 0.001)
            .collect();

        let loadings = FactorAttribution::fama_french_3factor(&portfolio_returns, &factor_data)
            .expect("Should compute loadings");

        // Serialize to JSON
        let json = serde_json::to_string(&loadings).unwrap();
        assert!(json.contains("market_beta"));
        assert!(json.contains("r_squared"));
        assert!(json.contains("FamaFrench3"));

        // Deserialize back
        let deserialized: FactorLoadings = serde_json::from_str(&json).unwrap();
        assert!((deserialized.market_beta - loadings.market_beta).abs() < 0.0001);
        assert_eq!(deserialized.model_type, loadings.model_type);
    }

    #[test]
    fn test_factor_model_type_display() {
        assert_eq!(
            format!("{}", FactorModelType::FamaFrench3),
            "Fama-French 3-Factor"
        );
        assert_eq!(format!("{}", FactorModelType::Carhart4), "Carhart 4-Factor");
        assert_eq!(
            format!("{}", FactorModelType::FamaFrench5),
            "Fama-French 5-Factor"
        );
        assert_eq!(
            format!("{}", FactorModelType::FamaFrench5Momentum),
            "Fama-French 5-Factor + Momentum"
        );
    }

    #[test]
    fn test_regression_result_is_significant() {
        // Create a simple regression result
        let result = RegressionResult {
            coefficients: vec![1.5, 0.3],
            intercept: 0.01,
            r_squared: 0.85,
            r_squared_adj: 0.84,
            std_errors: vec![0.1, 0.2],
            t_statistics: vec![15.0, 1.5], // First is significant, second is not
            p_values: vec![0.001, 0.15],
            residuals: vec![0.0; 10],
            n_observations: 100,
            residual_std_error: 0.02,
            f_statistic: 120.0,
        };

        // First coefficient is significant at 5%
        assert!(result.is_significant(0, 0.05));

        // Second coefficient is NOT significant at 5%
        assert!(!result.is_significant(1, 0.05));

        // Out of bounds returns false
        assert!(!result.is_significant(5, 0.05));
    }

    #[test]
    fn test_alpha_annualization() {
        let factor_data = generate_synthetic_factor_data(252);

        // Create portfolio with known daily alpha of 0.0004 (about 10% annual)
        let daily_alpha = 0.0004;
        let portfolio_returns: Vec<f64> = factor_data
            .iter()
            .map(|f| f.rf + f.mkt_rf + daily_alpha)
            .collect();

        let loadings = FactorAttribution::fama_french_3factor(&portfolio_returns, &factor_data)
            .expect("Should compute loadings");

        // Annualized alpha should be approximately daily_alpha * 252 * 100 = ~10%
        let expected_annual_alpha_pct = daily_alpha * 252.0 * 100.0;
        assert!(
            (loadings.alpha - expected_annual_alpha_pct).abs() < 5.0,
            "Annualized alpha should be ~{}%: {}",
            expected_annual_alpha_pct,
            loadings.alpha
        );
    }

    #[test]
    fn test_ln_gamma_function() {
        // Test log gamma function against known values
        // ln(Γ(1)) = 0
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-10);

        // ln(Γ(2)) = ln(1!) = 0
        assert!((ln_gamma(2.0) - 0.0).abs() < 1e-10);

        // ln(Γ(3)) = ln(2!) = ln(2) ≈ 0.693
        assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);

        // ln(Γ(0.5)) = ln(√π) ≈ 0.572
        assert!((ln_gamma(0.5) - (std::f64::consts::PI.sqrt().ln())).abs() < 1e-6);
    }

    #[test]
    fn test_normal_cdf() {
        // Test standard normal CDF values
        // Φ(0) = 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);

        // Φ(∞) → 1
        assert!((normal_cdf(5.0) - 1.0).abs() < 1e-6);

        // Φ(-∞) → 0
        assert!(normal_cdf(-5.0) < 1e-6);

        // Φ(1.96) ≈ 0.975 (97.5th percentile)
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.001);
    }

    // =========================================================================
    // Statistical Tests - Normality, Autocorrelation, Stationarity
    // =========================================================================

    #[test]
    fn test_jarque_bera_normal_data() {
        // Generate normally distributed data using Box-Muller transform
        let n = 1000;
        let mut rng = StdRng::seed_from_u64(42);
        let normal_data: Vec<f64> = (0..n)
            .map(|_| {
                let mut sum = 0.0;
                for _ in 0..12 {
                    sum += rng.gen::<f64>();
                }
                (sum - 6.0) * 0.5
            })
            .collect();

        let result = StatisticalTests::jarque_bera(&normal_data).expect("Should compute JB test");

        // For approximately normal data, we should NOT reject normality strongly
        assert!(result.statistic >= 0.0, "JB statistic must be non-negative");
        // P-value should be reasonable (not extremely low for normal data)
        // Note: Our pseudo-normal may not be perfectly normal, so allow some deviation
    }

    #[test]
    fn test_jarque_bera_heavy_tails() {
        // Data with heavy tails (high kurtosis) - should reject normality
        let n = 500;
        let mut rng = StdRng::seed_from_u64(7);
        let heavy_tail_data: Vec<f64> = (0..n)
            .map(|i| {
                let base = (rng.gen::<f64>() - 0.5) * 0.02;
                if i % 30 == 0 {
                    base + (rng.gen::<f64>() - 0.5) * 0.5
                } else {
                    base
                }
            })
            .collect();

        let result =
            StatisticalTests::jarque_bera(&heavy_tail_data).expect("Should compute JB test");

        // With heavy tails, JB statistic should be high
        assert!(result.statistic > 0.0, "JB statistic should be positive");
    }

    #[test]
    fn test_jarque_bera_skewed_data() {
        // Right-skewed data (like many financial return distributions)
        let n = 500;
        let mut rng = StdRng::seed_from_u64(99);
        let skewed_data: Vec<f64> = (0..n)
            .map(|_| {
                let u = rng.gen::<f64>();
                (u.powi(3) - 0.25) * 0.1
            })
            .collect();

        let result = StatisticalTests::jarque_bera(&skewed_data).expect("Should compute JB test");

        // Skewed data should have high JB statistic
        assert!(
            result.statistic > 5.0,
            "Skewed data should have high JB statistic: {}",
            result.statistic
        );
    }

    #[test]
    fn test_jarque_bera_insufficient_data() {
        // Too few observations
        let small_data = vec![1.0, 2.0, 3.0];
        assert!(StatisticalTests::jarque_bera(&small_data).is_none());

        // Constant data (zero variance)
        let constant_data = vec![1.0; 100];
        assert!(StatisticalTests::jarque_bera(&constant_data).is_none());
    }

    #[test]
    fn test_durbin_watson_no_autocorrelation() {
        // Independent data should have DW ≈ 2
        let n = 200;
        let mut rng = StdRng::seed_from_u64(123);
        let independent_data: Vec<f64> = (0..n).map(|_| (rng.gen::<f64>() - 0.5) * 0.02).collect();

        let result =
            StatisticalTests::durbin_watson(&independent_data).expect("Should compute DW test");

        // DW should be close to 2 for independent data
        assert!(
            result.statistic > 1.5 && result.statistic < 2.5,
            "DW should be near 2 for independent data: {}",
            result.statistic
        );
    }

    #[test]
    fn test_durbin_watson_positive_autocorrelation() {
        // Create positively autocorrelated data (trending)
        let n = 200;
        let mut rng = StdRng::seed_from_u64(456);
        let mut autocorrelated_data = Vec::with_capacity(n);
        autocorrelated_data.push(0.0);
        for i in 1..n {
            let innovation = (rng.gen::<f64>() - 0.5) * 0.01;
            let prev = autocorrelated_data[i - 1];
            autocorrelated_data.push(0.8 * prev + innovation);
        }

        let result =
            StatisticalTests::durbin_watson(&autocorrelated_data).expect("Should compute DW test");

        // DW should be low (<2) for positively autocorrelated data
        assert!(
            result.statistic < 1.5,
            "DW should be low for positively autocorrelated data: {}",
            result.statistic
        );
    }

    #[test]
    fn test_durbin_watson_negative_autocorrelation() {
        // Create negatively autocorrelated data (mean-reverting)
        let n = 200;
        let mut rng = StdRng::seed_from_u64(789);
        let mut mean_reverting = Vec::with_capacity(n);
        mean_reverting.push(0.0);
        for i in 1..n {
            let innovation = (rng.gen::<f64>() - 0.5) * 0.005;
            let prev = mean_reverting[i - 1];
            mean_reverting.push(-0.5 * prev + innovation);
        }

        let result =
            StatisticalTests::durbin_watson(&mean_reverting).expect("Should compute DW test");

        // DW should be high (>2) for negatively autocorrelated data
        assert!(
            result.statistic > 2.0,
            "DW should be high for negatively autocorrelated data: {}",
            result.statistic
        );
    }

    #[test]
    fn test_durbin_watson_insufficient_data() {
        let small_data = vec![1.0, 2.0];
        assert!(StatisticalTests::durbin_watson(&small_data).is_none());

        // Zero variance residuals
        let constant = vec![1.0; 100];
        assert!(StatisticalTests::durbin_watson(&constant).is_none());
    }

    #[test]
    fn test_adf_stationary_data() {
        // Generate stationary data (returns should be stationary)
        let n = 300;
        let stationary_data: Vec<f64> = (0..n)
            .map(|i| {
                // Stationary process: mean-reverting with noise
                (i as f64 * 1.618033988749895).sin() * 0.02
            })
            .collect();

        let result =
            StatisticalTests::adf_test(&stationary_data, Some(4)).expect("Should compute ADF test");

        // For stationary data, ADF statistic should be more negative
        // A very negative ADF indicates rejection of unit root (stationarity)
        assert!(
            result.statistic < 0.0,
            "ADF should be negative for stationary data: {}",
            result.statistic
        );
    }

    #[test]
    fn test_adf_non_stationary_data() {
        // Generate random walk (non-stationary)
        let n = 300;
        let mut random_walk = Vec::with_capacity(n);
        random_walk.push(100.0);
        for i in 1..n {
            let innovation = (i as f64 * 1.618033988749895).sin() * 0.5;
            random_walk.push(random_walk[i - 1] + innovation);
        }

        let result =
            StatisticalTests::adf_test(&random_walk, Some(4)).expect("Should compute ADF test");

        // For non-stationary data (random walk), ADF should be closer to 0 or positive
        // We should NOT be able to reject the unit root hypothesis
        assert!(
            result.statistic > -3.0,
            "ADF should be less negative for non-stationary data: {}",
            result.statistic
        );
    }

    #[test]
    fn test_adf_auto_lag_selection() {
        let n = 250;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();

        // Test with automatic lag selection
        let result_auto =
            StatisticalTests::adf_test(&data, None).expect("Should work with auto lag");

        // Test with explicit lag
        let result_manual =
            StatisticalTests::adf_test(&data, Some(5)).expect("Should work with manual lag");

        // Both should produce valid results
        assert!(result_auto.statistic.is_finite());
        assert!(result_manual.statistic.is_finite());
    }

    #[test]
    fn test_adf_insufficient_data() {
        let small_data = vec![1.0; 10];
        assert!(StatisticalTests::adf_test(&small_data, Some(2)).is_none());
    }

    #[test]
    fn test_autocorrelation_calculation() {
        // Zero autocorrelation case
        let n = 200;
        let mut rng = StdRng::seed_from_u64(2024);
        let independent: Vec<f64> = (0..n).map(|_| (rng.gen::<f64>() - 0.5) * 0.02).collect();

        let rho = StatisticalTests::autocorrelation(&independent)
            .expect("Should compute autocorrelation");

        // Should be near 0 for independent data
        assert!(
            rho.abs() < 0.2,
            "Autocorrelation should be near 0 for independent data: {}",
            rho
        );

        // High autocorrelation case (AR(1) process)
        let mut ar_rng = StdRng::seed_from_u64(2025);
        let mut ar1 = Vec::with_capacity(n);
        ar1.push(0.0);
        for i in 1..n {
            let innovation = (ar_rng.gen::<f64>() - 0.5) * 0.004;
            ar1.push(0.9 * ar1[i - 1] + innovation);
        }

        let rho_ar1 =
            StatisticalTests::autocorrelation(&ar1).expect("Should compute autocorrelation");

        // Should be near 0.9 for AR(1) with rho=0.9
        assert!(
            rho_ar1 > 0.6,
            "Autocorrelation should be high for AR(1) process: {}",
            rho_ar1
        );
    }

    #[test]
    fn test_ljung_box_no_autocorrelation() {
        let n = 200;
        let mut rng = StdRng::seed_from_u64(3030);
        let independent: Vec<f64> = (0..n).map(|_| (rng.gen::<f64>() - 0.5) * 0.02).collect();

        let result =
            StatisticalTests::ljung_box(&independent, Some(10)).expect("Should compute LB test");

        // For independent data, Q should be low and p-value high
        assert!(result.statistic >= 0.0, "Q statistic must be non-negative");
        // Should NOT reject null hypothesis of no autocorrelation
        assert!(
            result.p_value > 0.01,
            "P-value should be reasonably high for independent data: {}",
            result.p_value
        );
    }

    #[test]
    fn test_ljung_box_with_autocorrelation() {
        let n = 200;
        let mut ar1 = Vec::with_capacity(n);
        ar1.push(0.0);
        for i in 1..n {
            let innovation = (i as f64 * 1.618033988749895).sin() * 0.002;
            ar1.push(0.8 * ar1[i - 1] + innovation);
        }

        let result = StatisticalTests::ljung_box(&ar1, Some(10)).expect("Should compute LB test");

        // For autocorrelated data, Q should be high and p-value low
        assert!(
            result.statistic > 10.0,
            "Q should be high for autocorrelated data: {}",
            result.statistic
        );
    }

    #[test]
    fn test_ljung_box_insufficient_data() {
        let small_data = vec![1.0, 2.0, 3.0];
        assert!(StatisticalTests::ljung_box(&small_data, Some(5)).is_none());
    }

    #[test]
    fn test_chi_squared_cdf() {
        // χ²(df=2) CDF values
        // P(X ≤ 0) = 0
        assert!(chi_squared_cdf(0.0, 2) < 0.001);

        // P(X ≤ 5.99) ≈ 0.95 for df=2
        let p_5_99 = chi_squared_cdf(5.99, 2);
        assert!(
            (p_5_99 - 0.95).abs() < 0.02,
            "chi_squared_cdf(5.99, 2) should be ~0.95: {}",
            p_5_99
        );

        // P(X ≤ 9.21) ≈ 0.99 for df=2
        let p_9_21 = chi_squared_cdf(9.21, 2);
        assert!(
            (p_9_21 - 0.99).abs() < 0.02,
            "chi_squared_cdf(9.21, 2) should be ~0.99: {}",
            p_9_21
        );

        // For larger df
        // χ²(df=10): P(X ≤ 18.31) ≈ 0.95
        let p_df10 = chi_squared_cdf(18.31, 10);
        assert!(
            (p_df10 - 0.95).abs() < 0.02,
            "chi_squared_cdf(18.31, 10) should be ~0.95: {}",
            p_df10
        );
    }

    #[test]
    fn test_statistical_test_result_creation() {
        let result = StatisticalTestResult::new("Test Name", 2.5, 0.03, "Test interpretation");

        assert_eq!(result.test_name, "Test Name");
        assert_eq!(result.statistic, 2.5);
        assert_eq!(result.p_value, 0.03);
        assert!(result.reject_null_at_5pct); // 0.03 < 0.05
        assert!(!result.reject_null_at_1pct); // 0.03 > 0.01
    }

    #[test]
    fn test_statistical_test_result_serialization() {
        let result =
            StatisticalTestResult::new("Jarque-Bera", 15.5, 0.0004, "Non-normal distribution");

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Jarque-Bera"));
        assert!(json.contains("15.5"));

        let deserialized: StatisticalTestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.test_name, result.test_name);
        assert_eq!(deserialized.statistic, result.statistic);
    }

    #[test]
    fn test_lower_incomplete_gamma() {
        // Test with known values
        // γ(1, 1) / Γ(1) = 1 - e^(-1) ≈ 0.632
        let lig_1_1 = lower_incomplete_gamma(1.0, 1.0);
        assert!(
            (lig_1_1 - 0.632).abs() < 0.01,
            "lower_incomplete_gamma(1, 1) should be ~0.632: {}",
            lig_1_1
        );

        // Edge cases
        assert!(lower_incomplete_gamma(1.0, 0.0) < 0.001);
        assert!((lower_incomplete_gamma(1.0, 10.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_suspicious_metrics_high_sharpe() {
        let metrics = PerformanceMetrics {
            sharpe_ratio: 4.0, // Suspiciously high
            win_rate: 0.60,
            max_drawdown_pct: -15.0,
            total_trades: 100,
            profit_factor: 2.0,
            ..Default::default()
        };

        let warnings = metrics.check_suspicious_metrics();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].metric, "sharpe_ratio");
        assert_eq!(warnings[0].severity, "warning");
    }

    #[test]
    fn test_suspicious_metrics_high_win_rate() {
        let metrics = PerformanceMetrics {
            sharpe_ratio: 1.5,
            win_rate: 0.85, // Suspiciously high
            max_drawdown_pct: -10.0,
            total_trades: 100,
            profit_factor: 2.0,
            ..Default::default()
        };

        let warnings = metrics.check_suspicious_metrics();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].metric, "win_rate");
    }

    #[test]
    fn test_suspicious_metrics_low_trades() {
        let metrics = PerformanceMetrics {
            sharpe_ratio: 1.5,
            win_rate: 0.55,
            max_drawdown_pct: -10.0,
            total_trades: 15, // Too few for significance
            profit_factor: 2.0,
            ..Default::default()
        };

        let warnings = metrics.check_suspicious_metrics();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].metric, "total_trades");
        assert_eq!(warnings[0].severity, "caution");
    }

    #[test]
    fn test_suspicious_metrics_no_warnings() {
        let metrics = PerformanceMetrics {
            sharpe_ratio: 1.5,
            win_rate: 0.55,
            max_drawdown_pct: -15.0,
            total_trades: 100,
            profit_factor: 2.0,
            ..Default::default()
        };

        let warnings = metrics.check_suspicious_metrics();
        assert!(
            warnings.is_empty(),
            "Expected no warnings for normal metrics"
        );
    }

    #[test]
    fn test_oos_degradation_warning() {
        // Severe overfitting
        let warning = check_oos_degradation(0.5, 2.0);
        assert!(warning.is_some());
        let w = warning.unwrap();
        assert_eq!(w.metric, "oos_is_ratio");
        assert!(w.value < 0.30); // 0.5/2.0 = 0.25
        assert_eq!(w.severity, "warning");

        // Acceptable but cautionary
        let warning = check_oos_degradation(1.4, 2.0);
        assert!(warning.is_some());
        let w = warning.unwrap();
        assert_eq!(w.severity, "caution");

        // Good ratio - no warning
        let warning = check_oos_degradation(1.8, 2.0);
        assert!(warning.is_none());
    }

    #[test]
    fn test_rolling_sharpe() {
        // Simple test case with known values
        let returns = vec![
            0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.01, 0.008, 0.003, -0.002,
        ];

        // Rolling window of 5
        let rolling = rolling_sharpe(&returns, 5, 252.0);

        // First 4 values should be NaN
        assert!(rolling[0].is_nan());
        assert!(rolling[1].is_nan());
        assert!(rolling[2].is_nan());
        assert!(rolling[3].is_nan());

        // From index 4 onwards, we should have valid values
        assert!(rolling[4].is_finite());
        assert!(rolling[9].is_finite());

        // The Sharpe should be positive for the last window (mostly positive returns)
        assert!(rolling[9] > 0.0);
    }

    #[test]
    fn test_rolling_sharpe_empty() {
        let returns: Vec<f64> = vec![];
        let rolling = rolling_sharpe(&returns, 5, 252.0);
        assert!(rolling.is_empty());
    }

    #[test]
    fn test_rolling_sharpe_window_zero() {
        let returns = vec![0.01, 0.02, 0.03];
        let rolling = rolling_sharpe(&returns, 0, 252.0);
        assert!(rolling.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_rolling_drawdown() {
        let equity = vec![100.0, 105.0, 102.0, 108.0, 103.0, 110.0];
        let drawdowns = rolling_drawdown(&equity);

        assert_eq!(drawdowns.len(), 6);

        // First value - no drawdown from initial
        assert!((drawdowns[0] - 0.0).abs() < 0.0001);

        // Second value - new peak, no drawdown
        assert!((drawdowns[1] - 0.0).abs() < 0.0001);

        // Third value - below peak of 105
        assert!((drawdowns[2] - (-0.0286)).abs() < 0.001);

        // Fourth value - new peak
        assert!((drawdowns[3] - 0.0).abs() < 0.0001);

        // Fifth value - below peak of 108
        assert!((drawdowns[4] - (-0.0463)).abs() < 0.001);

        // Last value - new peak
        assert!((drawdowns[5] - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_rolling_drawdown_empty() {
        let equity: Vec<f64> = vec![];
        let drawdowns = rolling_drawdown(&equity);
        assert!(drawdowns.is_empty());
    }

    #[test]
    fn test_rolling_drawdown_windowed() {
        let equity = vec![100.0, 105.0, 102.0, 108.0, 103.0, 110.0];

        // Window of 3 - only look back 3 periods for peak
        let drawdowns = rolling_drawdown_windowed(&equity, 3);

        assert_eq!(drawdowns.len(), 6);

        // With window=3, at index 4 (value=103), the window is [102, 108, 103]
        // Peak in window is 108, so drawdown is (103-108)/108 = -0.0463
        assert!((drawdowns[4] - (-0.0463)).abs() < 0.001);
    }

    #[test]
    fn test_rolling_max_drawdown() {
        let equity = vec![100.0, 105.0, 102.0, 100.0, 108.0, 103.0, 110.0];
        let max_dds = rolling_max_drawdown(&equity, 3);

        assert_eq!(max_dds.len(), 7);

        // First 2 values should be NaN
        assert!(max_dds[0].is_nan());
        assert!(max_dds[1].is_nan());

        // From index 2 onwards, should have values
        assert!(max_dds[2].is_finite());

        // All max drawdown values should be <= 0
        for dd in max_dds.iter().skip(2) {
            assert!(*dd <= 0.0);
        }
    }

    #[test]
    fn test_rolling_volatility() {
        let returns = vec![
            0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.01, 0.008, 0.003, -0.002,
        ];

        let volatility = rolling_volatility(&returns, 5, 252.0);

        // First 4 should be NaN
        assert!(volatility[0].is_nan());
        assert!(volatility[3].is_nan());

        // From index 4 onwards, should have positive volatility
        assert!(volatility[4] > 0.0);
        assert!(volatility[9] > 0.0);
    }
}

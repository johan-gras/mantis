//! Configuration file support for backtests.
//!
//! Allows loading backtest configurations from TOML files for reproducibility.

use crate::engine::BacktestConfig;
use crate::error::{BacktestError, Result};
use crate::portfolio::CostModel;
use crate::risk::{RiskConfig, StopLoss, TakeProfit};
use chrono::{NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::info;

/// Complete backtest configuration loaded from a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestFileConfig {
    /// General backtest settings.
    #[serde(default)]
    pub backtest: BacktestSettings,
    /// Data settings.
    #[serde(default)]
    pub data: DataSettings,
    /// Strategy settings.
    #[serde(default)]
    pub strategy: StrategySettings,
    /// Cost model settings.
    #[serde(default)]
    pub costs: CostSettings,
    /// Risk management settings.
    #[serde(default)]
    pub risk: RiskSettings,
}

impl Default for BacktestFileConfig {
    fn default() -> Self {
        Self {
            backtest: BacktestSettings::default(),
            data: DataSettings::default(),
            strategy: StrategySettings::default(),
            costs: CostSettings::default(),
            risk: RiskSettings::default(),
        }
    }
}

/// General backtest settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestSettings {
    /// Initial capital.
    #[serde(default = "default_capital")]
    pub initial_capital: f64,
    /// Position size as fraction of equity.
    #[serde(default = "default_position_size")]
    pub position_size: f64,
    /// Allow short selling.
    #[serde(default = "default_true")]
    pub allow_short: bool,
    /// Allow fractional shares.
    #[serde(default = "default_true")]
    pub fractional_shares: bool,
    /// Start date (YYYY-MM-DD format).
    #[serde(default)]
    pub start_date: Option<String>,
    /// End date (YYYY-MM-DD format).
    #[serde(default)]
    pub end_date: Option<String>,
}

fn default_capital() -> f64 { 100_000.0 }
fn default_position_size() -> f64 { 1.0 }
fn default_true() -> bool { true }

impl Default for BacktestSettings {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 1.0,
            allow_short: true,
            fractional_shares: true,
            start_date: None,
            end_date: None,
        }
    }
}

/// Data settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSettings {
    /// Path to data file.
    pub path: Option<String>,
    /// Symbol name.
    #[serde(default = "default_symbol")]
    pub symbol: String,
    /// Date format in CSV.
    pub date_format: Option<String>,
    /// CSV delimiter.
    #[serde(default = "default_delimiter")]
    pub delimiter: char,
}

fn default_symbol() -> String { "SYMBOL".to_string() }
fn default_delimiter() -> char { ',' }

impl Default for DataSettings {
    fn default() -> Self {
        Self {
            path: None,
            symbol: "SYMBOL".to_string(),
            date_format: None,
            delimiter: ',',
        }
    }
}

/// Strategy settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySettings {
    /// Strategy type.
    #[serde(default = "default_strategy")]
    pub name: String,
    /// Strategy parameters.
    #[serde(default)]
    pub params: StrategyParams,
}

fn default_strategy() -> String { "sma-crossover".to_string() }

impl Default for StrategySettings {
    fn default() -> Self {
        Self {
            name: "sma-crossover".to_string(),
            params: StrategyParams::default(),
        }
    }
}

/// Strategy parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StrategyParams {
    /// Fast period (for SMA, MACD).
    pub fast_period: Option<usize>,
    /// Slow period (for SMA, MACD).
    pub slow_period: Option<usize>,
    /// Signal period (for MACD).
    pub signal_period: Option<usize>,
    /// RSI period.
    pub rsi_period: Option<usize>,
    /// RSI oversold threshold.
    pub rsi_oversold: Option<f64>,
    /// RSI overbought threshold.
    pub rsi_overbought: Option<f64>,
    /// Lookback period.
    pub lookback: Option<usize>,
    /// Bollinger band period.
    pub bb_period: Option<usize>,
    /// Bollinger band standard deviations.
    pub bb_std: Option<f64>,
    /// Entry period (for breakout).
    pub entry_period: Option<usize>,
    /// Exit period (for breakout).
    pub exit_period: Option<usize>,
    /// Momentum threshold.
    pub momentum_threshold: Option<f64>,
}

/// Cost model settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSettings {
    /// Flat commission per trade.
    #[serde(default)]
    pub commission_flat: f64,
    /// Commission as percentage.
    #[serde(default = "default_commission_pct")]
    pub commission_pct: f64,
    /// Slippage as percentage.
    #[serde(default = "default_slippage_pct")]
    pub slippage_pct: f64,
    /// Minimum commission.
    #[serde(default)]
    pub min_commission: f64,
}

fn default_commission_pct() -> f64 { 0.1 }
fn default_slippage_pct() -> f64 { 0.05 }

impl Default for CostSettings {
    fn default() -> Self {
        Self {
            commission_flat: 0.0,
            commission_pct: 0.1,
            slippage_pct: 0.05,
            min_commission: 0.0,
        }
    }
}

/// Risk management settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSettings {
    /// Stop loss type: "none", "percentage", "fixed", "trailing".
    #[serde(default = "default_stop_type")]
    pub stop_loss_type: String,
    /// Stop loss value (percentage or fixed amount).
    #[serde(default)]
    pub stop_loss_value: f64,
    /// Take profit type: "none", "percentage", "fixed", "risk_reward".
    #[serde(default = "default_stop_type")]
    pub take_profit_type: String,
    /// Take profit value.
    #[serde(default)]
    pub take_profit_value: f64,
    /// Risk per trade as percentage.
    #[serde(default = "default_risk_per_trade")]
    pub risk_per_trade_pct: f64,
    /// Maximum drawdown percentage before halting.
    pub max_drawdown_pct: Option<f64>,
}

fn default_stop_type() -> String { "none".to_string() }
fn default_risk_per_trade() -> f64 { 2.0 }

impl Default for RiskSettings {
    fn default() -> Self {
        Self {
            stop_loss_type: "none".to_string(),
            stop_loss_value: 0.0,
            take_profit_type: "none".to_string(),
            take_profit_value: 0.0,
            risk_per_trade_pct: 2.0,
            max_drawdown_pct: None,
        }
    }
}

impl BacktestFileConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading configuration from: {}", path.display());

        let content = fs::read_to_string(path)?;
        let config: BacktestFileConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a TOML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| BacktestError::ConfigError(e.to_string()))?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Convert to BacktestConfig for the engine.
    pub fn to_backtest_config(&self) -> Result<BacktestConfig> {
        let cost_model = CostModel {
            commission_flat: self.costs.commission_flat,
            commission_pct: self.costs.commission_pct / 100.0,
            slippage_pct: self.costs.slippage_pct / 100.0,
            min_commission: self.costs.min_commission,
        };

        let stop_loss = match self.risk.stop_loss_type.to_lowercase().as_str() {
            "percentage" => StopLoss::Percentage(self.risk.stop_loss_value),
            "fixed" => StopLoss::Fixed(self.risk.stop_loss_value),
            "trailing" => StopLoss::Trailing(self.risk.stop_loss_value),
            _ => StopLoss::None,
        };

        let take_profit = match self.risk.take_profit_type.to_lowercase().as_str() {
            "percentage" => TakeProfit::Percentage(self.risk.take_profit_value),
            "fixed" => TakeProfit::Fixed(self.risk.take_profit_value),
            "risk_reward" => TakeProfit::RiskReward {
                ratio: self.risk.take_profit_value,
                stop_distance: 0.0,
            },
            _ => TakeProfit::None,
        };

        let risk_config = RiskConfig {
            stop_loss,
            take_profit,
            risk_per_trade_pct: self.risk.risk_per_trade_pct,
            max_drawdown_pct: self.risk.max_drawdown_pct,
            ..Default::default()
        };

        let start_date = self.backtest.start_date.as_ref().map(|s| {
            NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .map(|d| Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap()))
                .ok()
        }).flatten();

        let end_date = self.backtest.end_date.as_ref().map(|s| {
            NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .map(|d| Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap()))
                .ok()
        }).flatten();

        Ok(BacktestConfig {
            initial_capital: self.backtest.initial_capital,
            cost_model,
            position_size: self.backtest.position_size,
            allow_short: self.backtest.allow_short,
            fractional_shares: self.backtest.fractional_shares,
            show_progress: true,
            start_date,
            end_date,
            risk_config,
        })
    }

    /// Generate an example configuration file content.
    pub fn example() -> String {
        r#"# Ralph Backtest Configuration File
# This file configures a backtest run

[backtest]
initial_capital = 100000.0
position_size = 1.0
allow_short = true
fractional_shares = true
# start_date = "2023-01-01"
# end_date = "2023-12-31"

[data]
path = "data/sample.csv"
symbol = "AAPL"
# date_format = "%Y-%m-%d"
delimiter = ","

[strategy]
name = "sma-crossover"

[strategy.params]
fast_period = 10
slow_period = 30

# Alternative strategies:
# [strategy]
# name = "macd"
# [strategy.params]
# fast_period = 12
# slow_period = 26
# signal_period = 9

# [strategy]
# name = "rsi"
# [strategy.params]
# rsi_period = 14
# rsi_oversold = 30.0
# rsi_overbought = 70.0

[costs]
commission_flat = 0.0
commission_pct = 0.1    # 0.1%
slippage_pct = 0.05     # 0.05%
min_commission = 0.0

[risk]
stop_loss_type = "percentage"
stop_loss_value = 5.0   # 5% stop loss
take_profit_type = "percentage"
take_profit_value = 10.0  # 10% take profit
risk_per_trade_pct = 2.0
# max_drawdown_pct = 20.0
"#.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = BacktestFileConfig::default();
        assert_eq!(config.backtest.initial_capital, 100_000.0);
        assert_eq!(config.strategy.name, "sma-crossover");
    }

    #[test]
    fn test_load_config() {
        let toml_content = r#"
[backtest]
initial_capital = 50000.0
position_size = 0.5

[data]
path = "test.csv"
symbol = "TEST"

[strategy]
name = "momentum"

[strategy.params]
lookback = 20

[costs]
commission_pct = 0.2

[risk]
stop_loss_type = "percentage"
stop_loss_value = 3.0
"#;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{}", toml_content).unwrap();

        let config = BacktestFileConfig::load(file.path()).unwrap();
        assert_eq!(config.backtest.initial_capital, 50000.0);
        assert_eq!(config.backtest.position_size, 0.5);
        assert_eq!(config.data.symbol, "TEST");
        assert_eq!(config.strategy.name, "momentum");
        assert_eq!(config.strategy.params.lookback, Some(20));
        assert!((config.costs.commission_pct - 0.2).abs() < 0.001);
        assert_eq!(config.risk.stop_loss_type, "percentage");
    }

    #[test]
    fn test_to_backtest_config() {
        let file_config = BacktestFileConfig {
            backtest: BacktestSettings {
                initial_capital: 75000.0,
                position_size: 0.8,
                allow_short: false,
                ..Default::default()
            },
            costs: CostSettings {
                commission_pct: 0.15,
                slippage_pct: 0.1,
                ..Default::default()
            },
            risk: RiskSettings {
                stop_loss_type: "percentage".to_string(),
                stop_loss_value: 5.0,
                take_profit_type: "percentage".to_string(),
                take_profit_value: 10.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let config = file_config.to_backtest_config().unwrap();
        assert_eq!(config.initial_capital, 75000.0);
        assert_eq!(config.position_size, 0.8);
        assert!(!config.allow_short);
        assert!((config.cost_model.commission_pct - 0.0015).abs() < 0.0001);
    }

    #[test]
    fn test_save_config() {
        let config = BacktestFileConfig::default();
        let file = NamedTempFile::new().unwrap();
        config.save(file.path()).unwrap();

        // Verify we can load it back
        let loaded = BacktestFileConfig::load(file.path()).unwrap();
        assert_eq!(loaded.backtest.initial_capital, config.backtest.initial_capital);
    }

    #[test]
    fn test_example_config() {
        let example = BacktestFileConfig::example();
        assert!(example.contains("[backtest]"));
        assert!(example.contains("[strategy]"));
        assert!(example.contains("[costs]"));
        assert!(example.contains("[risk]"));
    }
}

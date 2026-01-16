//! Experiment tracking and storage for reproducible research.
//!
//! **DEPRECATION NOTICE**: This module is deprecated and will be removed in a future version.
//! Use external experiment tracking tools that integrate with your ML workflow:
//! - MLflow for experiment tracking and model registry
//! - Weights & Biases (W&B) for experiment management
//! - DVC for data and experiment versioning
//! - Sacred for experiment configuration and logging
//!
//! These tools provide better integration with training pipelines, visualization dashboards,
//! and collaboration features that are outside the scope of a backtester.
//!
//! This module provides (deprecated) persistent storage and querying of backtest experiments using SQLite.
//! Each experiment is automatically logged with metadata, performance metrics, and configuration.

use crate::engine::BacktestResult;
use crate::error::{BacktestError, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info};

/// Experiment metadata and summary metrics for storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRecord {
    /// Unique experiment identifier.
    pub experiment_id: String,
    /// Timestamp when the experiment was run.
    pub timestamp: DateTime<Utc>,
    /// Duration of the backtest in milliseconds.
    pub duration_ms: Option<i64>,
    /// Strategy name.
    pub strategy_name: String,
    /// Configuration hash for deduplication.
    pub config_hash: String,
    /// Git commit SHA.
    pub git_commit: Option<String>,
    /// Git branch name.
    pub git_branch: Option<String>,
    /// Whether working tree had uncommitted changes.
    pub git_dirty: bool,
    /// Symbol(s) traded (comma-separated).
    pub symbols: String,
    /// Total return percentage.
    pub total_return: f64,
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Sortino ratio.
    pub sortino_ratio: f64,
    /// Calmar ratio.
    pub calmar_ratio: f64,
    /// Maximum drawdown percentage.
    pub max_drawdown: f64,
    /// Number of trades.
    pub num_trades: i64,
    /// Win rate percentage.
    pub win_rate: f64,
    /// Profit factor.
    pub profit_factor: f64,
    /// Full configuration as JSON.
    pub config_json: String,
    /// Data file checksums as JSON.
    pub data_files_json: String,
    /// User-defined tags (comma-separated).
    pub tags: Option<String>,
    /// User notes.
    pub notes: Option<String>,
}

impl ExperimentRecord {
    /// Create an experiment record from a backtest result.
    pub fn from_backtest_result(result: &BacktestResult, duration_ms: Option<i64>) -> Result<Self> {
        let config_json = serde_json::to_string(&result.config)?;
        let data_files_json = serde_json::to_string(&result.data_checksums)?;

        let git_commit = result.git_info.as_ref().map(|gi| gi.commit_sha.clone());
        let git_branch = result.git_info.as_ref().and_then(|gi| gi.branch.clone());
        let git_dirty = result.git_info.as_ref().map(|gi| gi.dirty).unwrap_or(false);

        Ok(ExperimentRecord {
            experiment_id: result.experiment_id.to_string(),
            timestamp: result.end_time,
            duration_ms,
            strategy_name: result.strategy_name.clone(),
            config_hash: result.config_hash.clone(),
            git_commit,
            git_branch,
            git_dirty,
            symbols: result.symbols.join(","),
            total_return: result.total_return_pct,
            sharpe_ratio: result.sharpe_ratio,
            sortino_ratio: result.sortino_ratio,
            calmar_ratio: result.calmar_ratio,
            max_drawdown: result.max_drawdown_pct,
            num_trades: result.total_trades as i64,
            win_rate: result.win_rate,
            profit_factor: result.profit_factor,
            config_json,
            data_files_json,
            tags: None,
            notes: None,
        })
    }
}

/// Query filter for searching experiments.
#[derive(Debug, Clone, Default)]
pub struct ExperimentFilter {
    /// Filter by strategy name (partial match).
    pub strategy_name: Option<String>,
    /// Filter by minimum Sharpe ratio.
    pub min_sharpe: Option<f64>,
    /// Filter by maximum drawdown.
    pub max_drawdown: Option<f64>,
    /// Filter by git commit SHA.
    pub git_commit: Option<String>,
    /// Filter by tags (all tags must be present).
    pub tags: Vec<String>,
    /// Limit number of results.
    pub limit: Option<usize>,
    /// Sort by field (sharpe_ratio, total_return, timestamp, etc.).
    pub sort_by: Option<String>,
    /// Sort descending.
    pub sort_desc: bool,
}

/// Persistent storage for experiment tracking.
pub struct ExperimentStore {
    db_path: String,
}

impl ExperimentStore {
    /// Create or open an experiment store at the given path.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let db_path = path.as_ref().display().to_string();
        let conn = Connection::open(&db_path)
            .map_err(|e| BacktestError::DatabaseError(format!("Failed to open database: {}", e)))?;

        // Create table if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                duration_ms INTEGER,
                strategy_name TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                git_commit TEXT,
                git_branch TEXT,
                git_dirty INTEGER NOT NULL,
                symbols TEXT NOT NULL,
                total_return REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                sortino_ratio REAL NOT NULL,
                calmar_ratio REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                num_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                profit_factor REAL NOT NULL,
                config_json TEXT NOT NULL,
                data_files_json TEXT NOT NULL,
                tags TEXT,
                notes TEXT
            )",
            [],
        )
        .map_err(|e| BacktestError::DatabaseError(format!("Failed to create table: {}", e)))?;

        // Create indices for fast queries
        let indices = vec![
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON experiments(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_strategy ON experiments(strategy_name)",
            "CREATE INDEX IF NOT EXISTS idx_sharpe ON experiments(sharpe_ratio DESC)",
            "CREATE INDEX IF NOT EXISTS idx_git_commit ON experiments(git_commit)",
        ];

        for sql in indices {
            conn.execute(sql, []).map_err(|e| {
                BacktestError::DatabaseError(format!("Failed to create index: {}", e))
            })?;
        }

        info!("Opened experiment store at {}", db_path);

        Ok(ExperimentStore { db_path })
    }

    /// Save an experiment to the store.
    pub fn save(&self, record: &ExperimentRecord) -> Result<()> {
        let conn = self.connect()?;

        conn.execute(
            "INSERT OR REPLACE INTO experiments (
                experiment_id, timestamp, duration_ms, strategy_name, config_hash,
                git_commit, git_branch, git_dirty, symbols,
                total_return, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
                num_trades, win_rate, profit_factor,
                config_json, data_files_json, tags, notes
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
            params![
                record.experiment_id,
                record.timestamp.to_rfc3339(),
                record.duration_ms,
                record.strategy_name,
                record.config_hash,
                record.git_commit,
                record.git_branch,
                record.git_dirty as i32,
                record.symbols,
                record.total_return,
                record.sharpe_ratio,
                record.sortino_ratio,
                record.calmar_ratio,
                record.max_drawdown,
                record.num_trades,
                record.win_rate,
                record.profit_factor,
                record.config_json,
                record.data_files_json,
                record.tags,
                record.notes,
            ],
        )
        .map_err(|e| BacktestError::DatabaseError(format!("Failed to save experiment: {}", e)))?;

        debug!("Saved experiment {}", record.experiment_id);
        Ok(())
    }

    /// Get an experiment by ID (supports partial ID matching).
    pub fn get(&self, experiment_id: &str) -> Result<Option<ExperimentRecord>> {
        let conn = self.connect()?;

        // First try exact match
        let result = conn
            .query_row(
                "SELECT experiment_id, timestamp, duration_ms, strategy_name, config_hash,
                    git_commit, git_branch, git_dirty, symbols,
                    total_return, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
                    num_trades, win_rate, profit_factor,
                    config_json, data_files_json, tags, notes
             FROM experiments WHERE experiment_id = ?1",
                params![experiment_id],
                Self::row_to_record,
            )
            .optional()
            .map_err(|e| {
                BacktestError::DatabaseError(format!("Failed to query experiment: {}", e))
            })?;

        if result.is_some() {
            return Ok(result);
        }

        // Try partial match if no exact match found
        let result = conn
            .query_row(
                "SELECT experiment_id, timestamp, duration_ms, strategy_name, config_hash,
                    git_commit, git_branch, git_dirty, symbols,
                    total_return, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
                    num_trades, win_rate, profit_factor,
                    config_json, data_files_json, tags, notes
             FROM experiments WHERE experiment_id LIKE ?1 || '%'",
                params![experiment_id],
                Self::row_to_record,
            )
            .optional()
            .map_err(|e| {
                BacktestError::DatabaseError(format!("Failed to query experiment: {}", e))
            })?;

        Ok(result)
    }

    /// List experiments matching the filter.
    pub fn list(&self, filter: &ExperimentFilter) -> Result<Vec<ExperimentRecord>> {
        let conn = self.connect()?;

        let mut sql = "SELECT experiment_id, timestamp, duration_ms, strategy_name, config_hash,
                              git_commit, git_branch, git_dirty, symbols,
                              total_return, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
                              num_trades, win_rate, profit_factor,
                              config_json, data_files_json, tags, notes
                       FROM experiments WHERE 1=1"
            .to_string();

        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        // Apply filters
        if let Some(ref strategy) = filter.strategy_name {
            sql.push_str(" AND strategy_name LIKE ?");
            params.push(Box::new(format!("%{}%", strategy)));
        }

        if let Some(min_sharpe) = filter.min_sharpe {
            sql.push_str(" AND sharpe_ratio >= ?");
            params.push(Box::new(min_sharpe));
        }

        if let Some(max_dd) = filter.max_drawdown {
            sql.push_str(" AND max_drawdown <= ?");
            params.push(Box::new(max_dd));
        }

        if let Some(ref commit) = filter.git_commit {
            sql.push_str(" AND git_commit = ?");
            params.push(Box::new(commit.clone()));
        }

        // Add sorting
        if let Some(ref sort_field) = filter.sort_by {
            sql.push_str(&format!(
                " ORDER BY {} {}",
                sort_field,
                if filter.sort_desc { "DESC" } else { "ASC" }
            ));
        } else {
            sql.push_str(" ORDER BY timestamp DESC");
        }

        // Add limit
        if let Some(limit) = filter.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| BacktestError::DatabaseError(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(params_refs.as_slice(), Self::row_to_record)
            .map_err(|e| BacktestError::DatabaseError(format!("Failed to execute query: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(
                row.map_err(|e| {
                    BacktestError::DatabaseError(format!("Failed to read row: {}", e))
                })?,
            );
        }

        Ok(results)
    }

    /// Count total experiments in the store.
    pub fn count(&self) -> Result<usize> {
        let conn = self.connect()?;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM experiments", [], |row| row.get(0))
            .map_err(|e| {
                BacktestError::DatabaseError(format!("Failed to count experiments: {}", e))
            })?;
        Ok(count as usize)
    }

    /// Delete an experiment by ID.
    pub fn delete(&self, experiment_id: &str) -> Result<()> {
        let conn = self.connect()?;
        conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?1",
            params![experiment_id],
        )
        .map_err(|e| BacktestError::DatabaseError(format!("Failed to delete experiment: {}", e)))?;
        info!("Deleted experiment {}", experiment_id);
        Ok(())
    }

    /// Add tags to an experiment.
    pub fn add_tags(&self, experiment_id: &str, tags: &[String]) -> Result<()> {
        let conn = self.connect()?;

        // Get existing tags
        let existing: Option<String> = conn
            .query_row(
                "SELECT tags FROM experiments WHERE experiment_id = ?1",
                params![experiment_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| BacktestError::DatabaseError(format!("Failed to query tags: {}", e)))?
            .flatten();

        let mut all_tags: Vec<String> = existing
            .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
            .unwrap_or_default();

        for tag in tags {
            if !all_tags.contains(tag) {
                all_tags.push(tag.clone());
            }
        }

        let tags_str = all_tags.join(",");
        conn.execute(
            "UPDATE experiments SET tags = ?1 WHERE experiment_id = ?2",
            params![tags_str, experiment_id],
        )
        .map_err(|e| BacktestError::DatabaseError(format!("Failed to update tags: {}", e)))?;

        Ok(())
    }

    /// Add notes to an experiment.
    pub fn add_notes(&self, experiment_id: &str, notes: &str) -> Result<()> {
        let conn = self.connect()?;
        conn.execute(
            "UPDATE experiments SET notes = ?1 WHERE experiment_id = ?2",
            params![notes, experiment_id],
        )
        .map_err(|e| BacktestError::DatabaseError(format!("Failed to update notes: {}", e)))?;
        Ok(())
    }

    // Helper methods

    fn connect(&self) -> Result<Connection> {
        Connection::open(&self.db_path).map_err(|e| {
            BacktestError::DatabaseError(format!("Failed to connect to database: {}", e))
        })
    }

    fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<ExperimentRecord> {
        let git_dirty: i32 = row.get(7)?;
        let timestamp_str: String = row.get(1)?;
        let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Ok(ExperimentRecord {
            experiment_id: row.get(0)?,
            timestamp,
            duration_ms: row.get(2)?,
            strategy_name: row.get(3)?,
            config_hash: row.get(4)?,
            git_commit: row.get(5)?,
            git_branch: row.get(6)?,
            git_dirty: git_dirty != 0,
            symbols: row.get(8)?,
            total_return: row.get(9)?,
            sharpe_ratio: row.get(10)?,
            sortino_ratio: row.get(11)?,
            calmar_ratio: row.get(12)?,
            max_drawdown: row.get(13)?,
            num_trades: row.get(14)?,
            win_rate: row.get(15)?,
            profit_factor: row.get(16)?,
            config_json: row.get(17)?,
            data_files_json: row.get(18)?,
            tags: row.get(19)?,
            notes: row.get(20)?,
        })
    }
}

/// Get the default experiment store path.
pub fn default_store_path() -> String {
    std::env::var("MANTIS_EXPERIMENTS_DB").unwrap_or_else(|_| {
        if let Some(home) = std::env::var_os("HOME") {
            format!("{}/.mantis/experiments.db", home.to_string_lossy())
        } else {
            "mantis_experiments.db".to_string()
        }
    })
}

/// Ensure the default experiment store directory exists.
pub fn ensure_store_directory() -> Result<()> {
    let store_path = default_store_path();
    if let Some(parent) = Path::new(&store_path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            BacktestError::DatabaseError(format!("Failed to create experiment directory: {}", e))
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{BacktestConfig, BacktestResult};
    use tempfile::NamedTempFile;
    use uuid::Uuid;

    fn create_test_result() -> BacktestResult {
        BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["AAPL".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100_000.0,
            final_equity: 125_000.0,
            total_return_pct: 25.0,
            annual_return_pct: 12.5,
            trading_days: 252,
            total_trades: 50,
            winning_trades: 30,
            losing_trades: 20,
            win_rate: 60.0,
            avg_win: 1000.0,
            avg_loss: -500.0,
            profit_factor: 2.0,
            max_drawdown_pct: -15.0,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            calmar_ratio: 0.83,
            trades: vec![],
            equity_curve: vec![],
            start_time: Utc::now(),
            end_time: Utc::now(),
            experiment_id: Uuid::new_v4(),
            git_info: None,
            config_hash: "test_hash".to_string(),
            data_checksums: std::collections::HashMap::new(),
            seed: None,
        }
    }

    #[test]
    fn test_create_store() {
        let temp_file = NamedTempFile::new().unwrap();
        let store = ExperimentStore::new(temp_file.path()).unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_save_and_get_experiment() {
        let temp_file = NamedTempFile::new().unwrap();
        let store = ExperimentStore::new(temp_file.path()).unwrap();

        let result = create_test_result();
        let record = ExperimentRecord::from_backtest_result(&result, Some(1000)).unwrap();

        store.save(&record).unwrap();
        assert_eq!(store.count().unwrap(), 1);

        let retrieved = store.get(&record.experiment_id).unwrap().unwrap();
        assert_eq!(retrieved.experiment_id, record.experiment_id);
        assert_eq!(retrieved.strategy_name, "TestStrategy");
        assert_eq!(retrieved.sharpe_ratio, 1.5);
    }

    #[test]
    fn test_list_experiments() {
        let temp_file = NamedTempFile::new().unwrap();
        let store = ExperimentStore::new(temp_file.path()).unwrap();

        // Save multiple experiments
        for i in 0..5 {
            let mut result = create_test_result();
            result.experiment_id = Uuid::new_v4();
            result.sharpe_ratio = i as f64;
            let record = ExperimentRecord::from_backtest_result(&result, None).unwrap();
            store.save(&record).unwrap();
        }

        assert_eq!(store.count().unwrap(), 5);

        // List all
        let all = store.list(&ExperimentFilter::default()).unwrap();
        assert_eq!(all.len(), 5);

        // Filter by min Sharpe
        let filtered = store
            .list(&ExperimentFilter {
                min_sharpe: Some(2.0),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(filtered.len(), 3); // Sharpe 2.0, 3.0, 4.0
    }

    #[test]
    fn test_add_tags() {
        let temp_file = NamedTempFile::new().unwrap();
        let store = ExperimentStore::new(temp_file.path()).unwrap();

        let result = create_test_result();
        let record = ExperimentRecord::from_backtest_result(&result, None).unwrap();
        let exp_id = record.experiment_id.clone();

        store.save(&record).unwrap();
        store
            .add_tags(&exp_id, &["baseline".to_string(), "production".to_string()])
            .unwrap();

        let retrieved = store.get(&exp_id).unwrap().unwrap();
        let tags = retrieved.tags.unwrap();
        assert!(tags.contains("baseline"));
        assert!(tags.contains("production"));
    }

    #[test]
    fn test_add_notes() {
        let temp_file = NamedTempFile::new().unwrap();
        let store = ExperimentStore::new(temp_file.path()).unwrap();

        let result = create_test_result();
        let record = ExperimentRecord::from_backtest_result(&result, None).unwrap();
        let exp_id = record.experiment_id.clone();

        store.save(&record).unwrap();
        store
            .add_notes(&exp_id, "This is a test experiment")
            .unwrap();

        let retrieved = store.get(&exp_id).unwrap().unwrap();
        assert_eq!(retrieved.notes.unwrap(), "This is a test experiment");
    }

    #[test]
    fn test_delete_experiment() {
        let temp_file = NamedTempFile::new().unwrap();
        let store = ExperimentStore::new(temp_file.path()).unwrap();

        let result = create_test_result();
        let record = ExperimentRecord::from_backtest_result(&result, None).unwrap();
        let exp_id = record.experiment_id.clone();

        store.save(&record).unwrap();
        assert_eq!(store.count().unwrap(), 1);

        store.delete(&exp_id).unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }
}

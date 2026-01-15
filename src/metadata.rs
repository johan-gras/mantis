//! Experiment metadata and reproducibility tracking.

use git2::Repository;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use tracing::{debug, warn};
use uuid::Uuid;

/// Git repository state at the time of execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitInfo {
    /// Current commit SHA (full 40-character hash).
    pub commit_sha: String,
    /// Current branch name.
    pub branch: Option<String>,
    /// Whether there are uncommitted changes in the working tree.
    pub dirty: bool,
}

/// Metadata about a data file used in a backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFileMetadata {
    /// File path.
    pub path: String,
    /// File size in bytes.
    pub size: u64,
    /// SHA256 checksum of the file contents.
    pub checksum: String,
}

impl GitInfo {
    /// Capture the current git state.
    ///
    /// Returns None if not in a git repository or git operations fail.
    pub fn capture() -> Option<Self> {
        let repo = match Repository::discover(".") {
            Ok(r) => r,
            Err(e) => {
                debug!("Not in a git repository: {}", e);
                return None;
            }
        };

        // Get HEAD commit
        let head = repo.head().ok()?;
        let commit = head.peel_to_commit().ok()?;
        let commit_sha = commit.id().to_string();

        // Get branch name
        let branch = if head.is_branch() {
            head.shorthand().map(|s| s.to_string())
        } else {
            None
        };

        // Check for uncommitted changes
        let mut status_opts = git2::StatusOptions::new();
        status_opts.include_untracked(true);
        let statuses = repo.statuses(Some(&mut status_opts)).ok()?;
        let dirty = !statuses.is_empty();

        if dirty {
            warn!("Running backtest with uncommitted changes in working tree");
        }

        Some(GitInfo {
            commit_sha,
            branch,
            dirty,
        })
    }
}

/// Compute SHA256 checksum of a file.
pub fn compute_file_checksum(path: impl AsRef<Path>) -> std::io::Result<String> {
    let path = path.as_ref();
    let data = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// Compute SHA256 hash of arbitrary bytes.
pub fn compute_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("{:x}", result)
}

/// Compute configuration hash from JSON serialization.
pub fn compute_config_hash<T: Serialize>(config: &T) -> String {
    match serde_json::to_vec(config) {
        Ok(bytes) => compute_hash(&bytes),
        Err(e) => {
            warn!("Failed to serialize config for hashing: {}", e);
            String::new()
        }
    }
}

/// Generate a unique experiment ID.
pub fn generate_experiment_id() -> Uuid {
    Uuid::new_v4()
}

/// Track data file metadata.
pub fn track_data_file(path: impl AsRef<Path>) -> std::io::Result<DataFileMetadata> {
    let path = path.as_ref();
    let metadata = std::fs::metadata(path)?;
    let size = metadata.len();
    let checksum = compute_file_checksum(path)?;

    Ok(DataFileMetadata {
        path: path.display().to_string(),
        size,
        checksum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compute_file_checksum() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "test data").unwrap();
        temp_file.flush().unwrap();

        let checksum = compute_file_checksum(temp_file.path()).unwrap();
        // Verify it's a valid SHA256 hex string (64 characters)
        assert_eq!(checksum.len(), 64);
        assert!(checksum.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_compute_hash() {
        let hash1 = compute_hash(b"hello");
        let hash2 = compute_hash(b"hello");
        let hash3 = compute_hash(b"world");

        // Same data produces same hash
        assert_eq!(hash1, hash2);
        // Different data produces different hash
        assert_ne!(hash1, hash3);
        // Valid SHA256 format
        assert_eq!(hash1.len(), 64);
    }

    #[test]
    fn test_generate_experiment_id() {
        let id1 = generate_experiment_id();
        let id2 = generate_experiment_id();

        // Each call generates a unique ID
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_track_data_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "sample data").unwrap();
        temp_file.flush().unwrap();

        let metadata = track_data_file(temp_file.path()).unwrap();

        assert!(metadata
            .path
            .contains(temp_file.path().file_name().unwrap().to_str().unwrap()));
        assert!(metadata.size > 0);
        assert_eq!(metadata.checksum.len(), 64);
    }

    #[test]
    fn test_compute_config_hash() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct TestConfig {
            param1: i32,
            param2: String,
        }

        let config1 = TestConfig {
            param1: 42,
            param2: "test".to_string(),
        };
        let config2 = TestConfig {
            param1: 42,
            param2: "test".to_string(),
        };
        let config3 = TestConfig {
            param1: 99,
            param2: "different".to_string(),
        };

        let hash1 = compute_config_hash(&config1);
        let hash2 = compute_config_hash(&config2);
        let hash3 = compute_config_hash(&config3);

        // Same config produces same hash
        assert_eq!(hash1, hash2);
        // Different config produces different hash
        assert_ne!(hash1, hash3);
    }
}

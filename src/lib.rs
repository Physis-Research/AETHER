pub mod data;
pub mod engine;
pub mod gp;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum AetherError {
    #[error("io failure: {0}")]
    Io(#[from] std::io::Error),
    #[error("integrity violation: {0}")]
    Integrity(String),
    #[error("serialization failure: {0}")]
    Serialization(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, AetherError>;
pub const SLIPPAGE_FLOOR: f64 = 0.001;
pub const FUNDING_RATE: f64 = 0.0001;
pub const REGIME_THRESHOLD: f64 = 0.015;
pub const IDX_VELOCITY: usize = 0;
pub const IDX_REL_MOM: usize = 1;
pub const IDX_VOL_DELTA: usize = 2;
pub const IDX_CORREL: usize = 3;
pub const IDX_EFFICIENCY: usize = 8;
pub const IDX_MARKET_REGIME: usize = 12;
pub const IDX_VOL_VOL: usize = 15;
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Performance {
    pub roi: f64,
    pub mdd: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub trades: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub genome: Vec<gp::Node>,
    pub fitness: f64,
    pub performance: Performance,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    pub long: Agent,
    pub short: Agent,
    pub performance: Performance,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAwareIndividual {
    pub bull: Individual,
    pub bear: Individual,
    pub performance: Performance,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Universe {
    pub data_matrices: Vec<Array2<f64>>,
    pub targets: Vec<Vec<f64>>,
    pub tickers: Vec<String>,
    pub dates: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub fold: usize,
    pub test_roi: f64,
    pub market_roi: f64,
    pub test_sharpe: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub folds: Vec<ValidationResult>,
    pub mean_test_roi: f64,
    pub baseline_buy_hold: f64,
    pub beats_baseline: bool,
}
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    pub generations: usize,
    pub pop_size: usize,
    pub early_stopping: usize,
    pub seed: u64,
    pub sharpe_annualization: f64,
}
impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            generations: 250,
            pop_size: 2000,
            early_stopping: 60,
            seed: 42,
            sharpe_annualization: (365.0 * 6.0f64).sqrt(),
        }
    }
}

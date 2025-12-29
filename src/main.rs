use aether::{data, engine};
use aether::{EvolutionConfig, Performance, RegimeAwareIndividual, Result};
use clap::{Parser, Subcommand};
use std::process;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "aether",
    about = "Adaptive Evolutionary Trading & Harvesting Engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Train {
        csv: String,
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    Validate {
        csv: String,
        #[arg(long, default_value = "3")]
        folds: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    Fetch {
        #[arg(long, default_value = "4h")]
        interval: String,
        #[arg(long, default_value = "365")]
        days: u64,
    },

    Audit {
        csv: String,
    },

    Clean,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {}", e);
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let start = Instant::now();
    match cli.command {
        Command::Train { csv, seed } => {
            println!("\n=== AETHER: TRAINING REGIME-AWARE ENGINE ===");
            let uni = data::prepare_universe(&csv)?;
            let config = EvolutionConfig {
                seed,
                ..Default::default()
            };
            let (b_m, b_t) = data::filter_by_regime(&uni, true);
            let bull = engine::evolve(&b_m, &b_t, config.clone(), true);
            let (r_m, r_t) = data::filter_by_regime(&uni, false);
            let bear = engine::evolve(&r_m, &r_t, config.clone(), false);
            let mut ind = RegimeAwareIndividual {
                bull,
                bear,
                performance: Performance::default(),
            };
            engine::evaluate_regime_aware(&mut ind, &uni, config.sharpe_annualization);
            println!("\n--- Results ---");
            println!("ROI:    {:.2}%", ind.performance.roi * 100.0);
            println!("Sharpe: {:.2}", ind.performance.sharpe);
            println!("MDD:    {:.2}%", ind.performance.mdd * 100.0);
            std::fs::create_dir_all("models")?;
            serde_json::to_writer_pretty(std::fs::File::create("models/aether_final.json")?, &ind)?;
        }
        Command::Validate { csv, folds, seed } => {
            println!("\n=== AETHER: VALIDATION ===");
            let summary = engine::walk_forward_validation(&csv, folds, 0.7, seed)?;
            println!("\n--- Summary ---");
            println!("Mean Test ROI: {:.2}%", summary.mean_test_roi * 100.0);
            println!("Baseline B&H:  {:.2}%", summary.baseline_buy_hold * 100.0);
            std::fs::create_dir_all("reports")?;
            serde_json::to_writer_pretty(
                std::fs::File::create("reports/validation.json")?,
                &summary,
            )?;
        }
        Command::Fetch { interval, days } => {
            data::fetch_and_process(&interval, days)?;
        }
        Command::Audit { csv } => {
            let universe = data::prepare_universe(&csv)?;
            engine::run_noise_audit(&universe);
        }
        Command::Clean => {
            let _ = std::fs::remove_dir_all("models");
            let _ = std::fs::remove_dir_all("reports");
            println!("Workspace cleaned.");
        }
    }
    println!("\nCompleted in {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}

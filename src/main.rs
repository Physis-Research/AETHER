use clap::{Parser, Subcommand};
use physis::Result;
use physis::{data, engine};
use std::process;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "physis",
    about = "Physics-Informed Strategy & Intelligent Simulation"
)]
struct Cli {
    #[arg(long, default_value_t = false)]
    verbose: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(alias = "Validate")]
    Validate {
        csv: String,
        #[arg(long, default_value = "3")]
        folds: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    #[command(alias = "Fetch")]
    Fetch {
        #[arg(long, default_value = "binance")]
        source: String,
        #[arg(long, default_value = "4h")]
        interval: String,
        #[arg(long, default_value = "365")]
        days: u64,
        #[arg(long)]
        tickers: Option<String>,
        #[arg(long, default_value_t = false)]
        append: bool,
        #[arg(long, default_value_t = false)]
        audit: bool,
    },

    #[command(alias = "Clean")]
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
    physis::set_verbose(cli.verbose);
    let start = Instant::now();
    match cli.command {
        Command::Validate { csv, folds, seed } => {
            let summary = engine::walk_forward_validation(&csv, folds, 0.7, seed)?;
            println!("mean_test_roi: {:.2}%", summary.mean_test_roi * 100.0);
            println!("baseline_bh: {:.2}%", summary.baseline_buy_hold * 100.0);
            if summary.beats_baseline {
                println!("status: beat_baseline");
            } else {
                println!("status: underperform_baseline");
            }
            std::fs::create_dir_all("reports")?;
            serde_json::to_writer_pretty(
                std::fs::File::create("reports/validation.json")?,
                &summary,
            )?;
        }
        Command::Fetch {
            source,
            interval,
            days,
            tickers,
            append,
            audit,
        } => {
            data::fetch_and_process(&source, &interval, days, tickers, append)?;
            if audit {
                let universe = data::prepare_universe("data/market_data.csv")?;
                engine::run_noise_audit(&universe);
            }
        }
        Command::Clean => {
            let _ = std::fs::remove_dir_all("reports");
            let _ = std::fs::remove_dir_all("data");
        }
    }
    println!("elapsed: {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}

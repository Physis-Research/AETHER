use crate::gp::TOTAL_FEATURES;
use crate::{
    AetherError, Result, Universe, IDX_EFFICIENCY, IDX_MARKET_REGIME, IDX_REL_MOM, IDX_VELOCITY,
    IDX_VOL_DELTA, IDX_VOL_VOL, REGIME_THRESHOLD,
};
use ndarray::{Array2, ShapeBuilder};
use std::collections::{BTreeMap, HashSet};
use std::fs;
pub fn prepare_universe(csv_path: &str) -> Result<Universe> {
    let content = fs::read_to_string(csv_path).map_err(AetherError::Io)?;
    let mut reader = csv::Reader::from_reader(content.as_bytes());
    let mut raw: BTreeMap<String, BTreeMap<String, (f64, f64)>> = BTreeMap::new();
    let mut all_dates = HashSet::<String>::new();

    for result in reader.records() {
        let record = result.map_err(|e| {
            AetherError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;
        if record.len() < 4 {
            continue;
        }
        let date = record[0].to_string();
        let close: f64 = record[1].parse().unwrap_or(0.0);
        let vol: f64 = record[2].parse().unwrap_or(0.0);
        let ticker = record[3].to_string();
        all_dates.insert(date.clone());
        raw.entry(ticker).or_default().insert(date, (close, vol));
    }

    let mut timeline: Vec<_> = all_dates.into_iter().collect();
    timeline.sort();
    let n_time = timeline.len();
    let keys: Vec<_> = raw.keys().cloned().collect();

    let mut all_rets = Vec::new();
    for ticker in &keys {
        let mut rets = Vec::with_capacity(n_time);
        let mut last = 0.0;
        for d in &timeline {
            let close = raw[ticker].get(d).map(|v| v.0).unwrap_or(0.0);
            rets.push(if last > 1e-9 && close > 1e-9 {
                (close / last).ln()
            } else {
                0.0
            });
            last = close;
        }
        all_rets.push(rets);
    }

    let mut market_regime = vec![0.0; n_time];
    for i in 20..n_time {
        let mut total_vol = 0.0;
        for series in &all_rets {
            let win = &series[i - 20..i];
            let mean = win.iter().sum::<f64>() / 20.0;
            total_vol += (win.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 20.0).sqrt();
        }
        market_regime[i] = total_vol / all_rets.len() as f64;
    }

    let lead_v = {
        let mut v = vec![0.0; n_time];
        for i in 1..n_time {
            let mut sum_ret = 0.0;
            let mut count = 0.0;
            for rets in &all_rets {
                sum_ret += rets[i];
                count += 1.0;
            }
            v[i] = if count > 0.0 { sum_ret / count } else { 0.0 };
        }
        v
    };

    let mut data_mats = Vec::new();
    let mut targets = Vec::new();
    let mut tickers = Vec::new();

    for ticker in &keys {
        let mut closes = vec![0.0; n_time];
        let mut volumes = vec![0.0; n_time];
        for (i, d) in timeline.iter().enumerate() {
            let val = raw[ticker].get(d).cloned().unwrap_or((0.0, 0.0));
            closes[i] = val.0;
            volumes[i] = val.1;
        }
        if closes.iter().any(|&c| c <= 1e-9) {
            continue;
        }

        let mut t = vec![0.0; n_time];
        for i in 0..n_time - 1 {
            t[i] = (closes[i + 1] / closes[i]).ln();
        }

        let mut matrix = Array2::<f64>::zeros((n_time, TOTAL_FEATURES).f());
        compute_physics_into(&closes, &volumes, &lead_v, &market_regime, &mut matrix);
        data_mats.push(matrix);
        targets.push(t);
        tickers.push(ticker.clone());
    }
    Ok(Universe {
        data_matrices: data_mats,
        targets,
        tickers,
        dates: timeline,
    })
}

pub fn compute_physics_into(
    closes: &[f64],
    volumes: &[f64],
    lead_v: &[f64],
    regime: &[f64],
    out: &mut Array2<f64>,
) {
    let n = closes.len();
    let mut rets = vec![0.0; n];
    for i in 1..n {
        rets[i] = (closes[i] / closes[i - 1]).ln();
    }

    for i in 20..n {
        out[[i, IDX_REL_MOM]] = rets[i] - lead_v[i];
        out[[i, 9]] = lead_v[i - 1];

        let v_avg = volumes[i - 5..i].iter().sum::<f64>() / 5.0;
        out[[i, IDX_VOL_DELTA]] = if v_avg > 1e-9 {
            (volumes[i] - v_avg) / v_avg
        } else {
            0.0
        };

        let change = (closes[i] - closes[i - 20]).abs();
        let volatility: f64 = (i - 19..=i)
            .map(|idx| (closes[idx] - closes[idx - 1]).abs())
            .sum();
        out[[i, IDX_EFFICIENCY]] = if volatility > 1e-9 {
            change / volatility
        } else {
            0.0
        };

        if i > 40 {
            let mut vols = Vec::with_capacity(20);
            for j in 0..20 {
                let sub_rets = &rets[i - 40 + j..i - 20 + j];
                let mean = sub_rets.iter().sum::<f64>() / 20.0;
                vols.push(
                    (sub_rets.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 20.0).sqrt(),
                );
            }
            let v_mean = vols.iter().sum::<f64>() / 20.0;
            out[[i, IDX_VOL_VOL]] =
                (vols.iter().map(|&x| (x - v_mean).powi(2)).sum::<f64>() / 20.0).sqrt();
        }

        let r_avg = rets[i - 5..i].iter().sum::<f64>() / 5.0;
        let o_avg = rets[i - 10..i - 5].iter().sum::<f64>() / 5.0;
        out[[i, IDX_VELOCITY]] = r_avg - o_avg;
    }
    for i in 0..n {
        out[[i, IDX_MARKET_REGIME]] = regime[i];
        out[[i, 13]] = volumes[i];
    }
}

pub fn filter_by_regime(uni: &Universe, is_bull: bool) -> (Vec<Array2<f64>>, Vec<Vec<f64>>) {
    let mut out_mats = Vec::new();
    let mut out_targets = Vec::new();
    for (m, t) in uni.data_matrices.iter().zip(uni.targets.iter()) {
        let (mut fm, mut ft) = (Vec::new(), Vec::new());
        for i in 0..m.nrows() {
            let r = m[[i, IDX_MARKET_REGIME]];
            if (is_bull && r < REGIME_THRESHOLD) || (!is_bull && r >= REGIME_THRESHOLD) {
                fm.push(m.row(i).to_owned());
                ft.push(t[i]);
            }
        }
        if !fm.is_empty() {
            let mut nm = Array2::zeros((fm.len(), m.ncols()).f());
            for (idx, row) in fm.into_iter().enumerate() {
                nm.row_mut(idx).assign(&row);
            }
            out_mats.push(nm);
            out_targets.push(ft);
        }
    }
    (out_mats, out_targets)
}

pub fn fetch_and_process(interval: &str, days: u64) -> Result<()> {
    println!(
        "Fetching {} days of {} data from Binance...",
        days, interval
    );
    let tickers = vec![
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "LINKUSDT",
    ];
    let mut records = Vec::new();
    for t in tickers {
        let url = format!(
            "https://api.binance.com/api/v3/klines?symbol={}&interval={}&limit=1000",
            t, interval
        );
        let resp: Vec<Vec<serde_json::Value>> = ureq::get(&url)
            .call()
            .map_err(|e| AetherError::Io(std::io::Error::other(e)))?
            .into_json()
            .map_err(|e| AetherError::Io(std::io::Error::other(e)))?;
        for k in resp {
            let dt = chrono::DateTime::from_timestamp(k[0].as_i64().unwrap_or(0) / 1000, 0)
                .unwrap_or_default()
                .format("%Y-%m-%d %H:%M:%S")
                .to_string();
            let c = k[4].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
            let v = k[5].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
            records.push(format!("{},{},{},{}", dt, c, v, t));
        }
    }
    fs::create_dir_all("data")?;
    println!("Writing output to data/market_data.csv (overwrites existing file).");
    fs::write(
        "data/market_data.csv",
        format!("date,close,volume,ticker\n{}", records.join("\n")),
    )
    .map_err(AetherError::Io)?;
    Ok(())
}

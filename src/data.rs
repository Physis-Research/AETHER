use crate::gp::{Terminal, TOTAL_FEATURES};
use crate::{AetherError, Result, Universe, FUNDING_RATE, REGIME_THRESHOLD, SLIPPAGE_FLOOR};
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
    crate::debug!("data: {} tickers, {} timestamps", keys.len(), n_time);

    let mut all_rets = Vec::new();
    for ticker in &keys {
        let mut series_rets = Vec::with_capacity(n_time);
        let mut last_close = 0.0;
        for d in &timeline {
            let close = raw[ticker].get(d).map(|v| v.0).unwrap_or(last_close);
            series_rets.push(if last_close > 1e-9 && close > 1e-9 {
                (close / last_close).ln()
            } else {
                0.0
            });
            if close > 1e-9 {
                last_close = close;
            }
        }
        all_rets.push(series_rets);
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

    let lead_v: Vec<f64> = (0..n_time)
        .map(|i| {
            let sum: f64 = all_rets.iter().map(|r| r[i]).sum();
            sum / all_rets.len() as f64
        })
        .collect();

    let mut data_mats = Vec::new();
    let mut targets = Vec::new();
    let mut tickers = Vec::new();

    for ticker in &keys {
        let mut series_closes = vec![0.0; n_time];
        let mut series_volumes = vec![0.0; n_time];
        let mut last_c = 0.0;
        for (i, d) in timeline.iter().enumerate() {
            let (c, v) = raw[ticker].get(d).cloned().unwrap_or((last_c, 0.0));
            series_closes[i] = c;
            series_volumes[i] = v;
            if c > 1e-9 {
                last_c = c;
            }
        }
        if series_closes.iter().any(|&c| c <= 1e-9) {
            crate::debug!("data: dropping {} due to missing/zero closes", ticker);
            continue;
        }

        let mut t = vec![0.0; n_time];
        for i in 0..n_time - 1 {
            t[i] = (series_closes[i + 1] / series_closes[i]).ln();
        }

        let mut matrix = Array2::<f64>::zeros((n_time, TOTAL_FEATURES).f());
        compute_physics_into(
            &series_closes,
            &series_volumes,
            &lead_v,
            &market_regime,
            &mut matrix,
        );
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
        let (r, ld, v) = (rets[i], lead_v[i], volumes[i]);
        out[[i, Terminal::RelMom as usize]] = r - ld;
        out[[i, Terminal::LeadV as usize]] = lead_v[i - 1];

        let v_avg = volumes[i - 5..i].iter().sum::<f64>() / 5.0;
        out[[i, Terminal::VolDelta as usize]] = if v_avg > 1e-9 {
            (v - v_avg) / v_avg
        } else {
            0.0
        };

        let change = (closes[i] - closes[i - 20]).abs();
        let path_len: f64 = (i - 19..=i)
            .map(|idx| (closes[idx] - closes[idx - 1]).abs())
            .sum();
        out[[i, Terminal::Efficiency as usize]] = if path_len > 1e-9 {
            change / path_len
        } else {
            0.0
        };

        out[[i, Terminal::Friction as usize]] = if v > 1e-9 {
            (closes[i] - closes[i - 1]).abs() / closes[i - 1] / (v / 1e6)
        } else {
            0.0
        };

        let m_win = &lead_v[i - 20..i];
        let a_win = &rets[i - 20..i];
        out[[i, Terminal::Correl as usize]] = crate::engine::pearson_correlation(m_win, a_win);

        if i >= 40 {
            let win = &rets[i - 40..i];
            let mean = win.iter().sum::<f64>() / 40.0;
            let (mut cd, mut mn, mut mx): (f64, f64, f64) = (0.0, 0.0, 0.0);
            for &rv in win {
                cd += rv - mean;
                mn = mn.min(cd);
                mx = mx.max(cd);
            }
            let std = (win.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 40.0).sqrt();
            out[[i, Terminal::Hurst as usize]] = if std > 1e-9 { (mx - mn) / std } else { 0.0 };

            let mut vols = Vec::with_capacity(20);
            for j in 0..20 {
                let sub = &rets[i - 40 + j..i - 20 + j];
                let sm = sub.iter().sum::<f64>() / 20.0;
                vols.push((sub.iter().map(|&x| (x - sm).powi(2)).sum::<f64>() / 20.0).sqrt());
            }
            let vm = vols.iter().sum::<f64>() / 20.0;
            out[[i, Terminal::VolVol as usize]] =
                (vols.iter().map(|&x| (x - vm).powi(2)).sum::<f64>() / 20.0).sqrt();
        }

        out[[i, Terminal::Velocity as usize]] = (rets[i - 5..i].iter().sum::<f64>() / 5.0)
            - (rets[i - 10..i - 5].iter().sum::<f64>() / 5.0);
        out[[i, Terminal::Resid as usize]] = r - (out[[i, Terminal::Correl as usize]] * ld);

        if i >= 60 {
            let win = &rets[i - 60..i];
            let l_ret: f64 = win.iter().sum();
            let mean = l_ret / 60.0;
            let std = (win.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 60.0).sqrt();
            out[[i, Terminal::LongRet as usize]] = l_ret;
            out[[i, Terminal::StdDev as usize]] = std;

            let (mut ef, mut es) = (rets[i - 1], rets[i - 1]);
            let (af, as_) = (2.0 / 11.0, 2.0 / 31.0);
            for j in 1..60 {
                ef = rets[i - j - 1] * af + ef * (1.0 - af);
                es = rets[i - j - 1] * as_ + es * (1.0 - as_);
            }
            out[[i, Terminal::EmaFast as usize]] = ef;
            out[[i, Terminal::EmaSlow as usize]] = es;
            out[[i, Terminal::BullStrength as usize]] = if ef > es { 1.0 } else { -1.0 };

            let (mut g, mut l) = (0.0, 0.0);
            for j in 0..14 {
                let d = rets[i - j] - rets[i - j - 1];
                if d > 0.0 {
                    g += d
                } else {
                    l -= d
                }
            }
            out[[i, Terminal::Rsi as usize]] =
                100.0 - (100.0 / (1.0 + (if l > 1e-9 { g / l } else { 100.0 })));

            let bm = rets[i - 20..=i].iter().sum::<f64>() / 21.0;
            let bs = (rets[i - 20..=i]
                .iter()
                .map(|&x| (x - bm).powi(2))
                .sum::<f64>()
                / 21.0)
                .sqrt();
            out[[i, Terminal::BbUpper as usize]] = bm + 2.0 * bs;
            out[[i, Terminal::BbLower as usize]] = bm - 2.0 * bs;
            out[[i, Terminal::BbWidth as usize]] = if bm.abs() > 1e-9 {
                (4.0 * bs) / bm
            } else {
                0.0
            };
        }
    }
    for i in 0..n {
        out[[i, Terminal::MarketRegime as usize]] = regime[i];
        out[[i, Terminal::AssetVol as usize]] = volumes[i];
        out[[i, Terminal::Slippage as usize]] = SLIPPAGE_FLOOR;
        out[[i, Terminal::Funding as usize]] = FUNDING_RATE;
    }
}

pub fn filter_by_regime(uni: &Universe, is_bull: bool) -> (Vec<Array2<f64>>, Vec<Vec<f64>>) {
    let mut out_mats = Vec::new();
    let mut out_targets = Vec::new();
    for (m, t) in uni.data_matrices.iter().zip(uni.targets.iter()) {
        let (mut fm, mut ft) = (Vec::new(), Vec::new());
        for i in 0..m.nrows() {
            let r = m[[i, Terminal::MarketRegime as usize]];
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
    crate::debug!(
        "data: regime={} selected {} instruments",
        if is_bull { "bull" } else { "bear" },
        out_mats.len()
    );
    (out_mats, out_targets)
}

pub fn fetch_and_process(
    source: &str,
    interval: &str,
    days: u64,
    tickers: Option<String>,
    append: bool,
) -> Result<()> {
    match source.to_lowercase().as_str() {
        "binance" => fetch_binance(interval, days, tickers, append),
        "yahoo" => fetch_yahoo(interval, days, tickers, append),
        _ => Err(AetherError::Integrity(format!(
            "Unsupported data source: {}",
            source
        ))),
    }
}

fn fetch_binance(interval: &str, _days: u64, tickers: Option<String>, append: bool) -> Result<()> {
    let tickers = tickers
        .map(|t| t.split(',').map(|s| s.to_string()).collect())
        .unwrap_or_else(|| {
            vec![
                "BTCUSDT".into(),
                "ETHUSDT".into(),
                "SOLUSDT".into(),
                "BNBUSDT".into(),
                "ADAUSDT".into(),
                "XRPUSDT".into(),
                "LINKUSDT".into(),
            ]
        });

    let mut records = Vec::new();
    for t in tickers {
        let url = format!(
            "https://api.binance.com/api/v3/klines?symbol={}&interval={}&limit=1000",
            t, interval
        );
        let resp: Vec<Vec<serde_json::Value>> = ureq::get(&url)
            .call()
            .map_err(|e| AetherError::Io(std::io::Error::other(e)))?
            .body_mut()
            .read_json()
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
    write_to_csv(records, append)
}

fn fetch_yahoo(interval: &str, days: u64, tickers: Option<String>, append: bool) -> Result<()> {
    let tickers = tickers
        .map(|t| t.split(',').map(|s| s.to_string()).collect())
        .unwrap_or_else(|| {
            vec![
                "AAPL".into(),
                "MSFT".into(),
                "GOOGL".into(),
                "AMZN".into(),
                "META".into(),
                "TSLA".into(),
                "NVDA".into(),
            ]
        });

    let range = match days {
        0..=1 => "1d",
        2..=5 => "5d",
        6..=30 => "1mo",
        31..=90 => "3mo",
        91..=180 => "6mo",
        181..=365 => "1y",
        366..=730 => "2y",
        731..=1825 => "5y",
        1826..=3650 => "10y",
        _ => "max",
    };

    let mut records = Vec::new();
    for t in tickers {
        let url = format!(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}?range={}&interval={}",
            t, range, interval
        );
        let resp: serde_json::Value = ureq::get(&url)
            .call()
            .map_err(|e| AetherError::Io(std::io::Error::other(e)))?
            .body_mut()
            .read_json()
            .map_err(|e| AetherError::Io(std::io::Error::other(e)))?;

        let result = &resp["chart"]["result"][0];
        let timestamps = result["timestamp"]
            .as_array()
            .ok_or_else(|| AetherError::Integrity(format!("No timestamps for ticker {}", t)))?;
        let indicators = &result["indicators"]["quote"][0];
        let closes = indicators["close"]
            .as_array()
            .ok_or_else(|| AetherError::Integrity(format!("No closes for ticker {}", t)))?;
        let volumes = indicators["volume"]
            .as_array()
            .ok_or_else(|| AetherError::Integrity(format!("No volumes for ticker {}", t)))?;

        for i in 0..timestamps.len() {
            let ts = timestamps[i].as_i64().unwrap_or(0);
            let c = closes[i].as_f64().unwrap_or(0.0);
            let v = volumes[i].as_f64().unwrap_or(0.0);
            if c > 1e-9 {
                let dt = chrono::DateTime::from_timestamp(ts, 0)
                    .unwrap_or_default()
                    .format("%Y-%m-%d %H:%M:%S")
                    .to_string();
                records.push(format!("{},{},{},{}", dt, c, v, t));
            }
        }
    }
    write_to_csv(records, append)
}

fn write_to_csv(records: Vec<String>, append: bool) -> Result<()> {
    fs::create_dir_all("data")?;
    let path = "data/market_data.csv";
    let header = "date,close,volume,ticker\n";
    if append && fs::metadata(path).is_ok() {
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(AetherError::Io)?;
        file.write_all(records.join("\n").as_bytes())
            .map_err(AetherError::Io)?;
        file.write_all(b"\n").map_err(AetherError::Io)?;
    } else {
        fs::write(path, format!("{}{}\n", header, records.join("\n"))).map_err(AetherError::Io)?;
    }
    Ok(())
}

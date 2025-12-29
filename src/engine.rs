use crate::gp::{crossover, evaluate, generate_rpn, mutate, Node, Terminal};
use crate::{
    Agent, AetherError, EvolutionConfig, Individual, Performance, RegimeAwareIndividual, Result,
    Universe, ValidationResult, ValidationSummary, FUNDING_RATE, IDX_MARKET_REGIME,
    REGIME_THRESHOLD, SLIPPAGE_FLOOR,
};
use ndarray::{Array2, ShapeBuilder};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

pub struct FitnessParams {
    pub sharpe: f64,
    pub sortino: f64,
    pub roi: f64,
    pub genome_length: usize,
}

pub fn compute_fitness(p: &FitnessParams, is_bull: bool) -> f64 {
    let bloat = (p.genome_length as f64 / 20.0).powi(2) * 0.02;
    let base = if is_bull {
        (p.roi.clamp(-1.0, 5.0) * 0.5) + (p.sortino.clamp(0.0, 10.0) * 0.3) + 0.2
    } else {
        (p.sharpe.clamp(0.0, 10.0) * 0.5) + (p.roi.clamp(-1.0, 5.0) * 0.3) + 0.2
    };
    (base - bloat).max(-1e9)
}

pub fn evolve(
    data: &[Array2<f64>],
    targets: &[Vec<f64>],
    config: EvolutionConfig,
    is_bull: bool,
) -> Individual {
    if data.is_empty() || targets.is_empty() || data[0].nrows() == 0 {
        return fallback_individual();
    }
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut pop_l = (0..config.pop_size)
        .map(|_| Agent {
            genome: generate_rpn(rng.random_range(3..=10), &mut rng),
            fitness: -1e9,
            performance: Performance::default(),
        })
        .collect::<Vec<_>>();
    let mut pop_s = (0..config.pop_size)
        .map(|_| Agent {
            genome: generate_rpn(rng.random_range(3..=10), &mut rng),
            fitness: -1e9,
            performance: Performance::default(),
        })
        .collect::<Vec<_>>();
    let mut best_l = pop_l[0].clone();
    let mut best_s = pop_s[0].clone();
    let mut best_fit = -1e9;
    let mut no_imp = 0;

    let n_samples = data[0].nrows();
    let split = (n_samples as f64 * 0.75) as usize;
    let wins = vec![0..split, split..n_samples];

    for gen in 1..=config.generations {
        let t = Instant::now();
        eval_pop(
            &mut pop_l, &best_s, true, data, targets, &wins, &config, is_bull,
        );
        pop_l.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        if pop_l[0].fitness > best_l.fitness {
            best_l = pop_l[0].clone();
        }

        eval_pop(
            &mut pop_s, &best_l, false, data, targets, &wins, &config, is_bull,
        );
        pop_s.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        if pop_s[0].fitness > best_s.fitness {
            best_s = pop_s[0].clone();
        }

        let fit = pop_l[0].fitness.max(pop_s[0].fitness);
        if fit > best_fit {
            best_fit = fit;
            no_imp = 0;
        } else {
            no_imp += 1;
        }

        if gen % 20 == 0 {
            println!(
                "gen {:>3} | fit: {:>7.3} | time: {:>5.2}s",
                gen,
                best_fit,
                t.elapsed().as_secs_f64()
            );
        }
        if no_imp >= config.early_stopping {
            break;
        }

        pop_l = next_gen(pop_l, config.pop_size, &mut rng);
        pop_s = next_gen(pop_s, config.pop_size, &mut rng);
    }
    Individual {
        long: best_l,
        short: best_s,
        performance: Performance::default(),
    }
}

fn fallback_individual() -> Individual {
    let genome = vec![Node::Terminal(Terminal::V)];
    Individual {
        long: Agent {
            genome: genome.clone(),
            fitness: -1e9,
            performance: Performance::default(),
        },
        short: Agent {
            genome,
            fitness: -1e9,
            performance: Performance::default(),
        },
        performance: Performance::default(),
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_pop(
    pop: &mut [Agent],
    partner: &Agent,
    is_l: bool,
    data: &[Array2<f64>],
    targets: &[Vec<f64>],
    wins: &[std::ops::Range<usize>],
    config: &EvolutionConfig,
    is_bull: bool,
) {
    let n_t = data.len();
    let n_s = data[0].nrows();
    let mut p_scores = Array2::<f64>::zeros((n_s, n_t).f());
    data.iter()
        .enumerate()
        .for_each(|(i, m)| evaluate(p_scores.column_mut(i), &partner.genome, m.view()));

    pop.par_iter_mut().for_each(|ind| {
        let (mut l_scores, mut s_scores) = (
            Array2::<f64>::zeros((n_s, n_t).f()),
            Array2::<f64>::zeros((n_s, n_t).f()),
        );
        if is_l {
            s_scores.assign(&p_scores);
        } else {
            l_scores.assign(&p_scores);
        }
        data.iter().enumerate().for_each(|(i, m)| {
            let mut res = ndarray::Array1::<f64>::zeros(n_s);
            evaluate(res.view_mut(), &ind.genome, m.view());
            if is_l {
                l_scores.column_mut(i).assign(&res);
            } else {
                s_scores.column_mut(i).assign(&res);
            }
        });
        let p_tr = backtest(
            &l_scores,
            &s_scores,
            targets,
            wins[0].clone(),
            config.sharpe_annualization,
        );
        let p_va = backtest(
            &l_scores,
            &s_scores,
            targets,
            wins[1].clone(),
            config.sharpe_annualization,
        );
        let f_tr = compute_fitness(
            &FitnessParams {
                sharpe: p_tr.sharpe,
                sortino: p_tr.sortino,
                roi: p_tr.roi,
                genome_length: ind.genome.len(),
            },
            is_bull,
        );
        let f_va = compute_fitness(
            &FitnessParams {
                sharpe: p_va.sharpe,
                sortino: p_va.sortino,
                roi: p_va.roi,
                genome_length: ind.genome.len(),
            },
            is_bull,
        );
        ind.fitness = 0.6 * f_tr + 0.4 * f_va;
        ind.performance = p_tr;
    });
}

fn next_gen(pop: Vec<Agent>, size: usize, rng: &mut StdRng) -> Vec<Agent> {
    let mut next = Vec::with_capacity(size);
    let elites = (size / 15).max(5);
    next.extend(pop.iter().take(elites).cloned());
    while next.len() < size {
        if rng.random_bool(0.15) {
            next.push(Agent {
                genome: generate_rpn(rng.random_range(3..=10), rng),
                fitness: -1e9,
                performance: Performance::default(),
            });
        } else {
            let p1 = &pop[rng.random_range(0..pop.len())];
            let genome = if rng.random_bool(0.7) {
                crossover(&p1.genome, &pop[rng.random_range(0..pop.len())].genome, rng)
            } else {
                mutate(&p1.genome, rng)
            };
            next.push(Agent {
                genome,
                fitness: -1e9,
                performance: Performance::default(),
            });
        }
    }
    next
}

pub fn backtest(
    l_s: &Array2<f64>,
    s_s: &Array2<f64>,
    targets: &[Vec<f64>],
    win: std::ops::Range<usize>,
    ann: f64,
) -> Performance {
    let (mut bal, mut peak, mut mdd, mut wins) = (1.0f64, 1.0f64, 0.0f64, 0usize);
    let mut returns = Vec::with_capacity(win.len());
    let mut last_pos = vec![0.0f64; targets.len()];

    for i in win {
        let (mut s_ret, mut n_pos, mut friction) = (0.0, 0.0, 0.0);
        for j in 0..targets.len() {
            let (l, s) = (l_s[[i, j]], s_s[[i, j]]);
            let mut curr_pos = 0.0;
            if l > s && l > 0.005 && (l - s).abs() > 0.002 {
                curr_pos = 1.0;
                s_ret += targets[j][i];
                n_pos += 1.0;
            } else if s > l && s > 0.005 && (l - s).abs() > 0.002 {
                curr_pos = -1.0;
                s_ret -= targets[j][i] + FUNDING_RATE;
                n_pos += 1.0;
            }
            friction += SLIPPAGE_FLOOR * (curr_pos - last_pos[j]).abs();
            last_pos[j] = curr_pos;
        }
        let step = if n_pos > 0.0 {
            ((s_ret - friction) / n_pos).exp()
        } else {
            1.0
        };
        bal *= step;
        returns.push(step - 1.0);
        if step > 1.0 {
            wins += 1;
        }
        peak = peak.max(bal);
        mdd = mdd.max((peak - bal) / peak);
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let std = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(1e-9);
    let down = (returns
        .iter()
        .filter(|&&r| r < 0.0)
        .map(|r| r.powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
        .max(1e-9);
    let (p_sum, n_sum) = returns.iter().fold((0.0, 0.0), |(p, n), &r| {
        if r > 0.0 {
            (p + r, n)
        } else {
            (p, n + r.abs())
        }
    });
    Performance {
        roi: bal - 1.0,
        mdd,
        sharpe: (mean / std) * ann,
        sortino: (mean / down) * ann,
        win_rate: wins as f64 / n.max(1.0),
        profit_factor: if n_sum > 1e-9 { p_sum / n_sum } else { 1.0 },
        trades: n as usize,
    }
}

pub fn walk_forward_validation(
    path: &str,
    folds: usize,
    train_size: f64,
    seed: u64,
) -> Result<ValidationSummary> {
    if folds == 0 {
        return Err(AetherError::Integrity("folds must be >= 1".to_string()));
    }
    let uni = crate::data::prepare_universe(path)?;
    if uni.dates.is_empty() || uni.targets.is_empty() || uni.data_matrices.is_empty() {
        return Err(AetherError::Integrity("no data available for validation".to_string()));
    }
    let f_size = uni.dates.len() / folds;
    let mut res = Vec::new();
    let mut skipped = 0usize;
    let config = EvolutionConfig {
        seed,
        ..Default::default()
    };
    for fold in 0..folds {
        let s = fold * f_size;
        let e = ((fold + 1) * f_size).min(uni.dates.len());
        let sp = s + ((e - s) as f64 * train_size) as usize;
        if sp >= e || e - sp < 50 {
            skipped += 1;
            continue;
        }
        let t_uni = Universe {
            dates: uni.dates[s..sp].to_vec(),
            tickers: uni.tickers.clone(),
            data_matrices: uni
                .data_matrices
                .iter()
                .map(|m: &Array2<f64>| m.slice(ndarray::s![s..sp, ..]).to_owned())
                .collect(),
            targets: uni.targets.iter().map(|t| t[s..sp].to_vec()).collect(),
        };
        let (bull_mats, bull_targs) = crate::data::filter_by_regime(&t_uni, true);
        let bull = evolve(&bull_mats, &bull_targs, config.clone(), true);
        let (bear_mats, bear_targs) = crate::data::filter_by_regime(&t_uni, false);
        let bear = evolve(&bear_mats, &bear_targs, config.clone(), false);
        let mut model = RegimeAwareIndividual {
            bull,
            bear,
            performance: Performance::default(),
        };
        let v_uni = Universe {
            dates: uni.dates[sp..e].to_vec(),
            tickers: uni.tickers.clone(),
            data_matrices: uni
                .data_matrices
                .iter()
                .map(|m: &Array2<f64>| m.slice(ndarray::s![sp..e, ..]).to_owned())
                .collect(),
            targets: uni.targets.iter().map(|t| t[sp..e].to_vec()).collect(),
        };
        evaluate_regime_aware(&mut model, &v_uni, config.sharpe_annualization);
        res.push(ValidationResult {
            fold: fold + 1,
            test_roi: model.performance.roi,
            market_roi: v_uni.targets[0].iter().sum::<f64>().exp() - 1.0,
            test_sharpe: model.performance.sharpe,
        });
    }
    if skipped > 0 {
        eprintln!("warning: skipped {} fold(s) due to insufficient data", skipped);
    }
    if res.is_empty() {
        return Ok(ValidationSummary {
            folds: res,
            mean_test_roi: 0.0,
            baseline_buy_hold: uni.targets[0].iter().sum::<f64>().exp() - 1.0,
            beats_baseline: false,
        });
    }
    let mean_roi = res.iter().map(|r| r.test_roi).sum::<f64>() / res.len() as f64;
    Ok(ValidationSummary {
        folds: res,
        mean_test_roi: mean_roi,
        baseline_buy_hold: uni.targets[0].iter().sum::<f64>().exp() - 1.0,
        beats_baseline: mean_roi > 0.0,
    })
}

pub fn evaluate_regime_aware(ind: &mut RegimeAwareIndividual, uni: &Universe, ann: f64) {
    let (n_t, n_s) = (uni.data_matrices.len(), uni.data_matrices[0].nrows());
    let (mut l_s, mut s_s) = (
        Array2::<f64>::zeros((n_s, n_t).f()),
        Array2::<f64>::zeros((n_s, n_t).f()),
    );
    for i in 0..n_s {
        for j in 0..n_t {
            let regime = uni.data_matrices[j][[i, IDX_MARKET_REGIME]];
            let model = if regime < REGIME_THRESHOLD {
                &ind.bull
            } else {
                &ind.bear
            };
            let (mut res_l, mut res_s) = (
                ndarray::Array1::<f64>::zeros(1),
                ndarray::Array1::<f64>::zeros(1),
            );
            evaluate(
                res_l.view_mut(),
                &model.long.genome,
                uni.data_matrices[j].slice(ndarray::s![i..=i, ..]),
            );
            evaluate(
                res_s.view_mut(),
                &model.short.genome,
                uni.data_matrices[j].slice(ndarray::s![i..=i, ..]),
            );
            l_s[[i, j]] = res_l[0];
            s_s[[i, j]] = res_s[0];
        }
    }
    ind.performance = backtest(&l_s, &s_s, &uni.targets, 0..n_s, ann);
}

pub fn run_noise_audit(uni: &Universe) {
    println!("\n=== NOISE AUDIT: TARGET RANDOMIZATION ===");
    let mut rng = StdRng::seed_from_u64(42);
    let real_sum: f64 = uni
        .data_matrices
        .iter()
        .zip(uni.targets.iter())
        .map(|(m, t)| {
            (0..crate::gp::TOTAL_FEATURES)
                .map(|f| pearson_correlation(&m.column(f).to_vec(), t).abs())
                .sum::<f64>()
        })
        .sum();
    let mut better = 0;
    for _ in 0..50 {
        let mut r_sum = 0.0;
        for (m, t) in uni.data_matrices.iter().zip(uni.targets.iter()) {
            let mut shuf = t.clone();
            shuf.shuffle(&mut rng);
            r_sum += (0..crate::gp::TOTAL_FEATURES)
                .map(|f| pearson_correlation(&m.column(f).to_vec(), &shuf).abs())
                .sum::<f64>();
        }
        if r_sum >= real_sum {
            better += 1;
        }
    }
    println!(
        "P-Value: {:.4} (Target randomization)",
        better as f64 / 50.0
    );
}

pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let (m_x, m_y) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut cov, mut v_x, mut v_y) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let (dx, dy) = (x[i] - m_x, y[i] - m_y);
        cov += dx * dy;
        v_x += dx * dx;
        v_y += dy * dy;
    }
    if v_x < 1e-10 || v_y < 1e-10 {
        0.0
    } else {
        cov / (v_x.sqrt() * v_y.sqrt())
    }
}

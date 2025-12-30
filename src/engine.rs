use crate::gp::{crossover, evaluate, generate_rpn, mutate, Node, Terminal};
use crate::{
    Agent, EvolutionConfig, Individual, Performance, PhysisError, RegimeAwareIndividual, Result,
    Universe, ValidationResult, ValidationSummary, FUNDING_RATE, REGIME_THRESHOLD,
};
use ndarray::{Array2, ShapeBuilder};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

fn compute_fitness(
    _sharpe: f64,
    sortino: f64,
    roi: f64,
    mdd: f64,
    len: usize,
    is_bull: bool,
) -> f64 {
    let bloat = (len as f64 / 15.0).powi(2) * 0.15;
    let base = if is_bull {
        (roi.clamp(-0.9, 15.0) * 0.4) + (sortino.clamp(0.0, 10.0) * 0.6)
    } else {
        (roi.clamp(-0.9, 8.0) * 0.3) + (sortino.clamp(0.0, 10.0) * 0.7)
    };
    (base - bloat - (mdd * 3.0).powi(2)).max(-1e9)
}

pub fn evolve(
    data: &[Array2<f64>],
    targets: &[Vec<f64>],
    config: EvolutionConfig,
    is_bull: bool,
) -> Individual {
    if data.is_empty() || targets.is_empty() || data[0].nrows() < 100 {
        return fallback_individual();
    }
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut pop_l = (0..config.pop_size)
        .map(|_| Agent {
            genome: generate_rpn(rng.random_range(4..=12), &mut rng),
            fitness: -1e9,
            performance: Performance::default(),
        })
        .collect::<Vec<_>>();
    let mut pop_s = (0..config.pop_size)
        .map(|_| Agent {
            genome: generate_rpn(rng.random_range(4..=12), &mut rng),
            fitness: -1e9,
            performance: Performance::default(),
        })
        .collect::<Vec<_>>();
    let (mut best_l, mut best_s) = (pop_l[0].clone(), pop_s[0].clone());
    let (mut best_fit, mut no_imp) = (-1e9, 0);

    let (n_s, _n_t) = (data[0].nrows(), data.len());
    let split = (n_s as f64 * 0.75) as usize;
    let wins = [0..split, split..n_s];

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
    let genome = vec![Node::Terminal(Terminal::Velocity)];
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
    let (n_t, n_s) = (data.len(), data[0].nrows());
    let mut p_scores = Array2::<f64>::zeros((n_s, n_t).f());
    for (i, m) in data.iter().enumerate() {
        evaluate(p_scores.column_mut(i), &partner.genome, m.view());
    }

    let exp_t: Vec<Vec<f64>> = targets
        .iter()
        .map(|t| t.iter().map(|&v| v.exp() - 1.0).collect())
        .collect();

    pop.par_iter_mut().for_each_init(
        || {
            (
                Array2::<f64>::zeros((n_s, n_t).f()),
                Array2::<f64>::zeros((n_s, n_t).f()),
            )
        },
        |(l_s, s_s), ind| {
            if is_l {
                s_s.assign(&p_scores);
            } else {
                l_s.assign(&p_scores);
            }
            for (i, m) in data.iter().enumerate() {
                if is_l {
                    evaluate(l_s.column_mut(i), &ind.genome, m.view());
                } else {
                    evaluate(s_s.column_mut(i), &ind.genome, m.view());
                }
            }

            let mut corr = 0.0;
            let sc = if is_l { &*l_s } else { &*s_s };
            for j in 0..n_t {
                corr += pearson_correlation(
                    sc.column(j).as_slice().unwrap(),
                    p_scores.column(j).as_slice().unwrap(),
                )
                .abs();
            }
            let div_p = (corr / n_t.max(1) as f64).powi(2) * 0.1;

            let p_tr = backtest(
                l_s,
                s_s,
                data,
                &exp_t,
                wins[0].clone(),
                config.sharpe_annualization,
            );
            let p_va = backtest(
                l_s,
                s_s,
                data,
                &exp_t,
                wins[1].clone(),
                config.sharpe_annualization,
            );
            let f_tr = compute_fitness(
                p_tr.sharpe,
                p_tr.sortino,
                p_tr.roi,
                p_tr.mdd,
                ind.genome.len(),
                is_bull,
            );
            let f_va = compute_fitness(
                p_va.sharpe,
                p_va.sortino,
                p_va.roi,
                p_va.mdd,
                ind.genome.len(),
                is_bull,
            );
            ind.fitness = (0.3 * f_tr + 0.7 * f_va) - div_p;
            ind.performance = p_va;
        },
    );
}

fn next_gen(pop: Vec<Agent>, size: usize, rng: &mut StdRng) -> Vec<Agent> {
    let mut next = Vec::with_capacity(size);
    let elites = (size / 20).max(2);
    next.extend(pop.iter().take(elites).cloned());

    let tournament = |rng: &mut StdRng, pop: &[Agent], size: usize| -> Agent {
        let mut best = &pop[rng.random_range(0..pop.len())];
        for _ in 0..size {
            let candidate = &pop[rng.random_range(0..pop.len())];
            if candidate.fitness > best.fitness {
                best = candidate;
            }
        }
        best.clone()
    };

    while next.len() < size {
        let r = rng.random_range(0.0..1.0);
        if r < 0.15 {
            next.push(Agent {
                genome: generate_rpn(rng.random_range(4..=12), rng),
                fitness: -1e9,
                performance: Performance::default(),
            });
        } else if r < 0.75 {
            let p1 = tournament(rng, &pop, 4);
            let p2 = tournament(rng, &pop, 4);
            let genome = crossover(&p1.genome, &p2.genome, rng);
            next.push(Agent {
                genome,
                fitness: -1e9,
                performance: Performance::default(),
            });
        } else {
            let p1 = tournament(rng, &pop, 4);
            let genome = mutate(&p1.genome, rng);
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
    data: &[Array2<f64>],
    exp_t: &[Vec<f64>],
    win: std::ops::Range<usize>,
    ann: f64,
) -> Performance {
    let (mut bal, mut peak, mut mdd, mut wins): (f64, f64, f64, usize) = (1.0, 1.0, 0.0, 0);
    let mut rets = Vec::with_capacity(win.len());
    let n_a = exp_t.len();
    if n_a == 0 {
        return Performance::default();
    }
    let mut last_pos = vec![0.0; n_a];
    let slip_idx = Terminal::Slippage as usize;

    for i in win {
        if i >= exp_t[0].len() - 1 {
            break;
        }
        let (mut s_ret, mut n_p, mut fric) = (0.0, 0.0, 0.0);
        for j in 0..n_a {
            let (l, s, lp) = (l_s[[i, j]], s_s[[i, j]], last_pos[j]);
            let mut cp = 0.0;
            if l > s {
                if l > (if lp > 0.02 { 0.001 } else { 0.01 }) {
                    cp = l.min(1.0);
                    s_ret += exp_t[j][i] * cp;
                    n_p += cp;
                }
            } else if s > l && s > (if lp < -0.02 { 0.001 } else { 0.01 }) {
                cp = -s.min(1.0);
                s_ret += exp_t[j][i] * cp - FUNDING_RATE;
                n_p += s.min(1.0);
            }
            fric += data[j][[i, slip_idx]] * (cp - lp).abs();
            last_pos[j] = cp;
        }
        let step = if n_p > 0.0 {
            (((s_ret - fric) / n_p) + 1.0).clamp(0.7, 1.3)
        } else {
            1.0
        };
        bal *= step;
        rets.push(step - 1.0);
        if step > 1.0 {
            wins += 1;
        }
        peak = peak.max(bal);
        mdd = mdd.max((peak - bal) / peak);
    }
    let n = rets.len() as f64;
    let mean = rets.iter().sum::<f64>() / n.max(1.0);
    let std = (rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n.max(1.0))
        .sqrt()
        .max(1e-9);
    let down = (rets
        .iter()
        .filter(|&&r| r < 0.0)
        .map(|r| r.powi(2))
        .sum::<f64>()
        / n.max(1.0))
    .sqrt()
    .max(1e-9);
    let (ps, ns) = rets.iter().fold((0.0, 0.0), |(p, n), &r| {
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
        profit_factor: if ns > 1e-9 { ps / ns } else { 1.0 },
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
        return Err(PhysisError::Integrity("folds >= 1".into()));
    }
    let uni = crate::data::prepare_universe(path)?;
    let f_size = uni.dates.len() / folds;
    let mut res = Vec::new();
    let config = EvolutionConfig {
        seed,
        ..Default::default()
    };

    for fold in 0..folds {
        let (s, e) = (fold * f_size, ((fold + 1) * f_size).min(uni.dates.len()));
        let sp = s + ((e - s) as f64 * train_size) as usize;
        if sp >= e || e - sp < 50 {
            continue;
        }

        let t_uni = Universe {
            dates: uni.dates[s..sp].to_vec(),
            tickers: uni.tickers.clone(),
            data_matrices: uni
                .data_matrices
                .iter()
                .map(|m| m.slice(ndarray::s![s..sp, ..]).to_owned())
                .collect(),
            targets: uni.targets.iter().map(|t| t[s..sp].to_vec()).collect(),
        };
        let (bull_m, bull_t) = crate::data::filter_by_regime(&t_uni, true);
        let (bear_m, bear_t) = crate::data::filter_by_regime(&t_uni, false);
        let mut model = RegimeAwareIndividual {
            bull: evolve(&bull_m, &bull_t, config.clone(), true),
            bear: evolve(&bear_m, &bear_t, config.clone(), false),
            performance: Performance::default(),
        };

        let v_uni = Universe {
            dates: uni.dates[sp..e].to_vec(),
            tickers: uni.tickers.clone(),
            data_matrices: uni
                .data_matrices
                .iter()
                .map(|m| m.slice(ndarray::s![sp..e, ..]).to_owned())
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
    let mean_roi = if res.is_empty() {
        0.0
    } else {
        res.iter().map(|r| r.test_roi).sum::<f64>() / res.len() as f64
    };
    Ok(ValidationSummary {
        folds: res,
        mean_test_roi: mean_roi,
        baseline_buy_hold: uni.targets[0].iter().sum::<f64>().exp() - 1.0,
        beats_baseline: mean_roi > 0.0,
    })
}

pub fn evaluate_regime_aware(ind: &mut RegimeAwareIndividual, uni: &Universe, ann: f64) {
    let (n_t, n_s) = (uni.data_matrices.len(), uni.data_matrices[0].nrows());
    let (mut bl_l, mut bl_s, mut br_l, mut br_s) = (
        Array2::<f64>::zeros((n_s, n_t).f()),
        Array2::<f64>::zeros((n_s, n_t).f()),
        Array2::<f64>::zeros((n_s, n_t).f()),
        Array2::<f64>::zeros((n_s, n_t).f()),
    );

    for j in 0..n_t {
        evaluate(
            bl_l.column_mut(j),
            &ind.bull.long.genome,
            uni.data_matrices[j].view(),
        );
        evaluate(
            bl_s.column_mut(j),
            &ind.bull.short.genome,
            uni.data_matrices[j].view(),
        );
        evaluate(
            br_l.column_mut(j),
            &ind.bear.long.genome,
            uni.data_matrices[j].view(),
        );
        evaluate(
            br_s.column_mut(j),
            &ind.bear.short.genome,
            uni.data_matrices[j].view(),
        );
    }

    let (mut l_s, mut s_s) = (
        Array2::<f64>::zeros((n_s, n_t).f()),
        Array2::<f64>::zeros((n_s, n_t).f()),
    );
    for i in 0..n_s {
        for j in 0..n_t {
            if uni.data_matrices[j][[i, Terminal::MarketRegime as usize]] < REGIME_THRESHOLD {
                l_s[[i, j]] = bl_l[[i, j]];
                s_s[[i, j]] = bl_s[[i, j]];
            } else {
                l_s[[i, j]] = br_l[[i, j]];
                s_s[[i, j]] = br_s[[i, j]];
            }
        }
    }
    let exp_t: Vec<Vec<f64>> = uni
        .targets
        .iter()
        .map(|t| t.iter().map(|&v| v.exp() - 1.0).collect())
        .collect();
    ind.performance = backtest(&l_s, &s_s, &uni.data_matrices, &exp_t, 0..n_s, ann);
}

pub fn run_noise_audit(uni: &Universe) {
    let terms = crate::gp::terminals();
    let mut real_c = Vec::new();
    for (m, t) in uni.data_matrices.iter().zip(uni.targets.iter()) {
        real_c.push(
            terms
                .iter()
                .map(|&idx| pearson_correlation(&m.column(idx as usize).to_vec(), t).abs())
                .collect::<Vec<_>>(),
        );
    }

    let names = [
        "velocity",
        "rel_mom",
        "vol_delta",
        "correl",
        "hurst",
        "friction",
        "efficiency",
        "lead_v",
        "regime",
        "asset_vol",
        "resid",
        "vol_vol",
        "slippage",
        "funding",
        "long_ret",
        "std_dev",
        "bull_strength",
        "ema_fast",
        "ema_slow",
        "rsi",
        "bb_upper",
        "bb_lower",
        "bb_width",
    ];
    for (i, &idx) in terms.iter().enumerate() {
        println!(
            "{}: {:.6}",
            names[idx as usize],
            real_c.iter().map(|c| c[i]).sum::<f64>() / real_c.len() as f64
        );
    }

    let r_sum: f64 = real_c.iter().flat_map(|c| c.iter()).sum();
    let (mut better, mut rng) = (0, StdRng::seed_from_u64(42));
    for _ in 0..50 {
        let mut s_sum = 0.0;
        for (m, t) in uni.data_matrices.iter().zip(uni.targets.iter()) {
            let mut sh = t.clone();
            sh.shuffle(&mut rng);
            s_sum += terms
                .iter()
                .map(|&idx| pearson_correlation(&m.column(idx as usize).to_vec(), &sh).abs())
                .sum::<f64>();
        }
        if s_sum >= r_sum {
            better += 1;
        }
    }
    println!("p_value: {:.4}", better as f64 / 50.0);
}

pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let (sx, sy) = x
        .iter()
        .zip(y.iter())
        .fold((0.0, 0.0), |(ax, ay), (&vx, &vy)| (ax + vx, ay + vy));
    let (mx, my) = (sx / n, sy / n);
    let (c, vx, vy) = x
        .iter()
        .zip(y.iter())
        .fold((0.0, 0.0, 0.0), |(ac, avx, avy), (&xv, &yv)| {
            let (dx, dy) = (xv - mx, yv - my);
            (ac + dx * dy, avx + dx * dx, avy + dy * dy)
        });
    if vx < 1e-10 || vy < 1e-10 {
        0.0
    } else {
        c / (vx * vy).sqrt()
    }
}

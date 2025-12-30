use ndarray::{ArrayView2, ArrayViewMut1};
use rand::Rng;
use serde::{Deserialize, Serialize};
pub const TOTAL_FEATURES: usize = 23;
const CHUNK_SIZE: usize = 512;
const MAX_STACK_DEPTH: usize = 128;
const MAX_GENOME_LEN: usize = 64;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Abs,
    Tanh,
    Log,
    Min,
    Max,
    Gate,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(usize)]
pub enum Terminal {
    Velocity = 0,
    RelMom = 1,
    VolDelta = 2,
    Correl = 3,
    Hurst = 4,
    Friction = 5,
    Efficiency = 6,
    LeadV = 7,
    MarketRegime = 8,
    AssetVol = 9,
    Resid = 10,
    VolVol = 11,
    Slippage = 12,
    Funding = 13,
    LongRet = 14,
    StdDev = 15,
    BullStrength = 16,
    EmaFast = 17,
    EmaSlow = 18,
    Rsi = 19,
    BbUpper = 20,
    BbLower = 21,
    BbWidth = 22,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Node {
    Op(Op, u8),
    Terminal(Terminal),
    Constant(f64),
}
const OPS: [(Op, u8); 10] = [
    (Op::Add, 2),
    (Op::Sub, 2),
    (Op::Mul, 2),
    (Op::Div, 2),
    (Op::Tanh, 1),
    (Op::Gate, 3),
    (Op::Abs, 1),
    (Op::Log, 1),
    (Op::Min, 2),
    (Op::Max, 2),
];
const TERMINALS: [Terminal; 23] = [
    Terminal::Velocity,
    Terminal::RelMom,
    Terminal::VolDelta,
    Terminal::Correl,
    Terminal::Hurst,
    Terminal::Friction,
    Terminal::Efficiency,
    Terminal::LeadV,
    Terminal::MarketRegime,
    Terminal::AssetVol,
    Terminal::Resid,
    Terminal::VolVol,
    Terminal::Slippage,
    Terminal::Funding,
    Terminal::LongRet,
    Terminal::StdDev,
    Terminal::BullStrength,
    Terminal::EmaFast,
    Terminal::EmaSlow,
    Terminal::Rsi,
    Terminal::BbUpper,
    Terminal::BbLower,
    Terminal::BbWidth,
];

pub fn terminals() -> &'static [Terminal] {
    &TERMINALS
}

pub fn evaluate(mut res: ArrayViewMut1<'_, f64>, genome: &[Node], data: ArrayView2<'_, f64>) {
    let n_samples = data.nrows();
    let mut stack = [[0.0f64; CHUNK_SIZE]; MAX_STACK_DEPTH];
    let raw = data.as_slice_memory_order().unwrap();
    let cols: Vec<_> = (0..TOTAL_FEATURES)
        .map(|i| &raw[i * n_samples..(i + 1) * n_samples])
        .collect();

    for chunk_start in (0..n_samples).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(n_samples);
        let size = chunk_end - chunk_start;
        let mut sp = 0;

        for node in genome {
            match node {
                Node::Terminal(t) => {
                    stack[sp][..size].copy_from_slice(&cols[*t as usize][chunk_start..chunk_end]);
                    sp += 1;
                }
                Node::Constant(v) => {
                    stack[sp][..size].fill(*v);
                    sp += 1;
                }
                Node::Op(op, _) => match op {
                    Op::Add => {
                        sp -= 1;
                        let (s1, s0) = stack.split_at_mut(sp);
                        s1[sp - 1][..size]
                            .iter_mut()
                            .zip(s0[0][..size].iter())
                            .for_each(|(a, b)| *a += b);
                    }
                    Op::Sub => {
                        sp -= 1;
                        let (s1, s0) = stack.split_at_mut(sp);
                        s1[sp - 1][..size]
                            .iter_mut()
                            .zip(s0[0][..size].iter())
                            .for_each(|(a, b)| *a -= b);
                    }
                    Op::Mul => {
                        sp -= 1;
                        let (s1, s0) = stack.split_at_mut(sp);
                        s1[sp - 1][..size]
                            .iter_mut()
                            .zip(s0[0][..size].iter())
                            .for_each(|(a, b)| *a *= b);
                    }
                    Op::Div => {
                        sp -= 1;
                        let (s1, s0) = stack.split_at_mut(sp);
                        s1[sp - 1][..size]
                            .iter_mut()
                            .zip(s0[0][..size].iter())
                            .for_each(|(a, b)| *a = if b.abs() > 1e-9 { *a / b } else { 0.0 });
                    }
                    Op::Abs => {
                        stack[sp - 1][..size].iter_mut().for_each(|x| *x = x.abs());
                    }
                    Op::Tanh => {
                        stack[sp - 1][..size].iter_mut().for_each(|x| *x = x.tanh());
                    }
                    Op::Log => {
                        stack[sp - 1][..size]
                            .iter_mut()
                            .for_each(|x| *x = (x.abs() + 1e-9).ln());
                    }
                    Op::Min => {
                        sp -= 1;
                        let (s1, s0) = stack.split_at_mut(sp);
                        s1[sp - 1][..size]
                            .iter_mut()
                            .zip(s0[0][..size].iter())
                            .for_each(|(a, b)| *a = a.min(*b));
                    }
                    Op::Max => {
                        sp -= 1;
                        let (s1, s0) = stack.split_at_mut(sp);
                        s1[sp - 1][..size]
                            .iter_mut()
                            .zip(s0[0][..size].iter())
                            .for_each(|(a, b)| *a = a.max(*b));
                    }
                    Op::Gate => {
                        sp -= 2;
                        let (s1, s0) = stack.split_at_mut(sp);
                        let cond = &s1[sp - 1][..size];
                        let if_true = &s0[0][..size];
                        let if_false = &s0[1][..size];
                        let mut out = [0.0f64; CHUNK_SIZE];
                        for i in 0..size {
                            out[i] = if cond[i] > 0.0 {
                                if_true[i]
                            } else {
                                if_false[i]
                            };
                        }
                        s1[sp - 1][..size].copy_from_slice(&out[..size]);
                    }
                },
            }
        }
        if sp > 0 {
            let res_chunk = &mut res.as_slice_mut().unwrap()[chunk_start..chunk_end];
            res_chunk.copy_from_slice(&stack[0][..size]);
            res_chunk.iter_mut().for_each(|x| {
                if !x.is_finite() {
                    *x = 0.0;
                }
            });
        }
    }
}
pub fn generate_rpn(depth: usize, rng: &mut impl Rng) -> Vec<Node> {
    let mut genome = Vec::with_capacity(MAX_GENOME_LEN);
    build_recursive(depth.clamp(1, 8), &mut genome, rng);
    if !validate_genome(&genome) {
        vec![Node::Terminal(Terminal::Velocity)]
    } else {
        genome
    }
}
fn build_recursive(depth: usize, genome: &mut Vec<Node>, rng: &mut impl Rng) {
    let term_prob = if depth <= 1 { 0.8 } else { 0.3 };
    if depth == 0 || rng.random_bool(term_prob) {
        if rng.random_bool(0.15) {
            genome.push(Node::Constant(rng.random_range(-50..=50) as f64 / 10.0));
        } else {
            genome.push(Node::Terminal(
                TERMINALS[rng.random_range(0..TERMINALS.len())],
            ));
        }
    } else {
        let (op, arity) = OPS[rng.random_range(0..OPS.len())];
        for _ in 0..arity {
            build_recursive(depth - 1, genome, rng);
        }
        genome.push(Node::Op(op, arity));
    }
}
pub fn validate_genome(genome: &[Node]) -> bool {
    let mut depth = 0i32;
    for node in genome {
        depth += 1;
        if depth > MAX_STACK_DEPTH as i32 {
            return false;
        }
        if let Node::Op(_, arity) = node {
            depth -= *arity as i32;
        }
        if depth < 1 {
            return false;
        }
    }
    depth == 1
}
pub fn crossover(p1: &[Node], p2: &[Node], rng: &mut impl Rng) -> Vec<Node> {
    let r1 = get_subtree_range(p1, rng.random_range(0..p1.len()));
    let r2 = get_subtree_range(p2, rng.random_range(0..p2.len()));
    let mut child = Vec::with_capacity(MAX_GENOME_LEN);
    child.extend_from_slice(&p1[..r1.start]);
    child.extend_from_slice(&p2[r2.start..r2.end]);
    child.extend_from_slice(&p1[r1.end..]);
    child.truncate(MAX_GENOME_LEN);
    if validate_genome(&child) {
        child
    } else {
        p1.to_vec()
    }
}
pub fn mutate(genome: &[Node], rng: &mut impl Rng) -> Vec<Node> {
    let range = get_subtree_range(genome, rng.random_range(0..genome.len()));
    let mut mutated = Vec::with_capacity(MAX_GENOME_LEN);
    mutated.extend_from_slice(&genome[..range.start]);
    build_recursive(rng.random_range(1..=4), &mut mutated, rng);
    mutated.extend_from_slice(&genome[range.end..]);
    mutated.truncate(MAX_GENOME_LEN);
    if validate_genome(&mutated) {
        mutated
    } else {
        genome.to_vec()
    }
}
fn get_subtree_range(genome: &[Node], end_idx: usize) -> std::ops::Range<usize> {
    let mut stack = 0i32;
    for i in (0..=end_idx).rev() {
        stack += 1;
        if let Node::Op(_, arity) = genome[i] {
            stack -= arity as i32;
        }
        if stack == 1 {
            return i..end_idx + 1;
        }
    }
    end_idx..end_idx + 1
}

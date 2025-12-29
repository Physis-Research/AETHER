use ndarray::{ArrayView2, ArrayViewMut1};
use rand::Rng;
use serde::{Deserialize, Serialize};
pub const TOTAL_FEATURES: usize = 16;
const CHUNK_SIZE: usize = 512;
const MAX_STACK_DEPTH: usize = 64;
const MAX_GENOME_LEN: usize = 30;
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
    Sigmoid,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Terminal {
    V = 0,
    RelMom = 1,
    VolDelta = 2,
    Correl = 3,
    F = 4,
    K = 5,
    L = 6,
    Friction = 7,
    Hurst = 8,
    LeadV = 9,
    FastFric = 10,
    SlowFric = 11,
    Regime = 12,
    AssetVol = 13,
    Resid = 14,
    VolVol = 15,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Node {
    Op(Op, u8),
    Terminal(Terminal),
    Constant(f64),
}
const OPS: [(Op, u8); 7] = [
    (Op::Add, 2),
    (Op::Sub, 2),
    (Op::Mul, 2),
    (Op::Div, 2),
    (Op::Tanh, 1),
    (Op::Gate, 3),
    (Op::Sigmoid, 1),
];
const TERMINALS: [Terminal; 16] = [
    Terminal::V,
    Terminal::RelMom,
    Terminal::VolDelta,
    Terminal::Correl,
    Terminal::F,
    Terminal::K,
    Terminal::L,
    Terminal::Friction,
    Terminal::Hurst,
    Terminal::LeadV,
    Terminal::FastFric,
    Terminal::SlowFric,
    Terminal::Regime,
    Terminal::AssetVol,
    Terminal::Resid,
    Terminal::VolVol,
];
pub fn terminals() -> &'static [Terminal] {
    &TERMINALS
}
#[inline(always)]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
#[allow(clippy::needless_range_loop)]
pub fn evaluate(mut res: ArrayViewMut1<'_, f64>, genome: &[Node], data: ArrayView2<'_, f64>) {
    let n_samples = data.nrows();
    let mut stack = [[0.0f64; CHUNK_SIZE]; MAX_STACK_DEPTH];
    for chunk_start in (0..n_samples).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(n_samples);
        let size = chunk_end - chunk_start;
        let mut sp = 0;
        for node in genome {
            match node {
                Node::Terminal(t) => {
                    let col = data.column(*t as usize);
                    for i in 0..size {
                        stack[sp][i] = col[chunk_start + i];
                    }
                    sp += 1;
                }
                Node::Constant(v) => {
                    stack[sp][..size].fill(*v);
                    sp += 1;
                }
                Node::Op(op, _) => match op {
                    Op::Add => {
                        for i in 0..size {
                            stack[sp - 2][i] += stack[sp - 1][i];
                        }
                        sp -= 1;
                    }
                    Op::Sub => {
                        for i in 0..size {
                            stack[sp - 2][i] -= stack[sp - 1][i];
                        }
                        sp -= 1;
                    }
                    Op::Mul => {
                        for i in 0..size {
                            stack[sp - 2][i] *= stack[sp - 1][i];
                        }
                        sp -= 1;
                    }
                    Op::Div => {
                        for i in 0..size {
                            stack[sp - 2][i] = if stack[sp - 1][i].abs() > 1e-9 {
                                stack[sp - 2][i] / stack[sp - 1][i]
                            } else {
                                0.0
                            };
                        }
                        sp -= 1;
                    }
                    Op::Abs => {
                        for i in 0..size {
                            stack[sp - 1][i] = stack[sp - 1][i].abs();
                        }
                    }
                    Op::Tanh => {
                        for i in 0..size {
                            stack[sp - 1][i] = stack[sp - 1][i].tanh();
                        }
                    }
                    Op::Log => {
                        for i in 0..size {
                            stack[sp - 1][i] = (stack[sp - 1][i].abs() + 1e-9).ln();
                        }
                    }
                    Op::Min => {
                        for i in 0..size {
                            stack[sp - 2][i] = stack[sp - 2][i].min(stack[sp - 1][i]);
                        }
                        sp -= 1;
                    }
                    Op::Max => {
                        for i in 0..size {
                            stack[sp - 2][i] = stack[sp - 2][i].max(stack[sp - 1][i]);
                        }
                        sp -= 1;
                    }
                    Op::Gate => {
                        for i in 0..size {
                            stack[sp - 3][i] = if stack[sp - 3][i] > 0.0 {
                                stack[sp - 2][i]
                            } else {
                                stack[sp - 1][i]
                            };
                        }
                        sp -= 2;
                    }
                    Op::Sigmoid => {
                        for i in 0..size {
                            stack[sp - 1][i] = sigmoid(stack[sp - 1][i]);
                        }
                    }
                },
            }
        }
        if sp > 0 {
            for i in 0..size {
                res[chunk_start + i] = stack[0][i];
            }
        }
    }
}
pub fn generate_rpn(depth: usize, rng: &mut impl Rng) -> Vec<Node> {
    let mut genome = Vec::with_capacity(MAX_GENOME_LEN);
    build_recursive(depth.clamp(1, 8), &mut genome, rng);
    if !validate_genome(&genome) {
        vec![Node::Terminal(Terminal::V)]
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
pub fn simplify(genome: &[Node]) -> Vec<Node> {
    let mut stack: Vec<Vec<Node>> = Vec::new();
    for node in genome {
        match node {
            Node::Terminal(_) | Node::Constant(_) => stack.push(vec![*node]),
            Node::Op(op, arity) => {
                if stack.len() >= *arity as usize {
                    let mut args: Vec<_> = (0..*arity).map(|_| stack.pop().unwrap()).collect();
                    args.reverse();
                    if args
                        .iter()
                        .all(|a| a.len() == 1 && matches!(a[0], Node::Constant(_)))
                    {
                        let v: Vec<_> = args
                            .iter()
                            .map(|a| if let Node::Constant(x) = a[0] { x } else { 0.0 })
                            .collect();
                        let res = match op {
                            Op::Add => Some(v[0] + v[1]),
                            Op::Sub => Some(v[0] - v[1]),
                            Op::Mul => Some(v[0] * v[1]),
                            Op::Div => Some(if v[1].abs() > 1e-9 { v[0] / v[1] } else { 0.0 }),
                            Op::Tanh => Some(v[0].tanh()),
                            Op::Sigmoid => Some(sigmoid(v[0])),
                            _ => None,
                        };
                        if let Some(val) = res {
                            stack.push(vec![Node::Constant(val)]);
                            continue;
                        }
                    }
                    let mut combined = Vec::new();
                    for a in args {
                        combined.extend(a);
                    }
                    combined.push(*node);
                    stack.push(combined);
                }
            }
        }
    }
    stack.pop().unwrap_or_else(|| genome.to_vec())
}
pub fn genome_distance(g1: &[Node], g2: &[Node]) -> f64 {
    let mut dist = (g1.len() as f64 - g2.len() as f64).abs() * 0.5;
    for i in 0..g1.len().min(g2.len()) {
        dist += match (&g1[i], &g2[i]) {
            (Node::Op(o1, _), Node::Op(o2, _)) if o1 == o2 => 0.0,
            (Node::Terminal(t1), Node::Terminal(t2)) if t1 == t2 => 0.0,
            (Node::Constant(v1), Node::Constant(v2)) => (v1 - v2).abs().min(1.0),
            _ => 1.0,
        };
    }
    dist
}

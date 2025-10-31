use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Normal;
use rayon::prelude::*;
use std::f64::consts::E;
use clap::Parser;
use serde::Deserialize;
use std::fs;

// ============================
// CLI and Config
// ============================

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration to use: local, prod, or test
    #[arg(default_value = "local")]
    config: String,
}

#[derive(Debug, Deserialize)]
struct Config {
    beam_size: usize,
    seeds: usize,
    parallel_chunks: usize,
    patience: usize,
    iterations: usize,
    description: String,
}

#[derive(Debug, Deserialize)]
struct ConfigFile {
    local: Config,
    prod: Config,
    prod_ultra: Config,
    prod_max: Config,
    prod_mega: Config,
    test: Config,
}

// ============================
// Constants
// ============================

const GRID_N: usize = 20;
const NUM_TOWNS: usize = GRID_N * GRID_N;
const NUM_ORBS: usize = 10;
const MIN_SEP: f64 = 2.5;

const S_CORNER: f64 = 0.098765432;
const S_BOUND: f64 = 0.135792468;
const S_INTER: f64 = 0.246913579;

// ============================
// Problem setup
// ============================

#[inline]
fn population(x: f64, y: f64) -> f64 {
    50.0 + (20.0 * (0.35 * x).sin()
        + 15.0 * (0.30 * y).cos()
        + 10.0 * (0.20 * (x + y)).sin())
    .floor()
}

#[inline]
fn selfie_rate(x: usize, y: usize) -> f64 {
    let is_corner = (x == 0 && y == 0)
        || (x == 0 && y == GRID_N - 1)
        || (x == GRID_N - 1 && y == 0)
        || (x == GRID_N - 1 && y == GRID_N - 1);

    if is_corner {
        return S_CORNER;
    }

    let on_boundary = x == 0 || y == 0 || x == GRID_N - 1 || y == GRID_N - 1;
    if on_boundary {
        S_BOUND
    } else {
        S_INTER
    }
}

#[inline]
fn alpha(d: f64) -> f64 {
    0.5 * E.powf(-0.40 * d)
}

struct Problem {
    towns: Array2<f64>,  // (400, 2)
    weights: Array1<f64>, // (400,)
}

impl Problem {
    fn new() -> Self {
        let mut towns = Array2::<f64>::zeros((NUM_TOWNS, 2));
        let mut weights = Array1::<f64>::zeros(NUM_TOWNS);

        let mut idx = 0;
        for y in 0..GRID_N {
            for x in 0..GRID_N {
                let fx = x as f64;
                let fy = y as f64;
                towns[[idx, 0]] = fx;
                towns[[idx, 1]] = fy;

                let p = population(fx, fy);
                let s = selfie_rate(x, y);
                weights[idx] = p * (1.0 - s);
                idx += 1;
            }
        }

        Problem { towns, weights }
    }

    #[inline]
    fn objective(&self, orbs: &Array2<f64>) -> f64 {
        let mut total = 0.0;
        for i in 0..NUM_TOWNS {
            let tx = self.towns[[i, 0]];
            let ty = self.towns[[i, 1]];
            let mut min_dist_sq = f64::MAX;

            for j in 0..NUM_ORBS {
                let ox = orbs[[j, 0]];
                let oy = orbs[[j, 1]];
                let dx = tx - ox;
                let dy = ty - oy;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                }
            }

            total += self.weights[i] * alpha(min_dist_sq.sqrt());
        }
        total
    }

    fn nearest_indices(&self, orbs: &Array2<f64>) -> Vec<usize> {
        let mut result = vec![0; NUM_TOWNS];
        for i in 0..NUM_TOWNS {
            let tx = self.towns[[i, 0]];
            let ty = self.towns[[i, 1]];
            let mut min_dist_sq = f64::MAX;
            let mut nearest = 0;

            for j in 0..NUM_ORBS {
                let ox = orbs[[j, 0]];
                let oy = orbs[[j, 1]];
                let dx = tx - ox;
                let dy = ty - oy;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                    nearest = j;
                }
            }
            result[i] = nearest;
        }
        result
    }
}

// ============================
// Geometry utilities
// ============================

#[inline]
fn distance(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    (dx * dx + dy * dy).sqrt()
}

#[inline]
fn clip(val: f64, min: f64, max: f64) -> f64 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

fn project_minsep(orbs: &Array2<f64>, idx: usize, cand: [f64; 2], min_sep: f64) -> [f64; 2] {
    let mut p = cand;

    for _ in 0..8 {
        let mut moved = false;
        for j in 0..NUM_ORBS {
            if j == idx {
                continue;
            }
            let other = [orbs[[j, 0]], orbs[[j, 1]]];
            let mut dx = p[0] - other[0];
            let mut dy = p[1] - other[1];
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < min_sep {
                moved = true;
                if dist < 1e-9 {
                    // Random nudge
                    let mut rng = thread_rng();
                    dx = rng.gen::<f64>() - 0.5;
                    dy = rng.gen::<f64>() - 0.5;
                    let norm = (dx * dx + dy * dy).sqrt();
                    dx /= norm;
                    dy /= norm;
                    p[0] = other[0] + dx * (min_sep + 1e-6);
                    p[1] = other[1] + dy * (min_sep + 1e-6);
                } else {
                    let scale = (min_sep + 1e-6) / dist;
                    p[0] = other[0] + dx * scale;
                    p[1] = other[1] + dy * scale;
                }
            }
        }
        if !moved {
            break;
        }
        p[0] = clip(p[0], 0.0, (GRID_N - 1) as f64);
        p[1] = clip(p[1], 0.0, (GRID_N - 1) as f64);
    }
    p
}

// ============================
// Seeds
// ============================

fn edge_biased_seed() -> Array2<f64> {
    let data = vec![
        1.0, 1.0, 6.5, 1.0, 11.0, 1.0, 15.5, 1.0, 1.0, 18.0, 6.5, 18.0, 11.0, 18.0, 15.5, 18.0,
        1.0, 9.5, 18.0, 9.5,
    ];
    Array2::from_shape_vec((NUM_ORBS, 2), data).unwrap()
}

fn weighted_farthest_seed<R: Rng>(
    problem: &Problem,
    rng: &mut R,
    trials_per_pick: usize,
) -> Array2<f64> {
    let mut centers: Vec<[f64; 2]> = Vec::with_capacity(NUM_ORBS);

    // Pick first center
    let total_weight: f64 = problem.weights.sum();
    let mut r = rng.gen::<f64>() * total_weight;
    let mut idx0 = 0;
    for i in 0..NUM_TOWNS {
        r -= problem.weights[i];
        if r <= 0.0 {
            idx0 = i;
            break;
        }
    }
    centers.push([problem.towns[[idx0, 0]], problem.towns[[idx0, 1]]]);

    // Iteratively pick remaining centers
    for _ in 1..NUM_ORBS {
        let mut best_score = -1.0;
        let mut best_town: Option<[f64; 2]> = None;

        for _ in 0..trials_per_pick {
            let cand_idx = rng.gen_range(0..NUM_TOWNS);
            let cand = [problem.towns[[cand_idx, 0]], problem.towns[[cand_idx, 1]]];

            let mut ok = true;
            let mut min_d = f64::MAX;
            for c in &centers {
                let d = distance(&cand, c);
                if d < MIN_SEP {
                    ok = false;
                    break;
                }
                if d < min_d {
                    min_d = d;
                }
            }
            if !ok {
                continue;
            }

            let score = problem.weights[cand_idx] * E.powf(-0.4 * min_d);
            if score > best_score {
                best_score = score;
                best_town = Some(cand);
            }
        }

        if let Some(town) = best_town {
            centers.push(town);
        } else {
            // Fallback: random position
            for _ in 0..200 {
                let pos = [
                    rng.gen::<f64>() * (GRID_N - 1) as f64,
                    rng.gen::<f64>() * (GRID_N - 1) as f64,
                ];
                let mut ok = true;
                for c in &centers {
                    if distance(&pos, c) < MIN_SEP {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    best_town = Some(pos);
                    break;
                }
            }
            centers.push(best_town.unwrap_or_else(|| {
                let last = centers.last().unwrap();
                [last[0] + rng.gen::<f64>() - 0.5, last[1] + rng.gen::<f64>() - 0.5]
            }));
        }
    }

    let mut orbs = Array2::<f64>::zeros((NUM_ORBS, 2));
    for i in 0..NUM_ORBS {
        orbs[[i, 0]] = centers[i][0];
        orbs[[i, 1]] = centers[i][1];
    }

    // Jitter and project
    let normal = Normal::new(0.0, 0.15).unwrap();
    for i in 0..NUM_ORBS {
        let jx = rng.sample(normal);
        let jy = rng.sample(normal);
        let cand = [
            clip(orbs[[i, 0]] + jx, 0.0, (GRID_N - 1) as f64),
            clip(orbs[[i, 1]] + jy, 0.0, (GRID_N - 1) as f64),
        ];
        let proj = project_minsep(&orbs, i, cand, MIN_SEP);
        orbs[[i, 0]] = proj[0];
        orbs[[i, 1]] = proj[1];
    }

    orbs
}

// ============================
// Local improvement
// ============================

fn improve_once<R: Rng>(problem: &Problem, orbs: &Array2<f64>, rng: &mut R) -> (Array2<f64>, f64) {
    let mut best_orbs = orbs.clone();
    let mut best_val = problem.objective(&best_orbs);

    let assign = problem.nearest_indices(&best_orbs);

    // Per-orb gradient steps
    for i in 0..NUM_ORBS {
        // Collect towns assigned to orb i
        let mut cluster_towns = Vec::new();
        let mut cluster_weights = Vec::new();
        for t in 0..NUM_TOWNS {
            if assign[t] == i {
                cluster_towns.push([problem.towns[[t, 0]], problem.towns[[t, 1]]]);
                cluster_weights.push(problem.weights[t]);
            }
        }
        if cluster_towns.is_empty() {
            continue;
        }

        let mut oi = [best_orbs[[i, 0]], best_orbs[[i, 1]]];

        for _ in 0..15 {
            let mut grad_x = 0.0;
            let mut grad_y = 0.0;

            for (idx, town) in cluster_towns.iter().enumerate() {
                let dx = oi[0] - town[0];
                let dy = oi[1] - town[1];
                let r = (dx * dx + dy * dy).sqrt() + 1e-9;
                let coeff = cluster_weights[idx] * (-0.4) * E.powf(-0.4 * r) / r;
                grad_x += coeff * dx;
                grad_y += coeff * dy;
            }

            if grad_x.abs() < 1e-12 && grad_y.abs() < 1e-12 {
                break;
            }

            let mut step_x = -0.6 * grad_x;
            let mut step_y = -0.6 * grad_y;
            step_x = clip(step_x, -2.0, 2.0);
            step_y = clip(step_y, -2.0, 2.0);

            // Backtracking line search
            let mut step_scale = 1.0;
            let mut accepted = false;
            for _ in 0..8 {
                let cand = [
                    clip(oi[0] + step_scale * step_x, 0.0, (GRID_N - 1) as f64),
                    clip(oi[1] + step_scale * step_y, 0.0, (GRID_N - 1) as f64),
                ];
                let cand_proj = project_minsep(&best_orbs, i, cand, MIN_SEP);

                let mut cand_orbs = best_orbs.clone();
                cand_orbs[[i, 0]] = cand_proj[0];
                cand_orbs[[i, 1]] = cand_proj[1];

                let cand_val = problem.objective(&cand_orbs);
                if cand_val > best_val + 1e-9 {
                    best_orbs = cand_orbs;
                    best_val = cand_val;
                    oi = cand_proj;
                    accepted = true;
                    break;
                }
                step_scale *= 0.5;
            }
            if !accepted {
                break;
            }
        }
    }

    // Small coordinate tweaks (hill-climb) - more granular steps
    let deltas = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];
    let dirs = [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [0.707, 0.707],
        [0.707, -0.707],
        [-0.707, 0.707],
        [-0.707, -0.707],
    ];

    for i in 0..NUM_ORBS {
        let mut improved = true;
        while improved {
            improved = false;
            for &dval in &deltas {
                for dir in &dirs {
                    let cand = [
                        clip(best_orbs[[i, 0]] + dval * dir[0], 0.0, (GRID_N - 1) as f64),
                        clip(best_orbs[[i, 1]] + dval * dir[1], 0.0, (GRID_N - 1) as f64),
                    ];
                    let cand_proj = project_minsep(&best_orbs, i, cand, MIN_SEP);

                    let mut cand_orbs = best_orbs.clone();
                    cand_orbs[[i, 0]] = cand_proj[0];
                    cand_orbs[[i, 1]] = cand_proj[1];

                    let val = problem.objective(&cand_orbs);
                    if val > best_val + 1e-9 {
                        best_orbs = cand_orbs;
                        best_val = val;
                        improved = true;
                    }
                }
            }
        }
    }

    // Multi-orb shake with simulated annealing acceptance
    if rng.gen::<f64>() < 0.2 {
        let num_shakes = rng.gen_range(1..=2);
        let normal = Normal::new(0.0, 1.2).unwrap();
        let mut cand_orbs = best_orbs.clone();
        
        for _ in 0..num_shakes {
            let j = rng.gen_range(0..NUM_ORBS);
            let nudge = [rng.sample(normal), rng.sample(normal)];
            let cand = [
                clip(cand_orbs[[j, 0]] + nudge[0], 0.0, (GRID_N - 1) as f64),
                clip(cand_orbs[[j, 1]] + nudge[1], 0.0, (GRID_N - 1) as f64),
            ];
            let cand_proj = project_minsep(&cand_orbs, j, cand, MIN_SEP);
            cand_orbs[[j, 0]] = cand_proj[0];
            cand_orbs[[j, 1]] = cand_proj[1];
        }

        let val = problem.objective(&cand_orbs);
        // Accept if better, or occasionally accept slightly worse (simulated annealing)
        if val > best_val || (val > best_val - 0.5 && rng.gen::<f64>() < 0.1) {
            best_orbs = cand_orbs;
            best_val = val;
        }
    }

    (best_orbs, best_val)
}

// ============================
// Search
// ============================

#[derive(Clone)]
struct Result {
    orbs: Array2<f64>,
    value: f64,
}

fn search(problem: &Problem, max_outer: usize, seed_beam: usize, random_seed: u64, patience: usize) -> Result {
    let mut rng = StdRng::seed_from_u64(random_seed);

    // Generate seeds
    let mut seeds = Vec::new();

    // Edge-biased seed
    seeds.push(edge_biased_seed());

    // Jittered edge-biased seeds
    for _ in 0..4 {
        let mut s = edge_biased_seed();
        let normal = Normal::new(0.0, 0.25).unwrap();
        for i in 0..NUM_ORBS {
            let jx = rng.sample(normal);
            let jy = rng.sample(normal);
            s[[i, 0]] += jx;
            s[[i, 1]] += jy;
            let cand = [
                clip(s[[i, 0]], 0.0, (GRID_N - 1) as f64),
                clip(s[[i, 1]], 0.0, (GRID_N - 1) as f64),
            ];
            let proj = project_minsep(&s, i, cand, MIN_SEP);
            s[[i, 0]] = proj[0];
            s[[i, 1]] = proj[1];
        }
        seeds.push(s);
    }

    // Weighted farthest seeds - more trials for better quality
    while seeds.len() < seed_beam {
        seeds.push(weighted_farthest_seed(problem, &mut rng, 150));
    }

    // Rank seeds
    let mut beam: Vec<Result> = seeds
        .into_iter()
        .map(|s| {
            let val = problem.objective(&s);
            Result { orbs: s, value: val }
        })
        .collect();
    beam.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
    beam.truncate(seed_beam);

    // Early stopping variables
    let mut best_so_far = beam[0].value;
    let mut no_improve_count = 0;

    // Outer improvement loop
    for iter in 0..max_outer {
        // Parallel improvement of beam members
        let new_results: Vec<(Array2<f64>, f64)> = beam
            .par_iter()
            .map(|r| {
                let mut local_rng = StdRng::seed_from_u64(random_seed + iter as u64 + 1);
                improve_once(problem, &r.orbs, &mut local_rng)
            })
            .collect();

        let mut new_beam = Vec::new();
        for (orbs, val) in new_results {
            new_beam.push(Result { orbs, value: val });
        }
        // Also keep originals
        new_beam.extend(beam.clone());

        // Sort and deduplicate
        new_beam.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        let mut unique = Vec::new();
        for cand in new_beam {
            if unique.is_empty() {
                unique.push(cand);
                continue;
            }

            let mut is_dupe = false;
            for u in &unique {
                let mut dist_sq = 0.0;
                for i in 0..NUM_ORBS {
                    for j in 0..2 {
                        let diff = cand.orbs[[i, j]] - u.orbs[[i, j]];
                        dist_sq += diff * diff;
                    }
                }
                if dist_sq < 1e-12 {
                    is_dupe = true;
                    break;
                }
            }

            if !is_dupe {
                unique.push(cand);
            }
            if unique.len() >= seed_beam {
                break;
            }
        }

        beam = unique;

        // Check for improvement and early stopping
        let improved = beam[0].value > best_so_far + 1e-9;
        if improved {
            best_so_far = beam[0].value;
            no_improve_count = 0;
            println!("  Iter {}: {:.6} ⬆️ IMPROVED", iter + 1, beam[0].value);
        } else {
            no_improve_count += 1;
            println!("  Iter {}: {:.6} (no improve: {})", iter + 1, beam[0].value, no_improve_count);
        }

        // Early stopping
        if no_improve_count >= patience {
            println!("✓ Early stopping at iteration {} (no improvement for {} iterations)", 
                     iter + 1, patience);
            break;
        }
    }

    beam.into_iter()
        .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap())
        .unwrap()
}

// ============================
// Main
// ============================

fn main() {
    // Parse command line arguments
    let args = Args::parse();
    
    println!("Orb Optimizer (Rust) — initializing...");
    
    // Load configuration
    let config_content = fs::read_to_string("config.toml")
        .expect("Failed to read config.toml");
    let config_file: ConfigFile = toml::from_str(&config_content)
        .expect("Failed to parse config.toml");
    
    let config = match args.config.as_str() {
        "local" => config_file.local,
        "prod" => config_file.prod,
        "prod_ultra" => config_file.prod_ultra,
        "prod_max" => config_file.prod_max,
        "prod_mega" => config_file.prod_mega,
        "test" => config_file.test,
        _ => {
            eprintln!("Unknown config: {}. Using 'local'", args.config);
            config_file.local
        }
    };
    
    println!("Configuration: {} - {}", args.config, config.description);
    println!("  Beam size: {}", config.beam_size);
    println!("  Seeds: {}", config.seeds);
    println!("  Parallel chunks: {}", config.parallel_chunks);
    println!("  Patience: {}", config.patience);
    println!("  Max iterations: {}\n", config.iterations);

    let problem = Problem::new();
    println!("Problem setup complete: {} towns, {} orbs", NUM_TOWNS, NUM_ORBS);
    println!("Target: 3200.51+\n");

    // Generate seeds based on config
    let seeds_to_try: Vec<u64> = (1..=config.seeds as u64).map(|i| i * 1234 + i * i).collect();
    
    println!("Processing {} seeds in parallel ({} at a time)...\n", seeds_to_try.len(), config.parallel_chunks);
    
    // Process seeds in parallel chunks
    let chunk_size = config.parallel_chunks;
    let mut best_overall: Option<Result> = None;
    let mut seeds_processed = 0;
    
    for chunk in seeds_to_try.chunks(chunk_size) {
        let chunk_results: Vec<(usize, Result)> = chunk
            .par_iter()
            .enumerate()
            .map(|(chunk_idx, &seed)| {
                let result = search(&problem, config.iterations, config.beam_size, seed, config.patience);
                let global_idx = seeds_processed + chunk_idx;
                println!("  Seed {}/{} ({}): {:.6}", global_idx + 1, config.seeds, seed, result.value);
                
                // Print coordinates for HIGH scores only (> 3200.51)
                if result.value > 3200.51 {
                    println!("    🎯 BREAKTHROUGH! Score: {:.6} - Coordinates:", result.value);
                    for i in 0..NUM_ORBS {
                        println!("      Orb {}: ({:.6}, {:.6})", i + 1, result.orbs[[i, 0]], result.orbs[[i, 1]]);
                    }
                }
                
                (global_idx, result)
            })
            .collect();
        
        // Check results and update best
        for (_idx, result) in chunk_results {
            let value = result.value;
            
            let is_better = if let Some(ref best) = best_overall {
                value > best.value
            } else {
                true
            };

            if is_better {
                println!("🔥 New best: {:.6}", value);
                best_overall = Some(result);
                
                // Early exit if we hit the target
                if value > 3200.51 {
                    println!("\n🎯 TARGET REACHED!");
                    break;
                }
            }
        }
        
        seeds_processed += chunk.len();
        
        // Early exit from outer loop if target reached
        if let Some(ref best) = best_overall {
            if best.value > 3200.51 {
                break;
            }
        }
        
        // Progress update every 100 seeds (or every 10 for test mode)
        let update_freq = if config.seeds <= 100 { 10 } else { 100 };
        if seeds_processed % update_freq == 0 {
            if let Some(ref best) = best_overall {
                println!("Progress: {}/{} seeds, Current best: {:.6}", seeds_processed, config.seeds, best.value);
            }
        }
    }

    let result = best_overall.unwrap();

    println!("\n=== FINAL RESULT ===");
    println!("Best objective value: {:.6}", result.value);
    if result.value > 3200.51 {
        println!("✅ SUCCESS! Score exceeds 3200.51");
    } else {
        println!("⚠️  Score below target (3200.51)");
    }
    println!("\nOrb coordinates (x, y):");
    for i in 0..NUM_ORBS {
        println!("({:.6}, {:.6})", result.orbs[[i, 0]], result.orbs[[i, 1]]);
    }
}


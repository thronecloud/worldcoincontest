use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Normal;
use rayon::prelude::*;
use std::f64::consts::E;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use chrono::{DateTime, Utc};
use std::path::Path;
use std::io::{Write, BufWriter};
use std::fs::{OpenOptions, File};
use std::sync::{Arc, Mutex};

// ============================
// CLI and Config
// ============================

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration to use: local, prod, test, or "view" to just see results
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
    local_max: Config,
    prod: Config,
    prod_ultra: Config,
    prod_max: Config,
    prod_mega: Config,
    test: Config,
}

// ============================
// Logging System
// ============================

struct Logger {
    file: Arc<Mutex<BufWriter<File>>>,
}

impl Logger {
    fn new() -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("bglogs.txt")
            .expect("Failed to open bglogs.txt");
        
        let buffered = BufWriter::new(file);
        Logger {
            file: Arc::new(Mutex::new(buffered)),
        }
    }
    
    fn log(&self, message: &str) {
        // Print to stdout and flush immediately for tmux
        println!("{}", message);
        std::io::stdout().flush().ok();
        
        // Also write to log file
        if let Ok(mut file) = self.file.lock() {
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            writeln!(file, "[{}] {}", timestamp, message).ok();
            file.flush().ok();
        }
    }
}

// Create global logger - will be initialized lazily
lazy_static::lazy_static! {
    static ref LOGGER: Logger = Logger::new();
}

// Macro for easy logging
macro_rules! log_info {
    ($($arg:tt)*) => {{
        let message = format!($($arg)*);
        LOGGER.log(&message);
    }};
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

// Pre-computed alpha lookup table for performance
// Max distance on 19x19 grid is sqrt(19^2 + 19^2) ≈ 26.87
// We use 0.01 precision, so 2700 entries
const ALPHA_TABLE_SIZE: usize = 2700;
const ALPHA_TABLE_STEP: f64 = 0.01;

lazy_static::lazy_static! {
    static ref ALPHA_LOOKUP: Vec<f64> = {
        let mut table = Vec::with_capacity(ALPHA_TABLE_SIZE);
        for i in 0..ALPHA_TABLE_SIZE {
            let d = i as f64 * ALPHA_TABLE_STEP;
            table.push(0.5 * E.powf(-0.40 * d));
        }
        table
    };
}

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
    // Use lookup table for common distances (huge speedup, avoids exp())
    let idx = (d / ALPHA_TABLE_STEP) as usize;
    if idx < ALPHA_TABLE_SIZE - 1 {
        // Linear interpolation for better accuracy
        let frac = (d / ALPHA_TABLE_STEP) - idx as f64;
        ALPHA_LOOKUP[idx] * (1.0 - frac) + ALPHA_LOOKUP[idx + 1] * frac
    } else {
        // Fallback for distances beyond table
        0.5 * E.powf(-0.40 * d)
    }
}

struct Problem {
    towns: Array2<f64>,  // (400, 2)
    weights: Array1<f64>, // (400,)
    // Pre-computed for faster access
    towns_flat: Vec<(f64, f64)>,  // Flat array of (x, y) for cache locality
    // Spatial grid for faster nearest-neighbor queries
    // Grid cells of size 3x3 to partition the 19x19 space
    spatial_grid: Vec<Vec<usize>>,  // Each cell contains town indices
    grid_cell_size: f64,
}

impl Problem {
    fn new() -> Self {
        let mut towns = Array2::<f64>::zeros((NUM_TOWNS, 2));
        let mut weights = Array1::<f64>::zeros(NUM_TOWNS);
        let mut towns_flat = Vec::with_capacity(NUM_TOWNS);

        let mut idx = 0;
        for y in 0..GRID_N {
            for x in 0..GRID_N {
                let fx = x as f64;
                let fy = y as f64;
                towns[[idx, 0]] = fx;
                towns[[idx, 1]] = fy;
                towns_flat.push((fx, fy));

                let p = population(fx, fy);
                let s = selfie_rate(x, y);
                weights[idx] = p * (1.0 - s);
                idx += 1;
            }
        }

        // Build spatial grid (7x7 grid covering the 19x19 space)
        let grid_dim = 7;
        let grid_cell_size = 19.0 / grid_dim as f64;
        let mut spatial_grid = vec![Vec::new(); grid_dim * grid_dim];
        
        for (town_idx, &(tx, ty)) in towns_flat.iter().enumerate() {
            let gx = ((tx / grid_cell_size) as usize).min(grid_dim - 1);
            let gy = ((ty / grid_cell_size) as usize).min(grid_dim - 1);
            let grid_idx = gy * grid_dim + gx;
            spatial_grid[grid_idx].push(town_idx);
        }

        Problem { 
            towns, 
            weights,
            towns_flat,
            spatial_grid,
            grid_cell_size,
        }
    }

    #[inline]
    fn objective(&self, orbs: &Array2<f64>) -> f64 {
        // Pre-extract orb positions for better cache locality
        let mut orb_positions = [(0.0, 0.0); NUM_ORBS];
        for j in 0..NUM_ORBS {
            orb_positions[j] = (orbs[[j, 0]], orbs[[j, 1]]);
        }

        let mut total = 0.0;
        
        // Use flat array for better cache performance
        for i in 0..NUM_TOWNS {
            let (tx, ty) = self.towns_flat[i];
            let mut min_dist_sq = f64::MAX;

            // Manual unrolling for 10 orbs (compiler should optimize this well)
            for &(ox, oy) in &orb_positions {
                let dx = tx - ox;
                let dy = ty - oy;
                let dist_sq = dx * dx + dy * dy;
                min_dist_sq = min_dist_sq.min(dist_sq);
            }

            total += self.weights[i] * alpha(min_dist_sq.sqrt());
        }
        total
    }

    fn nearest_indices(&self, orbs: &Array2<f64>) -> Vec<usize> {
        // Pre-extract orb positions for better cache locality
        let mut orb_positions = [(0.0, 0.0); NUM_ORBS];
        for j in 0..NUM_ORBS {
            orb_positions[j] = (orbs[[j, 0]], orbs[[j, 1]]);
        }

        let mut result = vec![0; NUM_TOWNS];
        for i in 0..NUM_TOWNS {
            let (tx, ty) = self.towns_flat[i];
            let mut min_dist_sq = f64::MAX;
            let mut nearest = 0;

            for (j, &(ox, oy)) in orb_positions.iter().enumerate() {
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

    // Incremental objective: compute delta when moving one orb
    // Much faster than full recomputation - only checks affected towns
    #[inline]
    fn objective_delta(&self, orbs: &Array2<f64>, orb_idx: usize, new_pos: [f64; 2]) -> f64 {
        let old_pos = (orbs[[orb_idx, 0]], orbs[[orb_idx, 1]]);
        
        // Pre-extract all orb positions
        let mut orb_positions = [(0.0, 0.0); NUM_ORBS];
        for j in 0..NUM_ORBS {
            if j == orb_idx {
                orb_positions[j] = (new_pos[0], new_pos[1]);
            } else {
                orb_positions[j] = (orbs[[j, 0]], orbs[[j, 1]]);
            }
        }

        let mut delta = 0.0;
        
        // Only check towns that might be affected by this orb move
        // A town is affected if the moving orb is within consideration distance
        for i in 0..NUM_TOWNS {
            let (tx, ty) = self.towns_flat[i];
            
            // Distance to old and new position
            let old_dx = tx - old_pos.0;
            let old_dy = ty - old_pos.1;
            let old_dist_to_moved = (old_dx * old_dx + old_dy * old_dy).sqrt();
            
            let new_dx = tx - new_pos[0];
            let new_dy = ty - new_pos[1];
            let new_dist_to_moved = (new_dx * new_dx + new_dy * new_dy).sqrt();
            
            // Find nearest orb distance in both scenarios
            let mut old_min = old_dist_to_moved;
            let mut new_min = new_dist_to_moved;
            
            for (j, &(ox, oy)) in orb_positions.iter().enumerate() {
                if j == orb_idx {
                    continue;  // Already handled
                }
                let dx = tx - ox;
                let dy = ty - oy;
                let dist = (dx * dx + dy * dy).sqrt();
                old_min = old_min.min(dist);
                new_min = new_min.min(dist);
            }
            
            // Only add delta if there's a change in nearest orb
            if (old_min - new_min).abs() > 1e-12 {
                delta += self.weights[i] * (alpha(new_min) - alpha(old_min));
            }
        }
        
        delta
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

// K-means++ style initialization for better coverage
fn kmeans_plus_plus_seed<R: Rng>(problem: &Problem, rng: &mut R) -> Array2<f64> {
    let mut centers: Vec<[f64; 2]> = Vec::with_capacity(NUM_ORBS);
    
    // Pick first center weighted by population
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
    
    // Pick remaining centers with probability proportional to squared distance to nearest center
    for _ in 1..NUM_ORBS {
        let mut distances = vec![f64::MAX; NUM_TOWNS];
        
        // Calculate min distance to existing centers for each town
        for i in 0..NUM_TOWNS {
            let town = [problem.towns[[i, 0]], problem.towns[[i, 1]]];
            let mut min_d_sq = f64::MAX;
            for c in &centers {
                let d_sq = (town[0] - c[0]).powi(2) + (town[1] - c[1]).powi(2);
                if d_sq < min_d_sq {
                    min_d_sq = d_sq;
                }
            }
            // Weight by both distance and town weight
            distances[i] = min_d_sq * problem.weights[i];
        }
        
        // Pick next center with probability proportional to weighted distance
        let total_dist: f64 = distances.iter().sum();
        let mut r = rng.gen::<f64>() * total_dist;
        let mut next_idx = 0;
        
        for i in 0..NUM_TOWNS {
            r -= distances[i];
            if r <= 0.0 {
                next_idx = i;
                break;
            }
        }
        
        let next_center = [problem.towns[[next_idx, 0]], problem.towns[[next_idx, 1]]];
        
        // Check min separation
        let mut ok = true;
        for c in &centers {
            if distance(&next_center, c) < MIN_SEP {
                ok = false;
                break;
            }
        }
        
        if ok {
            centers.push(next_center);
        } else {
            // Fallback: find nearest valid position
            for _ in 0..100 {
                let offset_x = rng.gen::<f64>() * 6.0 - 3.0;
                let offset_y = rng.gen::<f64>() * 6.0 - 3.0;
                let pos = [
                    clip(next_center[0] + offset_x, 0.0, (GRID_N - 1) as f64),
                    clip(next_center[1] + offset_y, 0.0, (GRID_N - 1) as f64),
                ];
                let mut valid = true;
                for c in &centers {
                    if distance(&pos, c) < MIN_SEP {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    centers.push(pos);
                    break;
                }
            }
            if centers.len() < NUM_ORBS {
                // Ultimate fallback
                centers.push([
                    rng.gen::<f64>() * (GRID_N - 1) as f64,
                    rng.gen::<f64>() * (GRID_N - 1) as f64,
                ]);
            }
        }
    }
    
    let mut orbs = Array2::<f64>::zeros((NUM_ORBS, 2));
    for i in 0..NUM_ORBS {
        orbs[[i, 0]] = centers[i][0];
        orbs[[i, 1]] = centers[i][1];
    }
    orbs
}

// Grid-based systematic placement
fn grid_systematic_seed<R: Rng>(rng: &mut R) -> Array2<f64> {
    // Create a systematic grid with some randomness
    let mut positions = Vec::new();
    
    // Define grid spacing
    let spacing = 5.5;
    let offset = 2.0;
    
    for i in 0..4 {
        for j in 0..4 {
            let x = offset + i as f64 * spacing + rng.gen::<f64>() * 1.5 - 0.75;
            let y = offset + j as f64 * spacing + rng.gen::<f64>() * 1.5 - 0.75;
            if x >= 0.0 && x <= (GRID_N - 1) as f64 && y >= 0.0 && y <= (GRID_N - 1) as f64 {
                positions.push([x, y]);
            }
        }
    }
    
    // Shuffle and pick best NUM_ORBS positions
    positions.shuffle(rng);
    
    let mut orbs = Array2::<f64>::zeros((NUM_ORBS, 2));
    let mut selected = Vec::new();
    
    for pos in positions.iter() {
        if selected.len() >= NUM_ORBS {
            break;
        }
        
        let mut ok = true;
        for s in &selected {
            if distance(pos, s) < MIN_SEP {
                ok = false;
                break;
            }
        }
        
        if ok {
            selected.push(*pos);
        }
    }
    
    // Fill remaining if needed
    while selected.len() < NUM_ORBS {
        let pos = [
            rng.gen::<f64>() * (GRID_N - 1) as f64,
            rng.gen::<f64>() * (GRID_N - 1) as f64,
        ];
        let mut ok = true;
        for s in &selected {
            if distance(&pos, s) < MIN_SEP {
                ok = false;
                break;
            }
        }
        if ok {
            selected.push(pos);
        }
    }
    
    for i in 0..NUM_ORBS {
        orbs[[i, 0]] = selected[i][0];
        orbs[[i, 1]] = selected[i][1];
    }
    orbs
}

// Spiral placement for good coverage
fn spiral_seed<R: Rng>(rng: &mut R) -> Array2<f64> {
    let center_x = (GRID_N as f64 - 1.0) / 2.0;
    let center_y = (GRID_N as f64 - 1.0) / 2.0;
    
    let mut orbs = Array2::<f64>::zeros((NUM_ORBS, 2));
    let mut placed = Vec::new();
    
    for i in 0..NUM_ORBS {
        let angle = i as f64 * 2.0 * std::f64::consts::PI / NUM_ORBS as f64;
        let radius = 5.0 + i as f64 * 0.8 + rng.gen::<f64>() * 2.0;
        
        let mut x = center_x + radius * angle.cos();
        let mut y = center_y + radius * angle.sin();
        
        // Clip to bounds
        x = clip(x, 0.0, (GRID_N - 1) as f64);
        y = clip(y, 0.0, (GRID_N - 1) as f64);
        
        // Project to maintain min separation
        let pos = project_minsep(&orbs, i, [x, y], MIN_SEP);
        orbs[[i, 0]] = pos[0];
        orbs[[i, 1]] = pos[1];
        placed.push(pos);
    }
    
    orbs
}

// Weight density clustering - place orbs in high-weight regions
fn weight_density_seed<R: Rng>(problem: &Problem, rng: &mut R) -> Array2<f64> {
    // Find high-weight clusters
    let mut weight_map = Array2::<f64>::zeros((GRID_N, GRID_N));
    for i in 0..NUM_TOWNS {
        let x = problem.towns[[i, 0]] as usize;
        let y = problem.towns[[i, 1]] as usize;
        weight_map[[x, y]] = problem.weights[i];
    }
    
    // Apply Gaussian blur to find density clusters
    let mut smooth_weights = weight_map.clone();
    for x in 1..GRID_N-1 {
        for y in 1..GRID_N-1 {
            let mut sum = weight_map[[x, y]] * 4.0;
            sum += weight_map[[x-1, y]] * 2.0;
            sum += weight_map[[x+1, y]] * 2.0;
            sum += weight_map[[x, y-1]] * 2.0;
            sum += weight_map[[x, y+1]] * 2.0;
            sum += weight_map[[x-1, y-1]];
            sum += weight_map[[x+1, y-1]];
            sum += weight_map[[x-1, y+1]];
            sum += weight_map[[x+1, y+1]];
            smooth_weights[[x, y]] = sum / 16.0;
        }
    }
    
    // Find top density regions
    let mut candidates = Vec::new();
    for x in 0..GRID_N {
        for y in 0..GRID_N {
            candidates.push((smooth_weights[[x, y]], [x as f64, y as f64]));
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    // Place orbs in high-density regions with separation
    let mut orbs = Array2::<f64>::zeros((NUM_ORBS, 2));
    let mut placed = 0;
    
    for (_, pos) in candidates.iter() {
        if placed >= NUM_ORBS {
            break;
        }
        
        // Add small random offset
        let x = pos[0] + rng.gen::<f64>() * 0.5 - 0.25;
        let y = pos[1] + rng.gen::<f64>() * 0.5 - 0.25;
        let cand = [
            clip(x, 0.0, (GRID_N - 1) as f64),
            clip(y, 0.0, (GRID_N - 1) as f64),
        ];
        
        // Check separation
        let mut ok = true;
        for j in 0..placed {
            if distance(&cand, &[orbs[[j, 0]], orbs[[j, 1]]]) < MIN_SEP {
                ok = false;
                break;
            }
        }
        
        if ok {
            orbs[[placed, 0]] = cand[0];
            orbs[[placed, 1]] = cand[1];
            placed += 1;
        }
    }
    
    orbs
}

// Load best positions from results.json as seeds
fn historical_best_seed(idx: usize) -> Option<Array2<f64>> {
    let results = ResultsFile::load();
    if idx >= results.high_scores.len() {
        return None;
    }
    
    let score = &results.high_scores[idx];
    let mut orbs = Array2::<f64>::zeros((NUM_ORBS, 2));
    for i in 0..NUM_ORBS.min(score.coordinates.len()) {
        orbs[[i, 0]] = score.coordinates[i].0;
        orbs[[i, 1]] = score.coordinates[i].1;
    }
    Some(orbs)
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

// Genetic crossover - combine two good solutions
fn crossover<R: Rng>(
    parent1: &Array2<f64>,
    parent2: &Array2<f64>,
    rng: &mut R,
) -> Array2<f64> {
    let mut child = Array2::<f64>::zeros((NUM_ORBS, 2));
    
    // Uniform crossover with repair
    for i in 0..NUM_ORBS {
        if rng.gen::<f64>() < 0.5 {
            child[[i, 0]] = parent1[[i, 0]];
            child[[i, 1]] = parent1[[i, 1]];
        } else {
            child[[i, 0]] = parent2[[i, 0]];
            child[[i, 1]] = parent2[[i, 1]];
        }
    }
    
    // Repair minimum separation violations
    for i in 0..NUM_ORBS {
        let pos = project_minsep(&child, i, [child[[i, 0]], child[[i, 1]]], MIN_SEP);
        child[[i, 0]] = pos[0];
        child[[i, 1]] = pos[1];
    }
    
    child
}

// Advanced perturbation with adaptive strength
fn adaptive_perturbation<R: Rng>(
    orbs: &Array2<f64>,
    rng: &mut R,
    strength: f64,
) -> Array2<f64> {
    let mut perturbed = orbs.clone();
    
    // Perturb 30-50% of orbs
    let num_to_perturb = rng.gen_range(3..=5);
    let mut indices: Vec<usize> = (0..NUM_ORBS).collect();
    indices.shuffle(rng);
    
    for &idx in indices.iter().take(num_to_perturb) {
        // Levy flight for better exploration
        let levy_step = if rng.gen::<f64>() < 0.1 {
            // Occasional large jump
            rng.gen::<f64>() * 4.0 - 2.0
        } else {
            // Normal perturbation
            let normal = Normal::new(0.0, strength).unwrap();
            rng.sample(normal)
        };
        
        perturbed[[idx, 0]] += levy_step;
        perturbed[[idx, 1]] += levy_step * (rng.gen::<f64>() * 2.0 - 1.0);
        
        let pos = [
            clip(perturbed[[idx, 0]], 0.0, (GRID_N - 1) as f64),
            clip(perturbed[[idx, 1]], 0.0, (GRID_N - 1) as f64),
        ];
        let proj = project_minsep(&perturbed, idx, pos, MIN_SEP);
        perturbed[[idx, 0]] = proj[0];
        perturbed[[idx, 1]] = proj[1];
    }
    
    perturbed
}

// Multi-orb coordinated optimization - jointly optimize 2 orbs
fn optimize_two_orbs<R: Rng>(problem: &Problem, orbs: &Array2<f64>, orb_i: usize, orb_j: usize, _rng: &mut R) -> (Array2<f64>, f64) {
    let mut best_orbs = orbs.clone();
    let mut best_val = problem.objective(&best_orbs);
    
    // Try small coordinated moves
    let deltas = [0.3, 0.2, 0.1, 0.05, 0.02];
    let dirs = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
    
    for &d1 in &deltas {
        for &d2 in &deltas {
            for dir1 in &dirs {
                for dir2 in &dirs {
                    let cand_i = [
                        clip(best_orbs[[orb_i, 0]] + d1 * dir1[0], 0.0, (GRID_N - 1) as f64),
                        clip(best_orbs[[orb_i, 1]] + d1 * dir1[1], 0.0, (GRID_N - 1) as f64),
                    ];
                    let cand_j = [
                        clip(best_orbs[[orb_j, 0]] + d2 * dir2[0], 0.0, (GRID_N - 1) as f64),
                        clip(best_orbs[[orb_j, 1]] + d2 * dir2[1], 0.0, (GRID_N - 1) as f64),
                    ];
                    
                    let mut cand_orbs = best_orbs.clone();
                    cand_orbs[[orb_i, 0]] = cand_i[0];
                    cand_orbs[[orb_i, 1]] = cand_i[1];
                    cand_orbs[[orb_j, 0]] = cand_j[0];
                    cand_orbs[[orb_j, 1]] = cand_j[1];
                    
                    // Check constraints
                    let proj_i = project_minsep(&cand_orbs, orb_i, cand_i, MIN_SEP);
                    cand_orbs[[orb_i, 0]] = proj_i[0];
                    cand_orbs[[orb_i, 1]] = proj_i[1];
                    
                    let proj_j = project_minsep(&cand_orbs, orb_j, cand_j, MIN_SEP);
                    cand_orbs[[orb_j, 0]] = proj_j[0];
                    cand_orbs[[orb_j, 1]] = proj_j[1];
                    
                    let val = problem.objective(&cand_orbs);
                    if val > best_val + 1e-9 {
                        best_orbs = cand_orbs;
                        best_val = val;
                    }
                }
            }
        }
    }
    
    (best_orbs, best_val)
}

// Swap two orbs positions
fn swap_orbs<R: Rng>(problem: &Problem, orbs: &Array2<f64>, rng: &mut R) -> (Array2<f64>, f64) {
    let best_orbs = orbs.clone();
    let best_val = problem.objective(&best_orbs);
    
    // Try swapping random pairs
    for _ in 0..5 {
        let i = rng.gen_range(0..NUM_ORBS);
        let j = rng.gen_range(0..NUM_ORBS);
        if i == j {
            continue;
        }
        
        let mut cand_orbs = orbs.clone();
        let temp_x = cand_orbs[[i, 0]];
        let temp_y = cand_orbs[[i, 1]];
        cand_orbs[[i, 0]] = cand_orbs[[j, 0]];
        cand_orbs[[i, 1]] = cand_orbs[[j, 1]];
        cand_orbs[[j, 0]] = temp_x;
        cand_orbs[[j, 1]] = temp_y;
        
        let val = problem.objective(&cand_orbs);
        if val > best_val + 1e-9 {
            return (cand_orbs, val);
        }
    }
    
    (best_orbs, best_val)
}

// Continuous refinement with very fine adjustments
fn continuous_refinement(problem: &Problem, orbs: &Array2<f64>) -> (Array2<f64>, f64) {
    let mut best_orbs = orbs.clone();
    let mut best_val = problem.objective(&best_orbs);
    
    // Ultra-fine adjustments
    let fine_deltas = [0.005, 0.002, 0.001, 0.0005];
    let dirs = [
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [0.707, 0.707], [0.707, -0.707], [-0.707, 0.707], [-0.707, -0.707],
    ];
    
    for orb_idx in 0..NUM_ORBS {
        for &delta in &fine_deltas {
            let mut improved = false;
            for dir in &dirs {
                let cand = [
                    clip(best_orbs[[orb_idx, 0]] + delta * dir[0], 0.0, (GRID_N - 1) as f64),
                    clip(best_orbs[[orb_idx, 1]] + delta * dir[1], 0.0, (GRID_N - 1) as f64),
                ];
                let cand_proj = project_minsep(&best_orbs, orb_idx, cand, MIN_SEP);
                
                let mut cand_orbs = best_orbs.clone();
                cand_orbs[[orb_idx, 0]] = cand_proj[0];
                cand_orbs[[orb_idx, 1]] = cand_proj[1];
                
                let val = problem.objective(&cand_orbs);
                if val > best_val + 1e-12 {  // Even smaller threshold
                    best_orbs = cand_orbs;
                    best_val = val;
                    improved = true;
                }
            }
            if !improved {
                break;  // No improvement at this scale, don't try finer
            }
        }
    }
    
    (best_orbs, best_val)
}

fn improve_once<R: Rng>(problem: &Problem, orbs: &Array2<f64>, rng: &mut R, iteration: usize, max_iterations: usize) -> (Array2<f64>, f64) {
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

        // Adaptive gradient descent with momentum
        let mut momentum_x = 0.0;
        let mut momentum_y = 0.0;
        let momentum_decay = 0.9;
        let mut learning_rate: f64 = 0.8;  // Increased initial learning rate
        
        for iter in 0..20 {  // More iterations
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

            let grad_norm = (grad_x * grad_x + grad_y * grad_y).sqrt();
            if grad_norm < 1e-12 {
                break;
            }
            
            // Adaptive learning rate based on gradient magnitude
            if grad_norm < 0.01 {
                learning_rate *= 0.8;  // Reduce when gradient is small (near optimum)
            } else if grad_norm > 1.0 {
                learning_rate *= 1.1;  // Increase when gradient is large (far from optimum)
                learning_rate = learning_rate.min(1.5_f64);  // Cap at 1.5
            }
            
            // Apply momentum
            momentum_x = momentum_decay * momentum_x - learning_rate * grad_x;
            momentum_y = momentum_decay * momentum_y - learning_rate * grad_y;
            
            // Decay learning rate over iterations
            if iter > 10 {
                learning_rate *= 0.95;
            }

            let mut step_x = momentum_x;
            let mut step_y = momentum_y;
            step_x = clip(step_x, -3.0, 3.0);  // Wider clip range
            step_y = clip(step_y, -3.0, 3.0);

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
    // Use larger steps first for faster convergence
    let deltas = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];
    let dirs = [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.707, 0.707],
        [0.707, -0.707],
        [-0.707, 0.707],
        [-0.707, -0.707],
    ];

    // Coordinate descent with early stopping per orb
    for i in 0..NUM_ORBS {
        let mut local_improvements = 0;
        for &dval in &deltas {
            let mut improved_at_scale = false;
            
            for dir in &dirs {
                let cand = [
                    clip(best_orbs[[i, 0]] + dval * dir[0], 0.0, (GRID_N - 1) as f64),
                    clip(best_orbs[[i, 1]] + dval * dir[1], 0.0, (GRID_N - 1) as f64),
                ];
                let cand_proj = project_minsep(&best_orbs, i, cand, MIN_SEP);

                // Skip if projection didn't move us
                if (cand_proj[0] - best_orbs[[i, 0]]).abs() < 1e-9 && 
                   (cand_proj[1] - best_orbs[[i, 1]]).abs() < 1e-9 {
                    continue;
                }

                let mut cand_orbs = best_orbs.clone();
                cand_orbs[[i, 0]] = cand_proj[0];
                cand_orbs[[i, 1]] = cand_proj[1];

                let val = problem.objective(&cand_orbs);
                if val > best_val + 1e-9 {
                    best_orbs = cand_orbs;
                    best_val = val;
                    improved_at_scale = true;
                    local_improvements += 1;
                }
            }
            
            // If we made improvements at this scale, restart from largest scale
            if improved_at_scale && local_improvements < 3 {
                break;
            }
            
            // Early stop if no improvements for several scales
            if !improved_at_scale && local_improvements == 0 && dval < 0.1 {
                break;
            }
        }
    }

    // Multi-orb shake with temperature-scheduled simulated annealing
    let temperature = 1.0 - (iteration as f64 / max_iterations as f64);  // Cool down over time
    let acceptance_prob = 0.3 * temperature;  // More accepting early, less later
    
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
        // Temperature-scheduled acceptance: more accepting early, less later
        let energy_diff = best_val - val;
        let accept = val > best_val || (energy_diff < 2.0 && rng.gen::<f64>() < acceptance_prob);
        if accept {
            best_orbs = cand_orbs;
            best_val = val;
        }
    }
    
    // NEW: Multi-orb coordinated optimization (20% chance)
    if rng.gen::<f64>() < 0.2 {
        let orb_i = rng.gen_range(0..NUM_ORBS);
        let orb_j = rng.gen_range(0..NUM_ORBS);
        if orb_i != orb_j {
            let (new_orbs, new_val) = optimize_two_orbs(problem, &best_orbs, orb_i, orb_j, rng);
            if new_val > best_val {
                best_orbs = new_orbs;
                best_val = new_val;
            }
        }
    }
    
    // NEW: Try swapping orbs (10% chance)
    if rng.gen::<f64>() < 0.1 {
        let (new_orbs, new_val) = swap_orbs(problem, &best_orbs, rng);
        if new_val > best_val {
            best_orbs = new_orbs;
            best_val = new_val;
        }
    }
    
    // NEW: Continuous refinement in later iterations (when close to optimum)
    if iteration > max_iterations / 2 {
        let (refined_orbs, refined_val) = continuous_refinement(problem, &best_orbs);
        if refined_val > best_val {
            best_orbs = refined_orbs;
            best_val = refined_val;
        }
    }

    (best_orbs, best_val)
}

// ============================
// Results Tracking
// ============================

#[derive(Serialize, Deserialize, Debug, Clone)]
struct HighScore {
    score: f64,
    coordinates: Vec<(f64, f64)>,
    config: String,
    seed: u64,
    timestamp: DateTime<Utc>,
    iterations_used: usize,
    beam_size: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct ResultsFile {
    high_scores: Vec<HighScore>,
}

impl ResultsFile {
    fn load() -> Self {
        if Path::new("results.json").exists() {
            let content = fs::read_to_string("results.json").unwrap_or_else(|_| "{}".to_string());
            serde_json::from_str(&content).unwrap_or_else(|_| ResultsFile { high_scores: Vec::new() })
        } else {
            ResultsFile { high_scores: Vec::new() }
        }
    }

    fn add_score(&mut self, score: HighScore) {
        self.high_scores.push(score);
        // Sort by score descending
        self.high_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    }

    fn save(&self) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        fs::write("results.json", json)?;
        Ok(())
    }
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

    // Generate seeds with diverse strategies
    let mut seeds = Vec::new();
    
    // 1. Historical best positions - DISABLED to prevent convergence
    // Historical seeds were causing repeated 3200.496 scores
    // Uncomment below to re-enable with strong perturbation
    /*
    if rng.gen::<f64>() < 0.3 {  // Only 30% chance to use historical
        if let Some(hist_seed) = historical_best_seed(0) {
            // Add strongly jittered version (not the exact historical)
            let mut jittered = hist_seed;
            let normal = Normal::new(0.0, 0.8).unwrap();  // Much stronger jitter
            
            // Perturb at least 3-5 orbs significantly
            let orbs_to_perturb = rng.gen_range(3..=5);
            let mut perturb_indices: Vec<usize> = (0..NUM_ORBS).collect();
            perturb_indices.shuffle(&mut rng);
            
            for &idx in perturb_indices.iter().take(orbs_to_perturb) {
                jittered[[idx, 0]] += rng.sample(normal) * 2.0;  // Stronger perturbation
                jittered[[idx, 1]] += rng.sample(normal) * 2.0;
                let pos = [
                    clip(jittered[[idx, 0]], 0.0, (GRID_N - 1) as f64),
                    clip(jittered[[idx, 1]], 0.0, (GRID_N - 1) as f64),
                ];
                let proj = project_minsep(&jittered, idx, pos, MIN_SEP);
                jittered[[idx, 0]] = proj[0];
                jittered[[idx, 1]] = proj[1];
            }
            seeds.push(jittered);
        }
    }
    */
    
    // 2. K-means++ for optimal coverage (25% of seeds) - INCREASED
    let kmeans_count = (seed_beam as f64 * 0.25).ceil() as usize;
    for _ in 0..kmeans_count {
        seeds.push(kmeans_plus_plus_seed(problem, &mut rng));
    }
    
    // 3. Weight density clustering (20% of seeds) - INCREASED
    let density_count = (seed_beam as f64 * 0.20).ceil() as usize;
    for _ in 0..density_count {
        seeds.push(weight_density_seed(problem, &mut rng));
    }
    
    // 4. Grid systematic placement (15% of seeds) - INCREASED with variations
    let grid_count = (seed_beam as f64 * 0.15).ceil() as usize;
    for _ in 0..grid_count {
        seeds.push(grid_systematic_seed(&mut rng));
    }
    
    // 5. Spiral placement (10% of seeds) - INCREASED
    let spiral_count = (seed_beam as f64 * 0.10).ceil() as usize;
    for _ in 0..spiral_count {
        seeds.push(spiral_seed(&mut rng));
    }

    // 6. Edge-biased seed and variations (10% of seeds)
    seeds.push(edge_biased_seed());
    
    // Jittered edge-biased seeds with varying perturbation levels
    let edge_variations = (seed_beam as f64 * 0.1).ceil() as usize;
    for i in 0..edge_variations.saturating_sub(1) {
        let mut s = edge_biased_seed();
        // Vary the perturbation strength
        let jitter_strength = 0.2 + (i as f64 * 0.15 / edge_variations as f64);
        let normal = Normal::new(0.0, jitter_strength).unwrap();
        for j in 0..NUM_ORBS {
            let jx = rng.sample(normal);
            let jy = rng.sample(normal);
            s[[j, 0]] += jx;
            s[[j, 1]] += jy;
            let cand = [
                clip(s[[j, 0]], 0.0, (GRID_N - 1) as f64),
                clip(s[[j, 1]], 0.0, (GRID_N - 1) as f64),
            ];
            let proj = project_minsep(&s, j, cand, MIN_SEP);
            s[[j, 0]] = proj[0];
            s[[j, 1]] = proj[1];
        }
        seeds.push(s);
    }

    // 7. Weighted farthest seeds for remaining slots (20% of seeds) - More diverse
    while seeds.len() < seed_beam {
        // Vary the trial count for diversity
        let trials = 150 + rng.gen_range(0..150);
        seeds.push(weighted_farthest_seed(problem, &mut rng, trials));
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

    // Adaptive exploration strength
    let mut exploration_strength: f64 = 1.0;
    let mut stagnation_counter = 0;
    
    // Outer improvement loop with advanced strategies
    for iter in 0..max_outer {
        // Parallel improvement of beam members
        let new_results: Vec<(Array2<f64>, f64)> = beam
            .par_iter()
            .map(|r| {
                let mut local_rng = StdRng::seed_from_u64(random_seed + iter as u64 + 1);
                improve_once(problem, &r.orbs, &mut local_rng, iter, max_outer)
            })
            .collect();

        let mut new_beam = Vec::new();
        for (orbs, val) in new_results {
            new_beam.push(Result { orbs, value: val });
        }
        
        // Genetic crossover - combine top solutions
        if beam.len() >= 2 && iter % 3 == 0 {  // Every 3rd iteration
            let num_crossovers = (seed_beam / 4).max(2);
            for _ in 0..num_crossovers {
                let parent1_idx = rng.gen_range(0..beam.len().min(5));  // Top 5
                let parent2_idx = rng.gen_range(0..beam.len().min(5));
                if parent1_idx != parent2_idx {
                    let child = crossover(&beam[parent1_idx].orbs, &beam[parent2_idx].orbs, &mut rng);
                    let child_val = problem.objective(&child);
                    new_beam.push(Result { orbs: child, value: child_val });
                }
            }
        }
        
        // Adaptive perturbation for exploration
        if stagnation_counter > 3 {
            exploration_strength *= 1.2;  // Increase exploration when stuck
            exploration_strength = exploration_strength.min(3.0);
            
            // Add strongly perturbed versions of top solutions
            for i in 0..beam.len().min(3) {
                let perturbed = adaptive_perturbation(&beam[i].orbs, &mut rng, exploration_strength);
                let perturbed_val = problem.objective(&perturbed);
                new_beam.push(Result { orbs: perturbed, value: perturbed_val });
            }
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
            let improvement = beam[0].value - best_so_far;
            best_so_far = beam[0].value;
            no_improve_count = 0;
            stagnation_counter = 0;
            exploration_strength = 1.0;  // Reset exploration
            log_info!("  Iter {}: {:.6} ⬆️ IMPROVED (+{:.6})", iter + 1, beam[0].value, improvement);
        } else {
            no_improve_count += 1;
            stagnation_counter += 1;
            
            // Show exploration status
            if stagnation_counter > 3 {
                log_info!("  Iter {}: {:.6} (exploring, strength: {:.2})", iter + 1, beam[0].value, exploration_strength);
            } else {
                log_info!("  Iter {}: {:.6} (no improve: {})", iter + 1, beam[0].value, no_improve_count);
            }
        }

        // Modified early stopping - allow more iterations during exploration
        let effective_patience = if exploration_strength > 2.0 {
            patience * 2  // Double patience during strong exploration
        } else {
            patience
        };
        
        if no_improve_count >= effective_patience {
            log_info!("✓ Early stopping at iteration {} (no improvement for {} iterations)", 
                     iter + 1, effective_patience);
            break;
        }
    }

    // Multi-stage restart if stuck below threshold
    if best_so_far < 3200.50 && max_outer > 50 {
        log_info!("  🔄 Stage 2: Aggressive restart with mutations...");
        
        // Keep best solution and create mutations
        let best_solution = beam[0].clone();
        let mut restart_seeds = Vec::new();
        
        // Add heavily mutated versions of best
        for _ in 0..seed_beam / 2 {
            let mutated = adaptive_perturbation(&best_solution.orbs, &mut rng, 2.5);
            restart_seeds.push(mutated);
        }
        
        // Add completely new random seeds
        while restart_seeds.len() < seed_beam {
            restart_seeds.push(weighted_farthest_seed(problem, &mut rng, 250));
        }
        
        // Create new beam from restart seeds
        let mut restart_beam: Vec<Result> = restart_seeds
            .into_iter()
            .map(|s| {
                let val = problem.objective(&s);
                Result { orbs: s, value: val }
            })
            .collect();
        
        // Keep the original best
        restart_beam.push(best_solution);
        restart_beam.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
        restart_beam.truncate(seed_beam);
        
        // Run aggressive optimization for 20 more iterations
        for iter in 0..20 {
            let new_results: Vec<(Array2<f64>, f64)> = restart_beam
                .par_iter()
                .map(|r| {
                    let mut local_rng = StdRng::seed_from_u64(random_seed + 5000 + iter as u64);
                    improve_once(problem, &r.orbs, &mut local_rng, iter, 20)
                })
                .collect();
            
            let mut new_beam = Vec::new();
            for (orbs, val) in new_results {
                new_beam.push(Result { orbs, value: val });
            }
            
            // Aggressive crossover
            if restart_beam.len() >= 2 {
                for _ in 0..seed_beam / 3 {
                    let p1 = rng.gen_range(0..restart_beam.len());
                    let p2 = rng.gen_range(0..restart_beam.len());
                    if p1 != p2 {
                        let child = crossover(&restart_beam[p1].orbs, &restart_beam[p2].orbs, &mut rng);
                        let child_val = problem.objective(&child);
                        new_beam.push(Result { orbs: child, value: child_val });
                    }
                }
            }
            
            new_beam.extend(restart_beam.clone());
            new_beam.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
            
            // Aggressive deduplication
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
            
            restart_beam = unique;
            
            if restart_beam[0].value > best_so_far {
                best_so_far = restart_beam[0].value;
                log_info!("  Stage 2 Iter {}: {:.6} ⬆️ BREAKTHROUGH!", iter + 1, restart_beam[0].value);
            }
        }
        
        // Return best from restart
        if restart_beam[0].value > beam[0].value {
            return restart_beam.into_iter()
                .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap())
                .unwrap();
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
    
    // Special case: just view results
    if args.config == "view" {
        println!("\n=== HIGH SCORES LEADERBOARD (from results.json) ===");
        let results_file = ResultsFile::load();
        if results_file.high_scores.is_empty() {
            println!("No scores above 3200.5 recorded yet.");
        } else {
            for (i, score) in results_file.high_scores.iter().enumerate() {
                println!("\n#{}: Score: {:.6}", i + 1, score.score);
                println!("  Config: {} | Seed: {} | Beam: {}", 
                         score.config, score.seed, score.beam_size);
                println!("  Time: {}", score.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
                println!("  Coordinates: {:?}", score.coordinates);
            }
            println!("\n🏆 Best score: {:.6}", results_file.high_scores[0].score);
            println!("Total high scores recorded: {}", results_file.high_scores.len());
        }
        return;
    }
    
    log_info!("Orb Optimizer (Rust) — initializing...");
    
    // Load configuration
    let config_content = fs::read_to_string("config.toml")
        .expect("Failed to read config.toml");
    let config_file: ConfigFile = toml::from_str(&config_content)
        .expect("Failed to parse config.toml");
    
    let config = match args.config.as_str() {
        "local" => config_file.local,
        "local_max" => config_file.local_max,
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
    
    log_info!("Configuration: {} - {}", args.config, config.description);
    log_info!("  Beam size: {}", config.beam_size);
    log_info!("  Seeds: {}", config.seeds);
    log_info!("  Parallel chunks: {}", config.parallel_chunks);
    log_info!("  Patience: {}", config.patience);
    log_info!("  Max iterations: {}\n", config.iterations);

    let problem = Problem::new();
    log_info!("Problem setup complete: {} towns, {} orbs", NUM_TOWNS, NUM_ORBS);
    log_info!("Target: 3200.51+\n");

    // Generate seeds based on config
    let seeds_to_try: Vec<u64> = (1..=config.seeds as u64).map(|i| i * 1234 + i * i).collect();
    
    log_info!("Processing {} seeds in parallel ({} at a time)...\n", seeds_to_try.len(), config.parallel_chunks);
    log_info!("Starting optimization... (this may take a while with large beam sizes)");
    log_info!("Looking for scores > 3200.51...\n");
    
    // Process seeds in parallel chunks
    let chunk_size = config.parallel_chunks;
    let mut best_overall: Option<Result> = None;
    let mut seeds_processed = 0;
    let start_time = Instant::now();
    
    for chunk in seeds_to_try.chunks(chunk_size) {
        let chunk_results: Vec<(usize, Result)> = chunk
            .par_iter()
            .enumerate()
            .map(|(chunk_idx, &seed)| {
                let seed_start = Instant::now();
                let result = search(&problem, config.iterations, config.beam_size, seed, config.patience);
                let seed_duration = seed_start.elapsed();
                let global_idx = seeds_processed + chunk_idx;
                log_info!("  Seed {}/{} ({}): {:.6} | Time: {:.1}s", 
                         global_idx + 1, config.seeds, seed, result.value, seed_duration.as_secs_f32());
                
                // Save and print high scores (> 3200.5)
                if result.value > 3200.5 {
                    // Prepare coordinates in [(x,y), ...] format
                    let coords: Vec<(f64, f64)> = (0..NUM_ORBS)
                        .map(|i| (result.orbs[[i, 0]], result.orbs[[i, 1]]))
                        .collect();
                    
                    // Create high score entry
                    let high_score = HighScore {
                        score: result.value,
                        coordinates: coords.clone(),
                        config: args.config.clone(),
                        seed,
                        timestamp: Utc::now(),
                        iterations_used: 0, // Will be updated if we track this
                        beam_size: config.beam_size,
                    };
                    
                    // Load, add, and save
                    let mut results_file = ResultsFile::load();
                    results_file.add_score(high_score);
                    if let Err(e) = results_file.save() {
                        eprintln!("Failed to save to results.json: {}", e);
                    } else {
                        log_info!("    💾 Saved to results.json!");
                    }
                    
                    // Print for scores > 3200.51
                    if result.value > 3200.51 {
                        log_info!("    🎯 BREAKTHROUGH! Score: {:.6}", result.value);
                        log_info!("    Coordinates: {:?}", coords);
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
                log_info!(">>> 🔥 NEW BEST: {:.6} <<<", value);
                if value > 3200.50 && value <= 3200.51 {
                    log_info!("    ⚠️  So close! Only {:.6} away from target!", 3200.51 - value);
                }
                best_overall = Some(result);
                
                // Early exit if we hit the target
                if value > 3200.51 {
                    log_info!("\n🎯 TARGET REACHED! Score: {:.6}", value);
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
        
        // Progress update every 100 seeds (or every 10 for test mode, every 50 for large beams)
        let update_freq = if config.seeds <= 100 { 
            10 
        } else if config.beam_size >= 500 { 
            50 
        } else { 
            100 
        };
        
        if seeds_processed % update_freq == 0 && seeds_processed > 0 {
            let elapsed = start_time.elapsed();
            let rate = seeds_processed as f32 / elapsed.as_secs_f32();
            let eta = ((config.seeds - seeds_processed) as f32 / rate) as u64;
            
            if let Some(ref best) = best_overall {
                log_info!("\n=== PROGRESS UPDATE ===");
                log_info!("  Seeds: {}/{} ({:.1}%)", seeds_processed, config.seeds, 
                         (seeds_processed as f32 / config.seeds as f32) * 100.0);
                log_info!("  Current best: {:.6}", best.value);
                log_info!("  Time elapsed: {}m {}s", elapsed.as_secs() / 60, elapsed.as_secs() % 60);
                log_info!("  Speed: {:.2} seeds/sec", rate);
                log_info!("  ETA: ~{}m {}s", eta / 60, eta % 60);
                if best.value > 3200.5 {
                    log_info!("  Distance to target: {:.6}", 3200.51 - best.value);
                }
                log_info!("");
            }
        }
    }

    let result = best_overall.unwrap();

    log_info!("\n=== FINAL RESULT ===");
    log_info!("Best objective value: {:.6}", result.value);
    if result.value > 3200.51 {
        log_info!("✅ SUCCESS! Score exceeds 3200.51");
    } else {
        log_info!("⚠️  Score below target (3200.51)");
    }
    log_info!("\nOrb coordinates (x, y):");
    for i in 0..NUM_ORBS {
        log_info!("({:.6}, {:.6})", result.orbs[[i, 0]], result.orbs[[i, 1]]);
    }
    
    // Display current high scores from results.json
    log_info!("\n=== HIGH SCORES LEADERBOARD (from results.json) ===");
    let results_file = ResultsFile::load();
    if results_file.high_scores.is_empty() {
        log_info!("No scores above 3200.5 recorded yet.");
    } else {
        for (i, score) in results_file.high_scores.iter().take(10).enumerate() {
            log_info!("#{}: {:.6} | Config: {} | Seed: {} | {}", 
                     i + 1, score.score, score.config, score.seed, 
                     score.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
            if i == 0 {
                // Show coordinates for the best score
                log_info!("  Best coordinates: {:?}", score.coordinates);
            }
        }
        log_info!("\nTotal high scores recorded: {}", results_file.high_scores.len());
    }
}


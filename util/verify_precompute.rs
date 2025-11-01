use std::f64::consts::E;

const GRID_N: usize = 20;
const S_CORNER: f64 = 0.098765432;
const S_BOUND: f64 = 0.135792468;
const S_INTER: f64 = 0.246913579;

fn population(x: f64, y: f64) -> f64 {
    50.0 + (20.0 * (0.35 * x).sin()
        + 15.0 * (0.30 * y).cos()
        + 10.0 * (0.20 * (x + y)).sin())
    .floor()
}

fn selfie_rate(x: usize, y: usize) -> f64 {
    let is_corner = (x == 0 && y == 0)
        || (x == 0 && y == GRID_N - 1)
        || (x == GRID_N - 1 && y == 0)
        || (x == GRID_N - 1 && y == GRID_N - 1);
    if is_corner { return S_CORNER; }
    let is_edge = x == 0 || x == GRID_N - 1 || y == 0 || y == GRID_N - 1;
    if is_edge { S_BOUND } else { S_INTER }
}

fn alpha(d: f64) -> f64 {
    0.5 * E.powf(-0.40 * d)
}

fn main() {
    let orbs = vec![
        (6.999999999999999, 7.0),
        (17.0, 17.0),
        (2.0, 1.0),
        (2.000114, 6.000214000000002),
        (8.000008000000001, 17.000078000000002),
        (11.0, 2.0),
        (6.0, 2.0),
        (2.9140015, 17.071712499999997),
        (17.32264310156189, 2.766113499999999),
        (3.0001863463924625, 11.999990643529175),
    ];

    println!("=== METHOD 1: Direct calculation (like Python) ===");
    let mut total1 = 0.0;
    for x in 0..GRID_N {
        for y in 0..GRID_N {
            let fx = x as f64;
            let fy = y as f64;
            let pop = population(fx, fy);
            let selfie = selfie_rate(x, y);
            let weight = pop * (1.0 - selfie);
            
            let mut min_dist = f64::MAX;
            for &(ox, oy) in &orbs {
                let dx = fx - ox;
                let dy = fy - oy;
                let dist = (dx * dx + dy * dy).sqrt();
                min_dist = min_dist.min(dist);
            }
            total1 += weight * alpha(min_dist);
        }
    }
    
    println!("=== METHOD 2: Pre-computed weights (like current Rust code) ===");
    // Pre-compute weights in the same order as Rust code
    let mut weights = vec![0.0; 400];
    let mut towns_flat = Vec::new();
    let mut idx = 0;
    for y in 0..GRID_N {
        for x in 0..GRID_N {
            let fx = x as f64;
            let fy = y as f64;
            towns_flat.push((fx, fy));
            let p = population(fx, fy);
            let s = selfie_rate(x, y);
            weights[idx] = p * (1.0 - s);
            idx += 1;
        }
    }
    
    let mut total2 = 0.0;
    for i in 0..400 {
        let (tx, ty) = towns_flat[i];
        let mut min_dist = f64::MAX;
        for &(ox, oy) in &orbs {
            let dx = tx - ox;
            let dy = ty - oy;
            let dist = (dx * dx + dy * dy).sqrt();
            min_dist = min_dist.min(dist);
        }
        total2 += weights[i] * alpha(min_dist);
    }

    println!("\nMethod 1 (direct):       {:.10}", total1);
    println!("Method 2 (pre-computed): {:.10}", total2);
    println!("Challenge server:        3200.5022521908063");
    println!("\nDifference (M1 - M2):    {:.15}", (total1 - total2).abs());
    println!("Difference (M1 - server): {:.15}", (total1 - 3200.5022521908063).abs());
    println!("Difference (M2 - server): {:.15}", (total2 - 3200.5022521908063).abs());
}

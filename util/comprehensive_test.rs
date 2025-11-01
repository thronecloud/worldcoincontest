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
    let is_corner = (x == 0 && y == 0) || (x == 0 && y == GRID_N - 1) || (x == GRID_N - 1 && y == 0) || (x == GRID_N - 1 && y == GRID_N - 1);
    if is_corner { return S_CORNER; }
    let on_boundary = x == 0 || y == 0 || x == GRID_N - 1 || y == GRID_N - 1;
    if on_boundary { S_BOUND } else { S_INTER }
}

fn alpha(d: f64) -> f64 {
    0.5 * E.powf(-0.40 * d)
}

fn main() {
    let orbs = vec![
        (6.999999999999999, 7.0), (17.0, 17.0), (2.0, 1.0), (2.000114, 6.000214000000002),
        (8.000008000000001, 17.000078000000002), (11.0, 2.0), (6.0, 2.0), (2.9140015, 17.071712499999997),
        (17.32264310156189, 2.766113499999999), (3.0001863463924625, 11.999990643529175),
    ];

    println!("=== COMPREHENSIVE VALIDATION ===\n");
    
    // Test 1: Verify population function
    println!("Test 1: Population at (0,0)");
    let p00 = population(0.0, 0.0);
    println!("  Calculated: {}", p00);
    println!("  Expected: 65.0");
    assert!((p00 - 65.0).abs() < 1e-10, "Population(0,0) mismatch!");
    println!("  ✅ PASS\n");
    
    // Test 2: Verify selfie rates
    println!("Test 2: Selfie rates");
    assert_eq!(selfie_rate(0, 0), S_CORNER, "Corner selfie rate wrong!");
    println!("  Corner (0,0): {} ✅", selfie_rate(0, 0));
    assert_eq!(selfie_rate(0, 1), S_BOUND, "Boundary selfie rate wrong!");
    println!("  Boundary (0,1): {} ✅", selfie_rate(0, 1));
    assert_eq!(selfie_rate(1, 1), S_INTER, "Interior selfie rate wrong!");
    println!("  Interior (1,1): {} ✅\n", selfie_rate(1, 1));
    
    // Test 3: Verify alpha function
    println!("Test 3: Alpha function");
    let a0 = alpha(0.0);
    let expected_a0 = 0.5;
    println!("  α(0) = {} (expected: {})", a0, expected_a0);
    assert!((a0 - expected_a0).abs() < 1e-10, "Alpha(0) mismatch!");
    let a1 = alpha(1.0);
    let expected_a1 = 0.5 * E.powf(-0.4);
    println!("  α(1) = {} (expected: {})", a1, expected_a1);
    assert!((a1 - expected_a1).abs() < 1e-10, "Alpha(1) mismatch!");
    println!("  ✅ PASS\n");
    
    // Test 4: Full score calculation
    println!("Test 4: Full score calculation");
    let mut total = 0.0;
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
            total += weight * alpha(min_dist);
        }
    }
    
    println!("  Calculated: {:.10}", total);
    println!("  Server:     3200.5022521908063");
    let diff = (total - 3200.5022521908063).abs();
    println!("  Difference: {:.15}", diff);
    
    if diff < 1e-9 {
        println!("  ✅ EXACT MATCH!\n");
    } else {
        println!("  ❌ MISMATCH - ERROR!\n");
        panic!("Score mismatch!");
    }
    
    println!("=== ALL TESTS PASSED ===");
    println!("✅ No approximations");
    println!("✅ No lookup tables");
    println!("✅ Exact formula match");
    println!("✅ Score matches server exactly");
}

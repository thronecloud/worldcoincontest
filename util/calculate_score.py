#!/usr/bin/env python3
"""
Calculate the objective score for a given set of orb coordinates.
Matches the Rust implementation exactly.
"""

import math

# Constants
GRID_N = 20
NUM_ORBS = 10
S_CORNER = 0.098765432
S_BOUND = 0.135792468
S_INTER = 0.246913579

def population(x, y):
    """Calculate population at grid position (x, y)"""
    return 50.0 + math.floor(
        20.0 * math.sin(0.35 * x) +
        15.0 * math.cos(0.30 * y) +
        10.0 * math.sin(0.20 * (x + y))
    )

def selfie_rate(x, y):
    """Calculate selfie rate at grid position (x, y)"""
    is_corner = (
        (x == 0 and y == 0) or
        (x == 0 and y == GRID_N - 1) or
        (x == GRID_N - 1 and y == 0) or
        (x == GRID_N - 1 and y == GRID_N - 1)
    )
    
    if is_corner:
        return S_CORNER
    
    is_edge = (x == 0 or x == GRID_N - 1 or y == 0 or y == GRID_N - 1)
    
    if is_edge:
        return S_BOUND
    
    return S_INTER

def alpha(d):
    """Influence decay function"""
    return 0.5 * math.exp(-0.4 * d)

def calculate_score(orbs):
    """
    Calculate the objective score for given orb positions.
    
    Args:
        orbs: List of (x, y) tuples representing orb positions
    
    Returns:
        float: The objective score
    """
    if len(orbs) != NUM_ORBS:
        raise ValueError(f"Expected {NUM_ORBS} orbs, got {len(orbs)}")
    
    # Generate all towns
    towns = []
    for x in range(GRID_N):
        for y in range(GRID_N):
            towns.append((x, y))
    
    total_score = 0.0
    
    # For each town, find the contribution from the nearest orb
    for tx, ty in towns:
        pop = population(tx, ty)
        selfie = selfie_rate(tx, ty)
        
        # Find nearest orb and its distance
        min_dist = float('inf')
        for ox, oy in orbs:
            dx = tx - ox
            dy = ty - oy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
        
        # Calculate contribution
        # NOTE: Rust code uses (1.0 - selfie_rate) as the weight!
        influence = alpha(min_dist)
        contribution = pop * (1.0 - selfie) * influence
        total_score += contribution
    
    return total_score

def main():
    # Your orb coordinates - paste them here
    orbs = [
        (6.999999999999999, 7.0),
        (17.0, 17.0),
        (2.0, 1.0),
        (2.000114, 6.000214000000002),
        (8.000008000000001, 17.000078000000002),
        (11.0, 2.0),
        (6.0, 2.0),
        (2.9140015, 17.071712499999997),
        (17.32264310156189, 2.766113499999999),
        (3.0001863463924625, 11.999990643529175)
    ]
    
    score = calculate_score(orbs)
    
    print("=" * 60)
    print("ORB SCORE CALCULATOR")
    print("=" * 60)
    print(f"\nNumber of orbs: {len(orbs)}")
    print(f"\nOrb positions:")
    for i, (x, y) in enumerate(orbs):
        print(f"  Orb {i+1}: ({x:.6f}, {y:.6f})")
    
    print(f"\n{'=' * 60}")
    print(f"OBJECTIVE SCORE: {score:.10f}")
    print(f"{'=' * 60}")
    
    # Check against target
    if score > 3200.51:
        print(f"✅ SUCCESS! Score exceeds 3200.51 (+{score - 3200.51:.6f})")
    elif score > 3200.5:
        print(f"🔥 Very close! Only {3200.51 - score:.6f} away from target")
    elif score > 3200.0:
        print(f"⚠️  Close. Need {3200.51 - score:.6f} more to reach target")
    else:
        print(f"❌ Score below 3200")
    
    print()

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Verify exact score calculation matching challenge server.
Server reported: 3200.5022521908063
"""

import math

# Constants from https://tryworldchallenge.com/
GRID_N = 20
S_CORNER = 0.098765432
S_BOUND = 0.135792468
S_INTER = 0.246913579

def population(x, y):
    """P(x,y) = 50 + floor(20*sin(0.35x) + 15*cos(0.30y) + 10*sin(0.20(x+y)))"""
    return 50.0 + math.floor(
        20.0 * math.sin(0.35 * x) +
        15.0 * math.cos(0.30 * y) +
        10.0 * math.sin(0.20 * (x + y))
    )

def selfie_rate(x, y):
    """Selfie rate based on position"""
    # Corners
    is_corner = (
        (x == 0 and y == 0) or
        (x == 0 and y == GRID_N - 1) or
        (x == GRID_N - 1 and y == 0) or
        (x == GRID_N - 1 and y == GRID_N - 1)
    )
    if is_corner:
        return S_CORNER
    
    # Boundary (non-corner)
    is_edge = (x == 0 or x == GRID_N - 1 or y == 0 or y == GRID_N - 1)
    if is_edge:
        return S_BOUND
    
    # Interior
    return S_INTER

def alpha(d):
    """Verification probability: α(d) = 0.50 * e^(-0.40*d)"""
    return 0.5 * math.exp(-0.4 * d)

def calculate_score_detailed(orbs):
    """
    Calculate score with detailed breakdown.
    
    For each town:
      - Find nearest orb distance d
      - Calculate: population(x,y) × (1 - selfie_rate) × α(d)
      - Sum across all 400 towns
    """
    total_score = 0.0
    
    # Debug first few towns
    print("\nSample calculation for first 5 towns:")
    print("=" * 80)
    
    for town_idx, x in enumerate(range(GRID_N)):
        for y in range(GRID_N):
            # Get town properties
            pop = population(x, y)
            selfie = selfie_rate(x, y)
            participation = 1.0 - selfie
            
            # Find nearest orb
            min_dist = float('inf')
            for ox, oy in orbs:
                dx = x - ox
                dy = y - oy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist
            
            # Calculate contribution
            verification_prob = alpha(min_dist)
            contribution = pop * participation * verification_prob
            total_score += contribution
            
            # Print first 5 towns for verification
            if town_idx * GRID_N + y < 5:
                print(f"Town ({x},{y}):")
                print(f"  Population: {pop}")
                print(f"  Selfie rate: {selfie}")
                print(f"  Participation: {participation}")
                print(f"  Nearest orb distance: {min_dist:.10f}")
                print(f"  α(d): {verification_prob:.10f}")
                print(f"  Contribution: {contribution:.10f}")
                print()
    
    return total_score

# Exact coordinates from your submission
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

print("=" * 80)
print("VERIFYING SCORE CALCULATION")
print("=" * 80)
print(f"\nCoordinates: {len(orbs)} orbs")
print("\nFormula: Σ [P(x,y) × (1 - s_t) × α(d)]")
print("  where α(d) = 0.5 × e^(-0.4×d)")
print("  and d = distance to nearest orb")

calculated_score = calculate_score_detailed(orbs)

print("=" * 80)
print("FINAL RESULTS:")
print("=" * 80)
print(f"Calculated score:      {calculated_score:.10f}")
print(f"Challenge server said: 3200.5022521908063")
print(f"Difference:            {abs(calculated_score - 3200.5022521908063):.15f}")
print()

if abs(calculated_score - 3200.5022521908063) < 1e-8:
    print("✅ EXACT MATCH! Our calculator is correct!")
elif abs(calculated_score - 3200.5022521908063) < 0.001:
    print("✅ Very close match (within 0.001) - likely floating point precision")
else:
    print("❌ MISMATCH - need to investigate formula")


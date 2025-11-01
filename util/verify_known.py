#!/usr/bin/env python3
"""Verify known good result"""

import math

GRID_N = 20
S_CORNER = 0.098765432
S_BOUND = 0.135792468
S_INTER = 0.246913579

def population(x, y):
    return 50.0 + math.floor(
        20.0 * math.sin(0.35 * x) +
        15.0 * math.cos(0.30 * y) +
        10.0 * math.sin(0.20 * (x + y))
    )

def selfie_rate(x, y):
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
    return 0.5 * math.exp(-0.4 * d)

def calculate_score(orbs):
    total = 0.0
    for x in range(GRID_N):
        for y in range(GRID_N):
            pop = population(x, y)
            selfie = selfie_rate(x, y)
            
            min_dist = min(math.sqrt((x - ox)**2 + (y - oy)**2) for ox, oy in orbs)
            influence = alpha(min_dist)
            total += pop * (1.0 - selfie) * influence
    
    return total

# Verified correct result from Rust output
verified_coords = [
    (17.001472, 16.998226),
    (8.000470, 16.999980),
    (3.000123, 12.000090),
    (2.913759, 17.072100),
    (6.999753, 7.000062),
    (17.324389, 2.764656),
    (10.999791, 1.999886),
    (2.000000, 5.999768),
    (2.000717, 1.001000),
    (6.000865, 2.000000)
]

calculated = calculate_score(verified_coords)
reported = 3200.483872

print("Verifying known good result:")
print(f"Reported by Rust: {reported:.6f}")
print(f"Calculated in Python: {calculated:.6f}")
print(f"Difference: {abs(calculated - reported):.10f}")

if abs(calculated - reported) < 0.01:
    print("✅ MATCH! Formula is correct!")
else:
    print("❌ MISMATCH! Formula has an error!")


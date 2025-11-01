#!/usr/bin/env python3
"""Quick test of score calculation"""

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
            # NOTE: Rust code uses (1.0 - selfie_rate) as the weight!
            total += pop * (1.0 - selfie) * influence
    
    return total

# From your test run output - these should give ~3200.15
test_coords = [
    (6.000568, 2.000140),
    (17.000522, 16.999773),
    (2.013481, 1.010749),
    (11.000885, 2.000888),
    (7.010370, 6.990492),
    (2.913870, 17.071793),
    (3.000370, 11.999641),
    (17.326382, 2.767977),
    (2.000169, 5.999409),
    (8.004995, 16.983262)
]

# From results.json top entry
json_coords = [
    (2.9999999999999996, 12.0),
    (1.999895, 1.0000349999999999),
    (17.000110551834457, 16.999894396495033),
    (10.999956299074868, 1.9999908925663087),
    (2.9138515, 17.071853999999997),
    (2.000476615116102, 5.999926132149629),
    (17.32261885016851, 2.7665173066421547),
    (8.000174914634153, 16.999811037251334),
    (6.999946895267688, 7.000040137295131),
    (6.0, 2.0)
]

print("Testing score calculation...\n")
print(f"Test coords (from quick test run): {calculate_score(test_coords):.6f}")
print(f"  Expected: ~3200.152399")
print()
print(f"JSON coords (top from results.json): {calculate_score(json_coords):.6f}")
print(f"  Reported in JSON: 3200.502414")


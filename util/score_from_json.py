#!/usr/bin/env python3
"""
Calculate score from results.json or any coordinate list.
Usage: python score_from_json.py [results.json]
"""

import json
import sys
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
    """Calculate the objective score for given orb positions."""
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
    if len(sys.argv) < 2:
        filename = "results.json"
    else:
        filename = sys.argv[1]
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if 'high_scores' not in data or not data['high_scores']:
            print(f"No high scores found in {filename}")
            return
        
        print("=" * 70)
        print(f"VERIFYING SCORES FROM {filename}")
        print("=" * 70)
        
        for i, entry in enumerate(data['high_scores'][:10]):
            coords = entry['coordinates']
            reported_score = entry['score']
            seed = entry.get('seed', 'N/A')
            config = entry.get('config', 'N/A')
            
            # Convert to tuple list
            orbs = [(x, y) for x, y in coords]
            
            # Calculate actual score
            actual_score = calculate_score(orbs)
            
            diff = abs(actual_score - reported_score)
            
            print(f"\n#{i+1}:")
            print(f"  Config: {config} | Seed: {seed}")
            print(f"  Reported: {reported_score:.10f}")
            print(f"  Calculated: {actual_score:.10f}")
            print(f"  Difference: {diff:.10f}", end="")
            
            if diff < 1e-6:
                print(" ✅")
            elif diff < 0.01:
                print(" ⚠️ (small difference)")
            else:
                print(" ❌ (ERROR!)")
        
        print("\n" + "=" * 70)
        
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: {filename} is not valid JSON")
        sys.exit(1)

if __name__ == "__main__":
    main()


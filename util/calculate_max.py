import math

GRID_N = 20
S_CORNER = 0.098765432
S_BOUND = 0.135792468
S_INTER = 0.246913579

def population(x, y):
    return 50.0 + math.floor(20.0 * math.sin(0.35 * x) + 
                              15.0 * math.cos(0.30 * y) + 
                              10.0 * math.sin(0.20 * (x + y)))

def selfie_rate(x, y):
    is_corner = (x == 0 and y == 0) or (x == 0 and y == GRID_N - 1) or \
                (x == GRID_N - 1 and y == 0) or (x == GRID_N - 1 and y == GRID_N - 1)
    if is_corner:
        return S_CORNER
    on_boundary = x == 0 or y == 0 or x == GRID_N - 1 or y == GRID_N - 1
    return S_BOUND if on_boundary else S_INTER

# Calculate total weight
total_weight = 0
for y in range(GRID_N):
    for x in range(GRID_N):
        p = population(float(x), float(y))
        s = selfie_rate(x, y)
        weight = p * (1.0 - s)
        total_weight += weight

# Maximum if all distances = 0
theoretical_max = 0.5 * total_weight

print(f"Total weight (all towns): {total_weight:.2f}")
print(f"Theoretical maximum (all d=0): {theoretical_max:.2f}")
print(f"Current best achieved: 3200.49")
print(f"Efficiency: {(3200.49/theoretical_max)*100:.2f}%")

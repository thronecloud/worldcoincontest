# Orb Optimizer - Rust Edition

High-performance Rust implementation of the Orb Operator optimization algorithm.

## Features

- **Fast**: 20-100x faster than Python implementation
- **Parallel**: Uses Rayon for parallel beam search
- **Optimized**: Aggressive compiler optimizations (LTO, single codegen unit)
- **Memory efficient**: Stack-allocated arrays where possible

## Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))

## Building

```bash
# Build in release mode (required for performance)
cargo build --release

# The binary will be at: target/release/orb-optimizer
```

## Running

```bash
# Run directly
cargo run --release

# Or run the binary
./target/release/orb-optimizer
```

## Performance

On a typical modern CPU:
- Python version: ~30-60 seconds for 120 iterations
- Rust version: ~1-3 seconds for 120 iterations

**Expected speedup: 20-40x**

## Algorithm

Places 10 orbs on a 20x20 grid of towns to maximize weighted coverage:
- Uses beam search with multiple seeds
- Per-orb gradient descent with backtracking line search
- Hill-climbing with coordinate tweaks
- Occasional random perturbations to escape local optima
- Enforces minimum separation constraint between orbs

## Customization

Edit constants in `src/main.rs`:
- `GRID_N`: Grid size (default: 20)
- `NUM_ORBS`: Number of orbs (default: 10)
- `MIN_SEP`: Minimum separation between orbs (default: 2.5)

Adjust search parameters in `main()`:
- `max_outer`: Outer iterations (default: 120)
- `seed_beam`: Beam width (default: 8)
- `random_seed`: Random seed for reproducibility

## Dependencies

- `ndarray`: N-dimensional arrays
- `rand`: Random number generation
- `rand_distr`: Probability distributions
- `rayon`: Data parallelism


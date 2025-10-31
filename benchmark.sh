#!/bin/bash

echo "=== Performance Comparison ==="
echo ""

if [ -f "orb_operator_opt.py" ]; then
    echo "Python version:"
    time python3 orb_operator_opt.py 2>&1 | tail -3
    echo ""
fi

echo "Rust version:"
time cargo run --release --quiet 2>&1 | tail -3


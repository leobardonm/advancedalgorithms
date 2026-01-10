# Knapsack Optimization – Sales Trailer Problem

## Team Members
- Diego Vergara
- Leobardo Navarro
- Santiago Arista

## Requirements
- Python 3.10+
- matplotlib

## Installation
```bash
pip install matplotlib
```

## Running experiments
```bash
python greedy.py
```

## Outputs
All outputs are saved in the `results/` directory:
- `results_timings.csv` - Raw timing data for all runs
- `runtime_C1.png` / `.pdf` - Runtime vs Capacity plot for Catalog C1
- `runtime_C2.png` / `.pdf` - Runtime vs Capacity plot for Catalog C2
- `gap_C1.png` / `.pdf` - Greedy Value Gap plot for Catalog C1
- `gap_C2.png` / `.pdf` - Greedy Value Gap plot for Catalog C2

## Reproducibility
- All methods are deterministic
- No random seeds required (tie-breaking uses weight then ID)
- Each configuration is run **5 times**
- Median runtime is reported in milliseconds
- Tested capacities: [0, 20, 35, 50, 65, 80, 95, 110, 140]

## Methods Implemented
1. **Greedy** - Orders by value/weight ratio, O(n log n)
2. **Dynamic Programming** - 1D rolling array with reconstruction, O(nW)
3. **Backtracking with Branch-and-Bound** - Fractional upper bound pruning, O(2^n) worst case
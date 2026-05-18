#!/usr/bin/env python
"""Entry point for running an experiment.

Usage:
    python scripts/run_experiment.py --config configs/ppfg.yaml
    python scripts/run_experiment.py --config configs/default.yaml --method independent --n_problems 50
"""
from hyp_forest.runner import main

if __name__ == "__main__":
    main()

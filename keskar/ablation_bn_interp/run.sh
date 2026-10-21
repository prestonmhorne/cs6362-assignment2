#!/bin/bash
# Phase 4 diagnostic (after reading the authors' code): re-evaluate the saved Phase 3 SB/LB models
# with interpolated running BN stats instead of recomputed ones. No training; phase3.py unchanged.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
$PY ../tools/run_patched.py phase3.py interp --panels F1 C1 --c1_variant alexnet --bn_mode interp --out ablation_bn_interp --data_root ../data
$PY ../tools/run_patched.py phase3.py interp --panels C2 --bn_mode interp --out ablation_bn_interp --data_root ../data
$PY ../tools/run_patched.py phase3.py plot --panels F1 C1 C2 --c1_variant alexnet --out ablation_bn_interp
echo "ablation done $(date)"

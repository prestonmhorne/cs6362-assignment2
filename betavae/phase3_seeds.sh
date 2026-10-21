#!/bin/bash
# Phase 3 (beta-VAE): seeds 1-2 for beta=4 and beta=1 with the unchanged Phase 1 code, to separate
# seed variance from systematic gaps. Waits for the Keskar BN ablation so only one GPU job runs.
cd "$(dirname "$0")"
while pgrep -f "ablation_bn_interp" >/dev/null; do sleep 30; done
echo "start $(date)"
# seed 0 checkpoints are reloaded (not retrained), so the summary covers seeds 0-2
../.venv/bin/python phase1.py --data ../data/dsprites.npz --out phase1_runs --seed_start 0 --num_seeds 3
echo "done $(date)"

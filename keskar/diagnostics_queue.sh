#!/bin/bash
# Post-Phase-4 diagnostics for the F1/C1 LB-side height mismatch (2026-10-20).
# The LLM's phase3.py is unchanged; only command-line flags differ.
#  1. Fixed 20-epoch training (authors' released code), patience disabled.
#  2. F1 run-to-run variation: seeds 1 and 2 with the Phase 3 protocol.
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY ../tools/run_patched.py phase3.py all --panels F1 C1 --c1_variant alexnet --max_epochs 20 --patience 1000 --out diag_20ep --data_root ../data >> diag_20ep.log 2>&1
for s in 1 2; do
  $PY ../tools/run_patched.py phase3.py all --panels F1 --seed $s --out diag_F1_seed$s --data_root ../data >> diag_F1_seed$s.log 2>&1
done
echo "diagnostics finished $(date)" >> diag_done.log

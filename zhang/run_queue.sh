#!/bin/bash
# Run Zhang jobs one at a time (concurrent GPU jobs exhausted memory on 2026-10-19).
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY ../tools/run_patched.py phase1.py train --grid reduced --archs mlp1x512 alexnet --out phase1_runs --data_root ../data >> phase1_runs_mlp_alexnet.log 2>&1
$PY ../tools/run_patched.py phase3.py train --grid reduced --archs inception --out phase3_runs --data_root ../data >> phase3_runs_inception.log 2>&1

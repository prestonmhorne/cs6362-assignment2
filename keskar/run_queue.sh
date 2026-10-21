#!/bin/bash
# Keskar Phase 3 runs. F1 + C1 start now (small); C2 waits for the Zhang queue to finish
# so the two memory-heavy jobs never overlap.
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY ../tools/run_patched.py phase3.py all --panels F1 C1 --c1_variant both --out phase3_runs --data_root ../data >> phase3_runs_F1_C1.log 2>&1
while pgrep -f "zhang/run_queue.sh|run_queue.sh train|phase3.py train --grid reduced" >/dev/null; do sleep 60; done
$PY ../tools/run_patched.py phase3.py all --panels C2 --cap C2:sb=40 --cap C2:lb=40 --stop_eval_every 5 --out phase3_runs --data_root ../data >> phase3_runs_C2.log 2>&1
$PY ../tools/run_patched.py phase3.py plot --panels F1 C1 C2 --c1_variant both --out phase3_runs >> phase3_runs_C2.log 2>&1

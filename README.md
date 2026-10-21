# CS 6362 Assignment 2: reproducing three figures with an LLM

| Paper | Figure | Folder | Notebook |
|---|---|---|---|
| Zhang et al., *Understanding deep learning requires rethinking generalization* (ICLR 2017) | Fig. 1 (a, b, c) | `zhang/` | `zhang/reproduce_figure1.ipynb` |
| Keskar et al., *On large-batch training for deep learning* (ICLR 2017) | Fig. 3 (F1, C1, C2) | `keskar/` | `keskar/reproduce_figure3.ipynb` |
| Higgins et al., *β-VAE* (ICLR 2017) | Fig. 7 A, B | `betavae/` | `betavae/reproduce_figure7.ipynb` |

All reproduction code (`phase1.py`, `phase3*.py`) was written by **Claude Sonnet 5.5 (Medium effort) in the claude.ai chat
interface** and is saved unmodified. The conversations are in each folder's `llm_transcript.md`.

## Files in each folder
- `phase1_prompt.md`: the prompt given to the LLM together with the paper PDF.
- `phase1.py`: the LLM's first implementation (Phase 1).
- `phase2_review.md`: our review of the Phase 1 output and the evidence collected (Phases 2–3).
- `phase3*.py` (Zhang, Keskar): the corrected implementation, also written by the LLM in the same chat (Phase 3). For β-VAE, Phase 3 reruns the unchanged `phase1.py` with more seeds (`phase3_seeds.sh`).
- `phase4.md`: comparison with existing/author code (Phase 4).
- `report.md`: the report.
- (top level) `claude_code_prompts.md`: prompts for the Claude Code help with each report's Section 3.
- `original_figure*.png`: the paper's figure, for comparison.
- `phase*_runs/`: logs, curves and figures from our runs.

## Setup
Python 3.12, PyTorch 2.x (we used 2.14.1 on Apple-silicon MPS; CUDA and CPU also work).
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Data goes in `data/`:
- MNIST: downloaded automatically by torchvision.
- CIFAR-10: torchvision downloads it automatically. We used the Hugging Face copy instead (the Toronto server was too
  slow): `python tools/cifar_from_parquet.py data/hf data` converts it to torchvision's layout. Its pickles fail
  torchvision's md5 check, so we launch scripts through `tools/run_patched.py`, which disables only that check. With
  the official download you can run the scripts directly.
- dSprites: `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` from https://github.com/google-deepmind/dsprites-dataset,
  saved as `data/dsprites.npz`.

## Reproducing the figures
Open each notebook from its own folder and run all cells. By default (`RUN_TRAINING = False`) they redraw the figures
from the saved logs in seconds. Set `RUN_TRAINING = True` to retrain. Rough times on an M4 Pro laptop:
- Zhang: ~4 h (Inception ≈ 34 min per 25k-step run).
- Keskar: ~3 h (C2 ≈ 30 min per SB/LB pair).
- β-VAE: ≈ 0.5–1.5 h per run, 6 runs.

The exact training commands are in the notebooks and in `zhang/run_queue.sh`, `keskar/run_queue.sh` and
`betavae/phase3_seeds.sh`. Keskar's diagnostics: `keskar/ablation_bn_interp/run.sh` (BatchNorm ablation) and
`keskar/diagnostics_queue.sh` (fixed 20-epoch runs and extra F1 seeds, after Phase 4).

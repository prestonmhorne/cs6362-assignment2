# Phase 2 review of the Phase 1 output (Sonnet 5.5, Medium effort, claude.ai)

Code saved unmodified as `phase1.py`. Run through `../tools/run_patched.py` only because
CIFAR-10 came from the Hugging Face mirror (md5 check skipped); no change to the LLM's code.

## Verified OK
- All three architectures match Table 1's parameter counts **exactly** (asserted at build time).
  The Inception match requires conv layers without bias and BatchNorm with a shift but no scale,
  which the LLM inferred from the counts. Worth a sentence in the report.
- Preprocessing: /255, center crop 2:30, per-image standardization with the 1/sqrt(N) std floor (TF semantics).
- SGD with momentum 0.9; LR 0.1 for Inception and 0.01 otherwise; ×0.95 per epoch (staircase); no weight decay.
- Random labels are drawn once (seeded) and fixed across epochs. Corruption is uniform over 10 classes, so a
  corrupted label can land on the true class (0.9p of labels actually wrong). Test labels stay clean.
- Shuffled pixels use one permutation for train and test; random pixels use a new permutation per image.
- Pipeline runs end to end (smoke grid: 9 runs, 5 min, all train_err = 0).
- Timing on the M4 Pro (MPS, batch 128): Inception 81 ms/step (34 min per 25k steps), AlexNet 15 ms (6.4 min), MLP 0.8 ms.

## Problems / disagreements to fix in Phase 3
1. **Panel (a) truncated by default.** The `reduced` grid enables early stopping (+1000 steps after convergence),
   so the curves end early. The paper's curves run the full x-axis (~25k steps). Fix: run the 5 Inception
   panel-(a) settings with `--early_stop off`.
2. **Gaussian inputs drawn in the whitened space** from per-channel stats of the *whitened* train set. The paper
   says "matching mean and variance to the original image dataset". The more literal reading is to sample in
   raw-pixel space with the dataset's mean/var, then apply the same crop and per-image whitening. Decide and document.
3. **Convergence rule is the LLM's own invention** (200-step rolling mean of minibatch train-mode loss < 0.05).
   It's acceptable, but report it as our choice, and check that the result for (b) doesn't hinge on tau
   (the plot step can re-threshold from logs: try tau ∈ {0.02, 0.05, 0.1}).
4. **Risk: LR decay vs. random labels.** With 390 steps/epoch, the LR at 25k steps is 0.95^64 ≈ 0.037× initial.
   Random-label runs (especially MLP and AlexNet) may stall before reaching tau. Being tested now in
   `phase1_runs/` (MLP and AlexNet reduced grid).
5. **Batch size 128 is unstated in the paper.** Because the LR decays per epoch, the batch size changes the schedule. Report it.
6. Minor: the docstring calls the file `zhang_fig1.py`; the default `--data_root ./data` is relative to the CWD.

## Phase 2 evidence to collect
- `phase1_runs/` results for MLP and AlexNet (does p = 1 converge? does the p = 0 test error roughly match Table 1:
  AlexNet ~76%, MLP ~50.5% accuracy?)
- `smoke_runs/figure1.png`: plumbing only (2,000-image subset, 600 steps); its numbers are meaningless.

## Evidence from the Phase 1 MLP runs (phase1_runs/, 25k steps, batch 128, tau 0.05)
| p | 0.0 | 0.1 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| crossed tau at step | 14200 | 23200 | never | never | never | never | never |
| final train err | 0.0008 | 0.0011 | 0.0020 | 0.0041 | 0.0074 | 0.0098 | 0.0115 |
| final test err | 0.486 | 0.540 | 0.583 | 0.671 | 0.747 | 0.827 | 0.898 |

- p=0 test error 48.6% vs Table 1's 49.5% → consistent.
- p=1 train error 1.15% vs Table 1's 0.66% (99.34% acc) → the paper's MLP did not fully fit random labels either.
- For p ≥ 0.2 the loss never drops below 0.05: the LR has decayed to ~0.037× by 25k steps, so running longer barely helps.
  A loss threshold therefore can't produce panel (b) for the MLP. Fix (Phase 3): define convergence by a
  training-accuracy criterion, which can be computed from the logged per-step accuracies without retraining.

## Phase 3 change 1 (Sonnet, same chat): early stop off for panel (a), Gaussian sampled in raw space, tau sweep, --grid panel_a
- Saved as phase3.py. Smoke test: the four non-Gaussian runs are bit-identical to Phase 1; only the Gaussian run changed.

## Phase 3 change 2 (Sonnet, same chat): accuracy-based convergence criterion for panel (b), computed from logs
- Saved as phase3.py (previous version kept as phase3_v1.py). Training code is byte-identical to v1; only plotting changed.
- Tested on the Phase 1 MLP logs (plotted with phase3.py into phase1_runs/; re-plot with phase1.py later for the Phase 1 evidence figure).
  MLP relative time-to-overfit at p=1: 2.13 (acc>=0.95), 2.27 (0.98), >=2.43 censored (0.99; p>=0.8 never reach 99%).
  Paper Fig. 1b MLP at p=1: ~2.0. Shape (monotone rise, ~2x) matches at 0.95/0.98.
- Plot nit: hollow star markers for censored points are hard to tell from filled ones at this size.

## Disclosures for the report (from the Zhang chat transcript)
- The chat ran in claude.ai incognito mode (no share link); the instructor approved a full markdown transcript instead
  (`llm_transcript.md`, copied message by message with the copy button; linear, no edited messages). Its three code
  blocks are byte-identical to phase1.py, phase3_v1.py and phase3.py. `llm_chat_partial.webarchive` is a Safari
  archive of the same chat (partial: claude.ai only keeps the visible part of long chats loaded).
- **Web search was used once**, in branch A, *after* all code was written: Sonnet searched Anthropic's support docs
  to answer "can I share this chat even though it is incognito?". No code was retrieved. Before that, Sonnet stated it
  made no tool calls. The search toggle was evidently not off despite the plan to disable it.
- Details Sonnet says came from its background knowledge rather than the paper: TF per-image standardization's
  1/sqrt(n) std floor; LRN hyperparameters (size 9, α=0.001/9, β=0.75, k=1).

## Final Figure 1 (phase3_runs/figure1.png): Inception from phase3_runs, MLP/AlexNet symlinked from phase1_runs; panel (b) uses train acc >= 0.98
- (a) Every setting reaches ~0 loss within 25k steps. Ordering: true labels < shuffled pixels < gaussian < random pixels < random labels
  (random labels plateau at 2.30 until ~7k steps, then fit by ~15k). This matches the paper's qualitative claim, with one exception:
  our gaussian run fits faster than random pixels.
- (b) Time to overfit rises with p (monotonically for Inception and AlexNet; the MLP dips slightly at p=0.8). At p=1 the ratios are Inception 3.5x, AlexNet 2.9x, MLP 2.3x. The paper's ordering
  is similar, with smaller factors. All points resolve at 0.98; at 0.99 the MLP is censored for p >= 0.8.
- (c) Test error rises linearly toward 0.9 at p=1 for all three. Inception at p=0 is 16.8% (Table 1, no regularizers: 14.25%).
- Caveat: AlexNet/MLP runs used the reduced-grid early stop (+1000 steps), so the loss-tau sweep is censored; the acc criterion is not.

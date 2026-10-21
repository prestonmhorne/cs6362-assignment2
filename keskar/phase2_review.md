# Phase 2 review of the Phase 1 output (Sonnet 5.5, Medium effort, claude.ai, separate chat)

Code saved unmodified as `phase1.py`; run through `../tools/run_patched.py` (CIFAR-10 from the HF mirror, md5 check skipped).

## Setup summary checked against the paper — correct
- Panels: F1 = MNIST FC (B.1), C1 = CIFAR-10 shallow conv (B.3), C2 = CIFAR-10 VGG-like (B.4). Table 1.
- Architectures match B.1/B.3/B.4, including the "C3 network" typo in B.4 (it means C2).
- LB = 10% of the training set, SB = 256, ADAM for both (§2.2). No ADAM hyperparameters given.
- α ∈ [−1, 2], f(αx_ℓ + (1−α)x_s); α=0 is SB, α=1 is LB; left axis cross-entropy, right axis accuracy;
  solid = train, dashed = test (§2.2.1, Fig. 3 caption).
- Training "without any budget or limits, until the loss function ceased to improve" (§2.2). No epoch count.
- 5 trials from "uniformly distributed random" starting points (§2.2); Table 2 train/test accuracies.
- Note: footnote 1 links the authors' GitHub code. **Not opened (Phase 4 only).**

## Smoke test (2 epochs, n_alpha 5; GPU shared with 2 Zhang jobs, so timings are pessimistic)
- F1 and C1 ran end to end (train → interp → curve.json). C2 SB finished; C2 LB died at epoch 1 when the laptop
  battery ran out (not a code error).
- s/epoch: F1 SB 5.3, LB 0.8; C1 SB 5.4, LB 4.1; C2 SB 86, LB 116 (with gradient checkpointing).
- C1 SB/LB distance-from-init ratio 2.8 after 2 epochs (paper reports 3–10 at convergence).
- C1 LB native endpoint (own running BN stats) test acc 11% vs 40% train-mode acc after only 20 updates: running
  BN stats lag badly early on. This supports recomputing BN stats at each α.

## Issues to examine (written before the full F1/C1 runs in phase1_runs/ finished; results below)
1. **Stopping rule.** The paper trains until the loss stops improving; the code uses a fixed 100 epochs. LB gets only
   1000 updates (F1: 10/epoch). Check whether the final train accuracy reaches Table 2 (F1 LB 99.92%, C1 LB 99.66%,
   C2 LB 99.99%; SB 99.66 / 99.89 / 99.99%). If not, LB is under-trained and the comparison is unfair.
2. **Shared initialization** is the LLM's inference. The paper only says each trial starts from a random point. Report it as our choice.
3. **BN at interpolated weights**: recompute from the full training set (default). Defensible; report it. `--bn_mode interp`
   exists as an ablation.
4. **Single seed.** The paper picks one random pair from 5 trials, so one pair is faithful to the figure, but Table 2 needs 5.
5. **C1 pooling stride/padding** is unspecified: chose conv s2 p2 + MaxPool(3, s2, p1), giving a 2x2x64 = 256-d flatten (very small).
6. **Same lr (1e-3) for SB and LB.** Unspecified in the paper; report it.
7. **Dropout in C2's dense layer** is assumed ("as in B.3").

## Evidence: full F1 and C1 runs with the Phase 1 code (phase1_runs/, 100 epochs, shared init, BN recompute)
| | F1 SB | F1 LB | C1 SB | C1 LB |
|---|---|---|---|---|
| train acc (eval) | 99.99 | 100.00 | 97.81 | 96.80 |
| Table 2 train acc | 99.66 | 99.92 | 99.89 | 99.66 |
| test acc | 98.69 | 98.61 | 69.48 | 65.86 |
| Table 2 test acc | 98.03 | 97.81 | 80.04 | 77.26 |
- Distance-from-init ratio ||x_s−x0||/||x_l−x0||: F1 14.2, C1 3.5 (paper: 3–10).
- **F1 matches qualitatively**: the loss is flat between α=0 and 1 and rises steeply just beyond α=1 (LB side sharp).
  The SB–LB test gap is 0.08 pt (paper 0.22).
- **C1 doesn't reach the paper's regime**:
  - Test accuracy is ~10 pt below Table 2, and train accuracy is below Table 2.
  - Training loss was still falling at epoch 100 (SB 0.099→0.076 over epochs 75–100; LB 0.129→0.058), so the
    100-epoch cap violates "until the loss function ceased to improve".
  - The SB–LB test gap (3.6 pt) has the right sign and size (paper 2.8 pt).
  - The curve shows a loss bump near α≈0.85 and a narrow dip at α=1, then a large rise beyond α=1.
- Suspected causes for C1, to test in Phase 3:
  1. The fixed epoch cap (under-training).
  2. The unspecified downsampling: conv stride 2 + pool stride 2, twice, collapses 32×32 to 2×2, giving a 256-d flatten
     feeding the 384-unit dense layer. That's a very small bottleneck, plausibly why test accuracy is ~70% rather than ~80%.

## Phase 3 change 1 (Sonnet, same chat): stopping rule + C1 downsampling ablation
- Saved as phase3.py. New: train until the eval-mode full-train loss fails to improve by 1% (relative) for 10 epochs,
  with a cap (default 300; C2 capped at 40 SB / 40 LB, checks every 5 epochs). C1 variants: `alexnet` (stride-1 convs,
  3x3 s2 pools, 4096-d flatten; new default) and `collapse` (the original reading).
- Sonnet's reasoning: the "2" in [64,5,5,2] is more plausibly the overlapping-pool stride of the "modified AlexNet".
  It also cited a remembered CIFAR-10 reference net with 384/192 dense layers, which it flagged as memory, not evidence.
  **Disclose this in the report**: it comes from the LLM's prior knowledge, not the paper.
- Smoke test: OK. C1 collapse reproduces the Phase 1 smoke numbers exactly (61.18% test after 2 epochs) → same init.
- Sonnet's own correction: under gradient checkpointing, BN running stats update twice per step (affects only C2 LB
  eval-mode numbers, not the curves, since BN is recomputed at every α).

## Phase 3 final: C2 (cap 40 epochs, stop checks every 5) and combined figure3_reproduction.png
- C2: SB stopped at epoch 30 (patience); LB hit the 40-epoch cap.
  α=0 (SB): train ≈100%, test ≈86%. α=1 (LB): train ≈98%, test ≈81%. A ~5-point generalization gap in the paper's direction.
- All four panels show the paper's key shape: loss rises much more sharply on the LB side (α>1) than on the SB side (α<0).
- Difference from the paper: C1 and C2 show a loss bump/accuracy dip at α≈0.85, i.e. a barrier between the two minimizers.
  The paper's curves are smooth and monotone between α=0 and 1. Likely cause: recomputing BatchNorm statistics per α, and/or
  the shared-init choice. Report this.

## Correction (2026-10-20, during Phase 4)
The note above that the paper's curves are "smooth and monotone between α=0 and 1" is wrong. At higher resolution, the
paper's C1 and C2 panels show the same bump/dip near α≈0.75, with similar heights to ours. Details, plus a BN
ablation, are in phase4.md.

# Phase 4: comparison with the authors' code (done after Phase 3 was committed, 674bce2 in our private working repo; see ../commit_history.md)

Authors' repo: https://github.com/keskarnitish/large-batch-training (linked in the paper's footnote 1).
- `plot_parametric_plot.py` + `network_zoo.py`: Keras 1 / Theano, networks C1–C4. This is what we compare against.
- `PyTorch/plot_parametric_pytorch.py`: a later "WIP" port for a VGG11 network (C3-like). Not one of our panels.
- There is no F1 (MNIST) code in the repo.

## Comparison with our Phase 3 code (phase3.py)
| Item | Authors' Keras code | Ours | Paper text |
|---|---|---|---|
| C1 architecture | conv5x5(64,'same')-BN-ReLU-maxpool3x3 s2 'same', twice; FC384-BN-ReLU-Dropout0.5; FC192-BN-ReLU-Dropout0.5; FC10 | identical to our `alexnet` variant (4096-d flatten). Our Phase 1 `collapse` reading was wrong | "[64,5,5,2]": ambiguous |
| C2 architecture | VGG-16-style 64-64-M-128-128-M-256×3-M-512×3-M-512×3-M, BN; **dropout 0.3/0.4 after most convs**, dropout 0.5 before and after FC512 | same conv stack and BN, **no conv dropout**, dropout only after FC512 | Appendix B.4 table: no conv dropout mentioned |
| BatchNorm at evaluation | `BatchNormalization(mode=2)` (Keras 1): normalises with the **current batch's statistics at train and test time**. Eval batch = 5000, so every α gets fresh batch statistics | `--bn_mode recompute`: running stats re-estimated on the full train set at each α | not stated |
| What is interpolated | trainable weights only (incl. BN γ, β); no running stats exist in mode 2 | all float tensors, then running stats overwritten by recompute. Equivalent | — |
| Initialisation | **shared**: `x0.h5` saved once, both runs start from it | shared (`--init shared`) | "a pair of SB and LB minimizers **from the 5 trials**" (suggests independent runs) |
| SB / LB batch | 256 / 5000 | 256 / 5000 (10% of 50k) | 256 / 10% of data |
| Optimizer | Adam, Keras defaults (lr 1e-3) | Adam lr 1e-3 | Adam, lr not stated |
| Preprocessing | /255, no augmentation | /255, no augmentation | none stated |
| Training length | fixed `nb_epoch = 20` | patience rule on full-train loss (C1 cap 300, C2 cap 40) | "until no improvement" (Fig. 2 shows 100 epochs) |
| α grid | `linspace(-1, 2, 25)` | `linspace(-1, 2, 61)` | α ∈ [−1, 2] |
| Loss axis | linear (Keras); log (`semilogy`) in the PyTorch port | linear | — |

## Discrepancies and their likely effect
1. **C1 architecture**: the authors' code confirms Sonnet's Phase 3 change (the `alexnet` variant). By the paper's own
   notation (B.3) the "2" in [64,5,5,2] is the conv stride; the authors' code uses it as the pool stride instead. Our main
   C1 panel already uses the code's reading; the `collapse` panel stays as the literal Phase 1 reading.
2. **C2 conv dropout**: our C2 has none, theirs has dropout after most convs. Dropout is a regularizer, so it probably
   explains part of the gap in SB test accuracy (ours: 81.6% with the trained running BN stats, 85.5% with BN
   recomputed at α=0). It should not change the shape of the parametric plot, because dropout is off at evaluation in
   both. We did **not** retrain C2 with dropout: that would need new model code, and the course rules allow us to
   use only the LLM chat's code. So this is a reported difference, not a tested one.
3. **BatchNorm**: Keras mode-2 batch statistics on 5000-image batches are, up to sampling noise, the population
   statistics we recompute. So our `recompute` choice is the closest match to the authors' evaluation.
4. **Shared vs. independent init**: the released code shares the init, matching our choice. The paper's sentence
   about picking a pair "from the 5 trials" is the only hint otherwise. We report this inconsistency between the
   paper and its code.
5. **Training length**: the released script trains a fixed 20 epochs. Our longer, patience-based training is closer to
   the paper's description, but this is another paper/code mismatch.

## Correction: the paper's Fig. 3 has the same barrier
The Phase 3 note (phase2_review.md, "Phase 3 final") said the paper's curves are smooth and monotone between α=0 and 1.
That was a misreading of the small figure. At 200 dpi (`original_figure3.png`), the paper's C1 and C2 panels (and F2,
C3, C4) show the same loss bump / accuracy dip just before the LB end. F1 is smooth in both.

| Panel | Paper Fig. 3 (read off the plot) | Ours, Phase 3 (BN recomputed) |
|---|---|---|
| F1 | smooth on [0, 1] | smooth; max train loss 0.011 at α=0.85 (99.6% acc) |
| C1 | bump at α≈0.75: train loss ≈0.85, train acc ≈83% | bump at α=0.80: train loss 0.68, train acc 81.1% |
| C2 | spike at α≈0.75: train loss ≈1.7, train acc ≈60% | spike at α=0.85: train loss 1.57, train acc 57.7% |

So the barrier is a **feature we reproduced**, not a mismatch. Its location and height are close to the paper's.
Their 25-point grid puts points at α = 0.75 and 0.875, so the location difference (0.75 vs 0.80–0.85) is within one
grid step of theirs. What does differ: the height of the curves on the LB side (α > 1). The paper's C1 train loss peaks
at ≈7.5 near α=1.4 and F1 reaches ≈11 at α=2; ours peak at ≈5.2 and ≈4.9. Later diagnostics (report Section 2, "Remaining mismatch") rule out both the fixed
20-epoch training (it lowers the LB side) and run-to-run noise (three F1 seeds peak at 4.8–5.0).

## Diagnostic run after Phase 4: how BN handling changes the barrier
Test: re-evaluate the same saved Phase 3 models with `--bn_mode interp`, i.e. interpolate the running BN stats
instead of recomputing them. No training and no code change (the flag already exists in the Phase 3 code). Outputs are
in `ablation_bn_interp/` (run script `ablation_bn_interp/run.sh`).

Peak train loss in α ∈ [0, 1] (accuracy at that point):
| Panel | recompute (Phase 3) | interp running stats |
|---|---|---|
| F1 | 0.011 at α=0.85 (99.6%) | 0.380 at α=0.85 (86.9%) |
| C1 (alexnet) | 0.677 at α=0.80 (81.1%) | 2.710 at α=0.85 (35.9%) |
| C2 | 1.569 at α=0.85 (57.7%) | 9.680 at α=0.55 (10.0%, chance); loss > 5.8 for all α in [0.25, 0.95] |

With interpolated running stats the barrier is several times higher than in the paper. With recomputed stats it matches
the paper's height. This is consistent with the authors' code: Keras `BatchNormalization(mode=2)` normalises with the
current batch's statistics, which is effectively what recomputing does. So the BN choice matters for the figure,
and recomputing is the one that reproduces the paper. (With interpolated stats, extrapolation gives NaN loss for α > 1 in all three
panels, and for α < −0.25 in C2, because the extrapolated running variance goes negative. The flat 10% accuracy in
`ablation_bn_interp/figure3_reproduction.png` there is an artefact of NaN outputs. So interpolated statistics can't
even produce the α ∈ [1, 2] half of the figure.) The barrier itself is a genuine non-convexity along the
segment. It sits near the LB end, which fits the paper's reading that the LB minimizer's basin is narrow.

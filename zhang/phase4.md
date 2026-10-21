# Phase 4: comparison with an existing implementation (done after Phase 3 was committed, 674bce2 in our private working repo; see ../commit_history.md)

## What exists
- The authors never released the code behind the paper. The first author (Chiyuan Zhang, GitHub `pluskid`) says in
  https://github.com/pluskid/fitting-random-labels that "the original code for the paper belongs to Google".
  That repo is a demo written later from scratch (last commit Jun 2020). It deliberately does **not** reproduce the
  paper's experiments: it uses a WideResNet-28-1 and an MLP, not Inception/AlexNet.
- So there is no authoritative implementation to diff against. The comparison below is against this demo plus the
  author's tips in its README.

## Comparison with our Phase 3 code (phase3.py)
| Item | Demo repo | Ours | Paper (App. A) |
|---|---|---|---|
| Label corruption | `mask = rand <= p`, replaced by `np.random.choice(10)` with a fixed seed (12345), so the true class can be redrawn | `rand < p`, replaced by `randint(0,10)` with a fixed generator seed, drawn once | uniform random class, fixed |
| Labels fixed across epochs | yes (stored in the dataset object) | yes (drawn once before training) | yes |
| **Test labels** | **corrupted too** (`val_loader` gets `corrupt_prob`) | true test labels | true test labels (Fig. 1c reports test error on clean labels) |
| Preprocessing | per-channel mean/std normalisation, no crop | 28x28 center crop + per-image standardisation | crop 28x28 + per-image whitening |
| Augmentation | off by default | off | off |
| Optimizer | SGD, momentum 0.9, lr 0.1, **weight decay 1e-4** | SGD, momentum 0.9, lr 0.1 (Inception) / 0.01 (AlexNet, MLP), no weight decay | as ours |
| LR schedule | ÷10 at epochs 150 and 225 (the jump to ~100% train acc in its random-label log is at epoch 150, i.e. at the LR drop) | ×0.95 per epoch | ×0.95 per epoch |
| Train metrics | full train set re-evaluated in eval mode each epoch | full train set in eval mode at log points | not stated |

## Discrepancies and what we changed
- **No changes to the code.** Every difference in the table is a point where the demo departs from the paper's
  Appendix A and our code follows the paper.
- The demo corrupts the test labels as well. With it, Fig. 1c's "test error" would be measured against noisy labels and
  would rise faster than 0.9·p. Copying the demo would have been a mistake.
- The README tips match what we did and found: keep random labels fixed across epochs, turn off augmentation and
  regularizers, and expect random labels to need more epochs. Our panel (a) random-label run sits at ln 10 ≈ 2.30
  until ~7k steps before fitting. That is the same plateau-then-fit pattern as the demo's WRN log (loss 2.30 for ~5
  epochs, then a fit).
- The demo also confirms the question "what fraction of labels actually differ": the redraw can land on the true
  class, so the fraction wrong is 0.9·p, and test error at p = 1 is ≈ 90%.

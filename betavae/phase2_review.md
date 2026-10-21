# Phase 2 review of the Phase 1 output (Sonnet 5.5, Medium effort, claude.ai, incognito chat)

Code saved unmodified as `phase1.py`. Transcript: `llm_transcript.md` (copy-button markdown). The chat was incognito, so there is no share link.

## Setup summary checked against the paper — correct
- Table 1 (2D shapes, VAE): Adagrad lr 1e-2; encoder FC 1200-1200 ReLU; 10 latents; decoder FC 1200-1200-1200-4096,
  "Tanh activation. Bernoulli". Matches.
- Fig. 7 caption: columns = latents ordered by learnt Gaussian variance (last row); row 1 = mean activation over the
  32x32 positions averaged across objects/rotations/scales; rows 2–3 = mean activation vs scale / rotation, one curve
  per shape (square red, oval green, heart blue); rows 4–8 = traversal of each latent over ±3 SD around the prior
  mean, other 9 latents fixed to the values inferred from one dataset image. 7B = same for VAE (β = 1). Matches.
- Eq. 4 objective (summed reconstruction − β·KL). Matches.

## Choices the paper doesn't pin down (report these)
1. Batch size 64 and 300k steps (~26 epochs). Unspecified in the paper.
2. "Tanh ... Bernoulli" read as tanh hidden units + sigmoid/logit output.
3. "Learnt variance" = dataset mean of the posterior variance per latent.
4. Unused-latent threshold KL < 0.1 nats; one unused column shown.
5. One fixed random traversal seed image (index from RandomState(0)), 5 traversal steps.
6. No train/test split; evaluation over all 737,280 images.
7. Seeds: default 1 per β. The paper's metric protocol discards the bottom 50% of 30 scores (10 seeds × 3 classifier seeds), so seed variance is real.

## Smoke test (2000 steps per β, GPU shared with two other jobs: ~19 ms/step)
| | β=4 | β=1 |
|---|---|---|
| active latents (KL>0.1) | 4 | 9 |
| total KL (nats) | 9.03 | 16.06 |
| recon NLL, sampled z (nats/image) | 122.55 | 114.49 |
Direction matches the paper's claim (stronger KL pressure → fewer active latents, worse reconstruction).
The paper's Fig. 7A shows 5 informative latents for β=4 (x, y, scale, and two for rotation).

## Full Phase 1 run, β=4 seed 0 (300k steps, ~60 min while sharing the GPU)
- 5 active latents (KL > 0.1), 5 collapsed to the prior (var 0.99, KL 0.00). This matches the count in the paper's
  Fig. 7A (x, y, scale, rotation ×2).
- Qualitative reading of fig7.png (ordered by variance):
  - z8 (KL 3.32): ≈ position Y (vertical gradient, flat in scale and rotation). Clean.
  - z2 (KL 2.61): **scale + diagonal position**. Entangled.
  - z9 (KL 2.36): **scale + position X-ish**. Entangled.
  - z4 (KL 2.13): ≈ position X (horizontal gradient), mild scale, shape-specific rotation ripple.
  - z7 (KL 1.17): rotation (periodic for oval and heart; the square shows 4-fold symmetry). Clean-ish.
- So: partially disentangled. Y and rotation are clean; scale is split across z2/z9 and mixed with position.
  The paper's 7A has separate clean latents for X, Y, scale and rotation. One seed can't tell us whether this is
  seed variance (the paper's own protocol discards the bottom 50% of runs) or a systematic gap (training length,
  batch size).
- Cosmetic: the suptitle overlaps the column titles.

## Full Phase 1 run, β=1 seed 0 (300k steps)
- recon NLL 38.58 (vs 65.50 at β=4), total KL 18.52 (vs 11.58), 6 active latents (vs 5). The direction matches the
  paper: β=4 trades reconstruction for a more compressed code.
- Qualitative reading of fig7.png:
  - z8 (KL 4.12): position Y. Clean. z1 (KL 3.86): position X. Clean.
  - z7 (KL 3.40): scale. Flat in rotation, but its position map is bowl-shaped (corrected on re-inspection, 2026-10-20).
  - z4, z0, z2 (KL 2.67 / 2.51 / 1.95): **rotation mixed with shape identity**. Traversals morph square ↔ oval ↔
    heart/triangle, and z0 at +3 produces two blobs.
- So β=1 is *not* fully entangled, as the paper's Fig. 7B shows. Position and scale come out clean here, arguably cleaner
  than in our β=4 run, where scale was split across z2/z9. The β contrast is visible mainly in rotation and shape:
  rotation is one clean latent at β=4 but smeared across three shape-mixing latents at β=1. With one seed per β,
  this is weak evidence. Seed variance is a plausible explanation (see the 50%-discard protocol).

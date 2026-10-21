# Zhang et al. 2017 — working notes

## Setup stated in the paper (Appendix A) — Phase 2 checklist
Check the chat LLM's code against each item.

| Item | Paper says | Where |
|---|---|---|
| Preprocessing | /255 → **center crop 28x28** → **per-image whitening** (TF `per_image_standardization`: subtract the image mean, divide by max(std, 1/sqrt(num_pixels))) | App. A |
| Augmentation | **None** for the randomization experiments | App. A, Table 1 |
| Optimizer | SGD, **momentum 0.9** | App. A |
| LR | **0.1** small Inception; **0.01** AlexNet and MLPs | App. A |
| Schedule | multiply by **0.95 every epoch** | App. A |
| Regularization | **no** weight decay, dropout, etc. | App. A |
| Inception (small) | Conv(96,3x3) → Inc(32+32) → Inc(32+48) → Down(80) → Inc(112+48) → Inc(96+64) → Inc(80+80) → Inc(48+96) → Down(96) → Inc(176+160) ×2 → global mean pool 7x7 → FC10. Conv module = conv→**BN**→ReLU. Down = conv3x3 s2 ‖ maxpool3x3 s2, concat | Fig. 3 |
| AlexNet (small) | 2 × [conv5x5 → maxpool3x3 → **LRN**] → FC384 → FC192 → FC10 | App. A |
| MLP 1x512 | one hidden layer, 512 units, ReLU | App. A |
| Batch size | **not stated** → we choose it and report the choice | — |
| Total steps | Fig. 1a x-axis goes to 25k steps | Fig. 1a |
| Random labels | **fixed once, consistent across epochs** | §2.1, App. A |
| Shuffled pixels | **one** permutation applied to all train **and test** images | §2.1 |
| Random pixels | a **different** permutation per image | §2.1 |
| Gaussian | i.i.d. Gaussian with **matching mean/variance** of the dataset | §2.1 |
| Corrupted labels | with prob. p, replace with a **uniform** random class (can land on the true class) | §2.1 |

Parameter counts to check the architectures against (Table 1): Inception 1,649,402; AlexNet 1,387,786; MLP 1x512 1,209,866 (= 28·28·3·512+512 + 512·10+10, which confirms 28x28 inputs).

## Questions the report must answer
- **Random labels across epochs:** generate them once (fixed seed) and reuse them every epoch. If they were redrawn each epoch, the target would be pure noise every time, so the network has nothing consistent to memorize. Training loss would stay around ln 10 ≈ 2.30, and the "fits random labels" result would vanish. (That's an experiment worth running as an ablation.)
- **Fraction actually wrong at corruption p:** a corrupted label is uniform over 10 classes, so it matches the truth 1/10 of the time. Fraction wrong = **0.9p**. At p = 1, test error ≈ **90%** (chance), which is the red dashed line in Fig. 1c.
- **Optimizer, schedule, preprocessing, regularization:** see the table above.

## Choices we must make and report
- **Convergence time (Fig. 1b):** y-axis is "time to overfit", *relative* (1.0 at p = 0). Proposal: the first step at which training accuracy reaches 100% (or loss < threshold), divided by the same number at p = 0, per architecture. State the threshold.
- **Grid reduction:** decided after timing; see report.md, "Choices we had to make".

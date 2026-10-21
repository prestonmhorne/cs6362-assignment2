I've attached the paper "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (Higgins et al., ICLR 2017). I want to reproduce Figure 7A (β-VAE, β = 4) and Figure 7B (standard VAE, β = 1) on the 2D Shapes (dSprites) dataset from scratch in PyTorch. Please do not search the web or use any existing implementation — work only from the paper. I already have the official dSprites file `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` (keys: imgs (737280, 64, 64), latents_values and latents_classes (737280, 6)).

1. First, summarize the exact setup the paper specifies for these panels: the encoder/decoder architecture, latent dimensionality, likelihood, optimizer and learning rate, batch size and training length, and the objective L_β. Then explain precisely what each row of Figure 7A/B shows: how the latents are ordered, how the "mean latent response" rows for position, scale and rotation are computed (what is averaged over, and how the three shapes are distinguished), and how each latent traversal is generated (range, which latent values are held fixed, and from which input). Point out anything the paper leaves unspecified.
2. Then write a complete, runnable PyTorch implementation (a single Python script or notebook cells) that trains a β = 4 model and a β = 1 model, then produces both panels: the latent-response rows and the traversal rows, in the style of the paper's figure.
3. Also report per-latent KL divergence and reconstruction loss for each model, since I need to discuss how β affects disentanglement versus reconstruction quality.
4. I'm running on an Apple M4 Pro laptop (PyTorch MPS backend, 24 GB RAM), so tell me the expected runtime and clearly list every deviation from the paper. Make the number of random seeds per β a command-line option.

Do not hardcode or fabricate any values in the plots; every number must come from actual training runs logged to disk.

# β-VAE Figure 7A/B reproduction on dSprites

## 1. What the paper specifies for these panels

### Setup (Table 1, Section 4.2, Fig. 7 caption, Eq. 4)

| Item | Paper |
|---|---|
| Data | 737,280 binary 64x64 images: 3 shapes x 6 scales x 40 rotations x 32 x-positions x 32 y-positions. Input is flattened to 4096. |
| Encoder | FC 1200, FC 1200, ReLU |
| Latents | 10 (Gaussian posterior q(z\|x) = N(μ(x), σ²(x)), prior N(0, I)) |
| Decoder | FC 1200, 1200, 1200, 4096, "Tanh activation", Bernoulli likelihood |
| Optimizer | Adagrad, learning rate 1e-2 |
| β | 4 for panel A, 1 for panel B (same architecture) |
| Objective (Eq. 4) | L = E_q(z\|x)[log p(x\|z)] - β · KL(q(z\|x) \|\| p(z)), maximised. The reconstruction term is a sum over the 4096 pixels (Eq. 7). |

Appendix A.6 also gives a normalised form, β_norm = βM/N, which for β=4, M=10, N=4096 is about 0.0098. It is the same objective written with per-pixel and per-latent means. The implementation below uses the summed form of Eq. 4.

### What each part of Fig. 7A/B shows

- **Columns.** Each column is one latent z_i. Columns are ordered by learnt Gaussian variance (the last row of the top group), ascending. The panels show the informative latents plus one rightmost "Z_unused" column, whose variance is about 0.99 (essentially the prior).
- **Row 1 (position).** The mean activation of z_i, meaning the posterior mean μ_i(x), as a function of the 32x32 object location. It is averaged over all shapes, scales and rotations. It is shown as a heatmap with red for high values.
- **Row 2 (scale).** The mean activation of z_i as a function of the 6 scale values, averaged over rotations and positions. There is one line per shape: square red, oval green, heart blue.
- **Row 3 (rotation).** The same as row 2, but over the 40 rotation values, averaged over scales and positions.
- **Variance row.** The learnt posterior variance of each latent, printed under the plots.
- **Rows 4 to 8 (traversals).** Take one image from the dataset and run inference on it. Then vary a single latent z_i over three standard deviations around the prior mean, which is the range [-3, 3] (the axis in the figure reads 3, 0, -3). The other 9 latents stay at the values inferred from that image. Each decoded image is shown. Reading the figure, there are 5 rows, so 5 traversal values.

### What the paper leaves unspecified

- Batch size, number of iterations or epochs, weight initialisation, and Adagrad's other settings.
- The decoder output nonlinearity. "Tanh" with a "Bernoulli" likelihood is ambiguous, because a Bernoulli needs a probability or logit at the output.
- How the encoder variance is parameterised.
- The exact "learnt variance" statistic (per-image, dataset mean, or something else).
- Which seed image is used for the traversals.
- The number of traversal steps (5 is read off the figure, not stated), and whether the fixed latents use μ or a sample.
- The threshold for calling a latent "unused".
- The colormap, the axis orientation of the position heatmaps, and which replica was plotted.
- The axis labels like "pos. X" and "rotation" appear to be hand-assigned. The paper does not describe an automatic procedure.
- Whether training used a split. Nothing indicates one.

## 2. Implementation

Save as `beta_vae_fig7.py`.

```python
#!/usr/bin/env python
"""
Reproduce Fig. 7A (beta=4) and Fig. 7B (beta=1) of Higgins et al., ICLR 2017
on dSprites, from the paper only.

Example:
  python beta_vae_fig7.py --data dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
      --betas 4 1 --num_seeds 3 --steps 300000 --out runs
Smoke test:
  python beta_vae_fig7.py --data ... --steps 3000 --num_seeds 1 --out runs_smoke
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

N_PIX = 4096


# ----------------------------------------------------------------- model
class BetaVAE(nn.Module):
    """Table 1, 2D shapes (VAE): enc FC1200-FC1200 (ReLU), 10 latents,
    dec FC1200-FC1200-FC1200-FC4096 (tanh hidden), Bernoulli output (logits)."""

    def __init__(self, z_dim=10):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Linear(N_PIX, 1200), nn.ReLU(),
            nn.Linear(1200, 1200), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(1200, z_dim)
        self.fc_logvar = nn.Linear(1200, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 1200), nn.Tanh(),
            nn.Linear(1200, 1200), nn.Tanh(),
            nn.Linear(1200, 1200), nn.Tanh(),
            nn.Linear(1200, N_PIX),  # Bernoulli logits
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.dec(z)


def forward_losses(model, x, sample=True):
    mu, logvar = model.encode(x)
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
    logits = model.decode(z)
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)  # -log p(x|z), (B,)
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)                        # (B, M)
    return recon, kl_dim, mu, logvar


# -------------------------------------------------------------- training
def train(model, data, beta, args, run_dir, device):
    opt = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    N, M = data.shape[0], model.z_dim
    acc_loss = torch.zeros((), device=device)
    acc_rec = torch.zeros((), device=device)
    acc_kl = torch.zeros(M, device=device)
    perm = torch.randperm(N, device=device)
    ptr = 0
    t0 = time.time()
    model.train()
    with open(run_dir / "train_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "elapsed_s", "loss", "recon", "kl_total"]
                   + [f"kl_z{m}" for m in range(M)])
        for step in range(1, args.steps + 1):
            if ptr + args.batch_size > N:
                perm = torch.randperm(N, device=device)
                ptr = 0
            idx = perm[ptr:ptr + args.batch_size]
            ptr += args.batch_size
            x = data[idx].float()
            recon, kl_dim, _, _ = forward_losses(model, x, sample=True)
            loss = (recon + beta * kl_dim.sum(1)).mean()   # negative of Eq. 4, summed over pixels
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            acc_loss += loss.detach()
            acc_rec += recon.mean().detach()
            acc_kl += kl_dim.mean(0).detach()
            if step % args.log_every == 0:
                n = args.log_every
                kl_m = (acc_kl / n).cpu().numpy()
                row = [step, f"{time.time() - t0:.1f}", f"{float(acc_loss / n):.4f}",
                       f"{float(acc_rec / n):.4f}", f"{kl_m.sum():.4f}"]
                w.writerow(row + [f"{v:.5f}" for v in kl_m])
                f.flush()
                print(f"[beta={beta:g}] step {step}/{args.steps} loss {float(acc_loss / n):.2f} "
                      f"recon {float(acc_rec / n):.2f} kl {kl_m.sum():.2f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                acc_loss.zero_(); acc_rec.zero_(); acc_kl.zero_()
    return time.time() - t0


@torch.no_grad()
def evaluate(model, data, beta, device, seed, bs=4096):
    """Full-dataset evaluation. Returns metrics dict and all posterior means (N, M)."""
    model.eval()
    torch.manual_seed(100000 + seed)
    N, M = data.shape[0], model.z_dim
    kl_sum = torch.zeros(M, device=device)
    var_sum = torch.zeros(M, device=device)
    rec_s = torch.zeros((), device=device)
    rec_mu = torch.zeros((), device=device)
    mus = []
    for i in range(0, N, bs):
        x = data[i:i + bs].float()
        mu, logvar = model.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        rec_s += F.binary_cross_entropy_with_logits(model.decode(z), x, reduction="none").sum()
        rec_mu += F.binary_cross_entropy_with_logits(model.decode(mu), x, reduction="none").sum()
        kl_sum += (0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)).sum(0)
        var_sum += logvar.exp().sum(0)
        mus.append(mu.cpu())
    kl = (kl_sum / N).cpu().numpy()
    var = (var_sum / N).cpu().numpy()
    recon = float(rec_s / N)
    metrics = {
        "beta": beta,
        "recon_nll_sampled_z": recon,                 # nats per image
        "recon_nll_mean_z": float(rec_mu / N),
        "recon_nll_per_pixel_sampled_z": recon / N_PIX,
        "kl_total": float(kl.sum()),
        "kl_per_latent": kl.tolist(),                 # indexed by latent id
        "post_var_per_latent": var.tolist(),
        "latent_order_by_var_ascending": [int(m) for m in np.argsort(var)],
        "neg_elbo_beta1": recon + float(kl.sum()),
        "train_objective_beta": recon + beta * float(kl.sum()),
    }
    return metrics, torch.cat(mus).numpy()


# --------------------------------------------------------------- plotting
def group_mean(values, keys, n_keys):
    s = np.bincount(keys, weights=values, minlength=n_keys)
    c = np.bincount(keys, minlength=n_keys)
    return s / np.maximum(c, 1)


@torch.no_grad()
def plot_fig7(model, mu_all, var_mean, kl, classes, x_img, device, path, title,
              kl_thr, n_steps=5):
    order = np.argsort(var_mean)
    informative = [int(m) for m in order if kl[m] > kl_thr]
    unused = [int(m) for m in order if kl[m] <= kl_thr]
    cols = informative + ([unused[-1]] if unused else [])   # one representative unused latent
    unused_set = set(unused)
    nc = len(cols)

    # dSprites latents_classes columns: 0 color, 1 shape, 2 scale, 3 orientation, 4 posX, 5 posY
    shape, scale, rot, px, py = (classes[:, k] for k in (1, 2, 3, 4, 5))
    pos_maps, sc_curves, rot_curves = {}, {}, {}
    for m in cols:
        v = mu_all[:, m].astype(np.float64)
        pos_maps[m] = group_mean(v, py * 32 + px, 1024).reshape(32, 32)       # rows posY, cols posX
        sc_curves[m] = group_mean(v, shape * 6 + scale, 18).reshape(3, 6)
        rot_curves[m] = group_mean(v, shape * 40 + rot, 120).reshape(3, 40)
    s_lo = min(c.min() for c in sc_curves.values()); s_hi = max(c.max() for c in sc_curves.values())
    r_lo = min(c.min() for c in rot_curves.values()); r_hi = max(c.max() for c in rot_curves.values())

    model.eval()
    mu0, _ = model.encode(x_img)
    vals = torch.linspace(3, -3, n_steps, device=device)   # top row +3, bottom row -3
    trav = {}
    for m in cols:
        z = mu0.repeat(n_steps, 1)
        z[:, m] = vals
        trav[m] = torch.sigmoid(model.decode(z)).view(n_steps, 64, 64).cpu().numpy()

    fig = plt.figure(figsize=(1.7 * nc + 1.4, 15))
    gs = GridSpec(4 + n_steps, nc, height_ratios=[1, 1, 1, 0.4] + [1] * n_steps,
                  hspace=0.12, wspace=0.12, left=0.09, right=0.99, top=0.95, bottom=0.02)
    colors = ["red", "green", "blue"]   # square, oval (ellipse), heart
    for j, m in enumerate(cols):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(pos_maps[m], cmap="jet")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"z{m}" + (" (unused)" if m in unused_set else ""), fontsize=9)
        if j == 0:
            ax.set_ylabel("position")
        ax = fig.add_subplot(gs[1, j])
        for s in range(3):
            ax.plot(sc_curves[m][s], color=colors[s], lw=1)
        ax.set_ylim(s_lo, s_hi); ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("scale")
        ax = fig.add_subplot(gs[2, j])
        for s in range(3):
            ax.plot(rot_curves[m][s], color=colors[s], lw=1)
        ax.set_ylim(r_lo, r_hi); ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("rotation")
        ax = fig.add_subplot(gs[3, j])
        ax.axis("off")
        ax.text(0.5, 0.5, f"var {var_mean[m]:.2f}\nKL {kl[m]:.2f}", ha="center", va="center", fontsize=8)
        if j == 0:
            ax.text(-0.05, 0.5, "learnt var / KL", ha="right", va="center", fontsize=8,
                    transform=ax.transAxes)
        for r in range(n_steps):
            ax = fig.add_subplot(gs[4 + r, j])
            ax.imshow(trav[m][r], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"z={vals[r].item():+.1f}", fontsize=8)
    fig.suptitle(title + "\n(shapes in rows 2-3: square red, oval green, heart blue)", fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ main
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs")
    p.add_argument("--betas", type=float, nargs="+", default=[4.0, 1.0])
    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--kl_thr", type=float, default=0.1, help="nats; below this a latent is called unused")
    p.add_argument("--traverse_index", type=int, default=-1, help="dataset index of traversal seed image")
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    args = p.parse_args()

    device = pick_device(args.device)
    print("device:", device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True, encoding="latin1")
    imgs = d["imgs"]
    classes = d["latents_classes"].astype(np.int64)
    assert imgs.shape == (737280, 64, 64) and imgs.max() <= 1
    N = imgs.shape[0]
    data = torch.from_numpy(np.ascontiguousarray(imgs.reshape(N, -1).astype(np.uint8))).to(device)
    del imgs
    t_idx = args.traverse_index if args.traverse_index >= 0 else int(np.random.RandomState(0).randint(N))
    x_img = data[t_idx:t_idx + 1].float()
    print("traversal seed image index:", t_idx)

    all_metrics = []
    for beta in args.betas:
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            run_dir = out / f"beta{beta:g}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(seed); np.random.seed(seed)
            model = BetaVAE(10).to(device)
            ckpt = run_dir / "model.pt"
            if ckpt.exists() and not args.force:
                st = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(st["state"]); train_s = st["train_seconds"]
                print(f"loaded {ckpt}")
            else:
                train_s = train(model, data, beta, args, run_dir, device)
                torch.save({"state": model.state_dict(), "train_seconds": train_s,
                            "args": vars(args), "beta": beta, "seed": seed}, ckpt)
            metrics, mu_all = evaluate(model, data, beta, device, seed)
            metrics.update({"seed": seed, "train_seconds": train_s, "steps": args.steps,
                            "batch_size": args.batch_size, "lr": args.lr, "traverse_index": t_idx})
            kl = np.array(metrics["kl_per_latent"])
            metrics["n_active_latents"] = int((kl > args.kl_thr).sum())
            json.dump(metrics, open(run_dir / "metrics.json", "w"), indent=2)
            plot_fig7(model, mu_all, np.array(metrics["post_var_per_latent"]), kl, classes, x_img,
                      device, run_dir / "fig7.png", f"beta={beta:g}, seed={seed}, steps={args.steps}",
                      args.kl_thr)
            all_metrics.append(metrics)
            order = metrics["latent_order_by_var_ascending"]
            print(f"\n== beta={beta:g} seed={seed} ==")
            print(f"recon NLL (sampled z): {metrics['recon_nll_sampled_z']:.2f} nats/image "
                  f"({metrics['recon_nll_per_pixel_sampled_z']:.5f}/pixel); mean-z: {metrics['recon_nll_mean_z']:.2f}")
            print(f"KL total: {metrics['kl_total']:.2f}; active latents (KL>{args.kl_thr}): {metrics['n_active_latents']}")
            print("per-latent KL, ordered by posterior variance ascending:")
            print("  " + "  ".join(f"z{m}:{kl[m]:.3f}" for m in order))

    json.dump(all_metrics, open(out / "summary_all_runs.json", "w"), indent=2)
    with open(out / "summary_all_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "seed", "recon_nll_sampled_z", "recon_nll_mean_z", "kl_total",
                    "n_active_latents", "train_seconds"] + [f"kl_z{m}" for m in range(10)])
        for r in all_metrics:
            w.writerow([r["beta"], r["seed"], r["recon_nll_sampled_z"], r["recon_nll_mean_z"],
                        r["kl_total"], r["n_active_latents"], r["train_seconds"]] + r["kl_per_latent"])
    print("\n== mean +- std over seeds ==")
    for beta in args.betas:
        rs = [r for r in all_metrics if r["beta"] == beta]
        for key in ["recon_nll_sampled_z", "kl_total", "n_active_latents"]:
            v = np.array([r[key] for r in rs], dtype=float)
            print(f"beta={beta:g} {key}: {v.mean():.3f} +- {v.std():.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
```

Outputs per run in `runs/beta{β}_seed{s}/`: `train_log.csv` (minibatch loss, reconstruction, KL, per-latent KL every `--log_every` steps), `metrics.json`, `model.pt`, and `fig7.png`. Across runs you also get `summary_all_runs.csv` and `.json`. All plotted numbers (variance, KL, response curves, traversals) are computed from the trained model. The script re-uses `model.pt` if it exists, so you can re-plot with a different `--kl_thr` or `--traverse_index` without retraining.

## 3. Reported quantities

- **Reconstruction loss:** the Bernoulli negative log-likelihood per image in nats, with z sampled from q(z|x). Also given per pixel, and with z = μ.
- **KL:** per-latent KL(q(z_m|x) || N(0,1)) averaged over all 737,280 images, plus the total and the number of latents above `--kl_thr`.
- **Also included:** the β-weighted training objective and the β=1 negative ELBO, so the two models can be compared on a common scale.
- **Expected pattern:** you should be able to test the paper's claim (Sec. 4.2) that the disentangled model has worse reconstruction and a lower total KL, and fewer active latents. Do not assume it. Read it from your logs.

## 4. Runtime, and deviations from the paper

### Runtime

I have not run this script, so the figures below are estimates, not measurements.

- The model has about 14.2M parameters. At batch size 64 one training step is roughly 5 GFLOP of compute, which is small for an M4 Pro GPU. Step time will be dominated by per-step MPS overhead, so I would expect a few milliseconds per step.
- That puts a 300k-step run at roughly 15 to 40 minutes per model, plus a few seconds for full-dataset evaluation and plotting.
- Total time is about (2 values of β) x (`--num_seeds`) x (time per run).
- Do a 3000-step smoke test first, then read `elapsed_s` in `train_log.csv` and extrapolate.
- Memory is not a concern. The dataset takes about 3 GB as uint8 on the device, and the script deletes the host copy of the images after the transfer.

### Deviations and my choices

1. **Batch size 64 and 300k steps** are my choices. The paper gives neither. That is about 26 epochs, and the first thing to vary if the figure does not resemble the paper's.
2. **Decoder output** uses tanh on the three hidden layers and Bernoulli logits (sigmoid) at the output. This is my reading of the ambiguous Table 1 entry.
3. **Initialisation, Adagrad settings, and posterior parameterisation** are PyTorch defaults and a log-variance head. The paper does not specify them.
4. **Training and evaluation use the full dataset**, with no held-out split.
5. **"Learnt variance"** is the dataset-mean of the posterior variance per latent.
6. **Unused latents** are those with KL below `--kl_thr` (default 0.1 nats, my choice). One unused column, the one with the highest variance, is shown.
7. **Traversal.** One random but fixed seed image is used, with 5 values from +3 to -3 and the other latents fixed at the inferred μ. The same image is used for all runs.
8. **Response rows** use the posterior mean μ. The position heatmap has rows = y and columns = x. I use the `jet` colormap, which is my choice.
9. **No semantic labels** (X, Y, scale, rotation) are added to columns. Column titles show the original latent index, which will not match the paper's indices because the initialisation differs. Judge the semantics from your own plots.
10. **Seeds and selection.** With `--num_seeds > 1` every seed's figure is saved, with no best-of selection. The paper's panel is presumably a single model. The paper's own metric protocol (A.4) discards the bottom 50% of 30 scores to control for diverged runs, which suggests seed variance matters.
11. **Numerics.** Everything is float32. MPS is not bitwise deterministic, so a fixed seed will not reproduce identical runs. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set so any unsupported op falls back to CPU.
12. **Not implemented:** the paper's disentanglement metric (Sec. 3) and the other panels (7C and 7D).

Whether your β=4 run shows the clean position, scale and rotation latents of Fig. 7A depends on your training length and seed. Check across seeds before drawing conclusions for your discussion.


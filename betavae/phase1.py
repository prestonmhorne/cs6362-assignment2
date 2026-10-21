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

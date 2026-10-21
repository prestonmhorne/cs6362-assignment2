#!/usr/bin/env python3
"""
Figure 1 of Zhang et al. (ICLR 2017), written from the paper text only.

  python zhang_fig1.py bench
  python zhang_fig1.py train --grid smoke      # plumbing check only
  python zhang_fig1.py train --grid reduced    # recommended
  python zhang_fig1.py train --grid full       # paper-like grid
  python zhang_fig1.py plot
Plotted numbers come only from <out>/runs/*/log.npz and meta.json.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, csv, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED_DATA_OFFSET = 10_000
LR_DECAY, MOMENTUM = 0.95, 0.9
WINDOW, CHECK_EVERY = 200, 100          # WINDOW must be a multiple of CHECK_EVERY
NUM_CLASSES = 10
P_FULL = [round(0.1 * i, 1) for i in range(11)]
P_REDUCED = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# ------------------------------------------------------------------ device
def get_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------ models
class ConvModule(nn.Module):
    """conv (no bias) -> BatchNorm (shift only, scale fixed to 1) -> ReLU."""
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.bn.weight.requires_grad_(False)   # inferred from Table 1 parameter count

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionModule(nn.Module):
    def __init__(self, cin, c1, c3):
        super().__init__()
        self.b1, self.b3 = ConvModule(cin, c1, 1, 1), ConvModule(cin, c3, 3, 1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x)], 1)


class DownsampleModule(nn.Module):
    def __init__(self, cin, c3):
        super().__init__()
        self.conv = ConvModule(cin, c3, 3, 2)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        return torch.cat([self.conv(x), self.pool(x)], 1)


class SmallInception(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvModule(3, 96, 3, 1),
            InceptionModule(96, 32, 32),      # 64 ch
            InceptionModule(64, 32, 48),      # 80
            DownsampleModule(80, 80),         # 160, 14x14
            InceptionModule(160, 112, 48),    # 160
            InceptionModule(160, 96, 64),     # 160
            InceptionModule(160, 80, 80),     # 160
            InceptionModule(160, 48, 96),     # 144
            DownsampleModule(144, 96),        # 240, 7x7
            InceptionModule(240, 176, 160),   # 336
            InceptionModule(336, 176, 160),   # 336
            nn.AvgPool2d(7),                  # global mean pooling
        )
        self.fc = nn.Linear(336, NUM_CLASSES)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class LRN(nn.Module):
    """Cross-channel LRN: x / (k + alpha * sum_window x^2)^beta (own impl, MPS-safe)."""
    def __init__(self, size=9, alpha=1e-3 / 9, beta=0.75, k=1.0):
        super().__init__()
        self.size, self.alpha, self.beta, self.k = size, alpha, beta, k
        self.register_buffer("kernel", torch.ones(1, 1, size))

    def forward(self, x):
        n, c, h, w = x.shape
        sq = (x * x).permute(0, 2, 3, 1).reshape(-1, 1, c)
        s = F.conv1d(sq, self.kernel, padding=self.size // 2)
        s = s.reshape(n, h, w, c).permute(0, 3, 1, 2)
        return x / (self.k + self.alpha * s).pow(self.beta)


class SmallAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2, 1), LRN(),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2, 1), LRN(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 384), nn.ReLU(),
            nn.Linear(384, 192), nn.ReLU(), nn.Linear(192, NUM_CLASSES))

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    def __init__(self, depth=1, width=512):
        super().__init__()
        layers, d = [nn.Flatten()], 3 * 28 * 28
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


ARCH = {  # lr from the paper; parameter counts from Table 1 (asserted below)
    "inception": dict(ctor=SmallInception, lr=0.1, params=1_649_402),
    "alexnet": dict(ctor=SmallAlexNet, lr=0.01, params=1_387_786),
    "mlp1x512": dict(ctor=lambda: MLP(1, 512), lr=0.01, params=1_209_866),
}


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(arch):
    model = ARCH[arch]["ctor"]()
    model.apply(init_weights)
    n = count_params(model)
    assert n == ARCH[arch]["params"], f"{arch}: {n} params, Table 1 says {ARCH[arch]['params']}"
    return model


# -------------------------------------------------------------------- data
def per_image_standardization(x):
    n = x[0].numel()
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    return (x - mean) / torch.clamp(std, min=1.0 / math.sqrt(n))


def load_base(args):
    from torchvision.datasets import CIFAR10   # only used to download/decode raw data
    tr = CIFAR10(args.data_root, train=True, download=True)
    te = CIFAR10(args.data_root, train=False, download=True)

    def prep(arr):
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
        return per_image_standardization(x[:, :, 2:30, 2:30]).contiguous()

    xtr, ytr = prep(tr.data), torch.tensor(tr.targets, dtype=torch.long)
    xte, yte = prep(te.data), torch.tensor(te.targets, dtype=torch.long)
    if args.train_subset:
        xtr, ytr = xtr[:args.train_subset], ytr[:args.train_subset]
    return xtr, ytr, xte, yte


def _apply_fixed(x, perm, over):
    b = x.shape[0]
    if over == "spatial":
        return x.reshape(b, 3, -1)[:, :, perm].reshape(x.shape)
    return x.reshape(b, -1)[:, perm].reshape(x.shape)


def _per_image_permute(x, g, over, chunk=5000):
    out = torch.empty_like(x)
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]
        b = xb.shape[0]
        if over == "spatial":
            flat = xb.reshape(b, 3, -1)
            perm = torch.rand(b, flat.shape[2], generator=g).argsort(dim=1)
            flat = flat.gather(2, perm.unsqueeze(1).expand(-1, 3, -1))
        else:
            flat = xb.reshape(b, -1)
            perm = torch.rand(b, flat.shape[1], generator=g).argsort(dim=1)
            flat = flat.gather(1, perm)
        out[s:s + chunk] = flat.reshape(xb.shape)
    return out


def build_data(base, setting, p, args):
    xtr, ytr, xte, yte = base
    g = torch.Generator().manual_seed(SEED_DATA_OFFSET + args.seed)
    over = args.perm_over
    if setting == "labels":                      # p=0 true, p=1 random labels
        if p > 0:
            ytr = ytr.clone()
            flip = torch.rand(ytr.shape[0], generator=g) < p
            rnd = torch.randint(0, NUM_CLASSES, (ytr.shape[0],), generator=g)
            ytr[flip] = rnd[flip]                # test labels stay clean
    elif setting == "shuffled_pixels":
        perm = torch.randperm(784 if over == "spatial" else 3 * 784, generator=g)
        xtr, xte = _apply_fixed(xtr, perm, over), _apply_fixed(xte, perm, over)
    elif setting == "random_pixels":
        xtr, xte = _per_image_permute(xtr, g, over), _per_image_permute(xte, g, over)
    elif setting == "gaussian":
        mu = xtr.mean(dim=(0, 2, 3), keepdim=True)
        sd = xtr.std(dim=(0, 2, 3), keepdim=True)
        xtr = mu + sd * torch.randn(xtr.shape, generator=g)
        xte = mu + sd * torch.randn(xte.shape, generator=g)
    else:
        raise ValueError(setting)
    return xtr, ytr, xte, yte


# ---------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    tot, wrong = 0.0, 0
    for i in range(0, x.shape[0], bs):
        out = model(x[i:i + bs])
        tot += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        wrong += (out.argmax(1) != y[i:i + bs]).sum().item()
    model.train()
    return tot / x.shape[0], wrong / x.shape[0]


def full_eval(model, xtr, ytr, xte, yte, step):
    trl, tre = evaluate(model, xtr, ytr)
    tel, tee = evaluate(model, xte, yte)
    return dict(step=step, train_loss=trl, train_err=tre, test_loss=tel, test_err=tee)


def first_crossing(loss, tau, window=WINDOW, every=CHECK_EVERY):
    csum = np.concatenate([[0.0], np.cumsum(loss, dtype=np.float64)])
    for t in range(window, len(loss) + 1, every):
        if (csum[t] - csum[t - window]) / window < tau:
            return t
    return None


def run_one(job, base, args, dev, out_dir):
    arch, setting, p = job["arch"], job["setting"], job["p"]
    run_id = f"{arch}__{setting}__p{p:.2f}__seed{args.seed}"
    run_dir = out_dir / "runs" / run_id
    if (run_dir / "meta.json").exists():
        print(f"  [skip] {run_id} already done")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    g_batch = torch.Generator().manual_seed(args.seed)
    xtr, ytr, xte, yte = [t.to(dev) for t in build_data(base, setting, p, args)]
    n, bs = xtr.shape[0], args.batch_size
    spe = n // bs                                   # drop last partial batch
    model = build_model(arch).to(dev)
    lr0 = ARCH[arch]["lr"]
    opt = torch.optim.SGD([q for q in model.parameters() if q.requires_grad],
                          lr=lr0, momentum=MOMENTUM, weight_decay=0.0)

    losses = torch.zeros(args.max_steps, device=dev)
    accs = torch.zeros(args.max_steps, device=dev)
    step, epoch, done = 0, 0, False
    first_cross, snap, stop_at, evals = None, None, None, []
    t_start = time.time()
    model.train()
    while not done:
        for gp in opt.param_groups:
            gp["lr"] = lr0 * (LR_DECAY ** epoch)    # staircase, per epoch
        perm = torch.randperm(n, generator=g_batch).to(dev)
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            xb, yb = xtr[idx], ytr[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses[step] = loss.detach()
            accs[step] = (logits.argmax(1) == yb).float().mean()
            step += 1

            if step % CHECK_EVERY == 0 and step >= WINDOW:
                m = losses[step - WINDOW:step].mean().item()
                if step % 1000 == 0:
                    print(f"    step {step:6d}  win-loss {m:.4f}  lr {lr0 * LR_DECAY ** epoch:.5f}"
                          f"  {time.time() - t_start:7.0f}s", flush=True)
                if first_cross is None and m < args.tau:
                    first_cross = step
                    snap = full_eval(model, xtr, ytr, xte, yte, step)
                    if args.early:
                        stop_at = step + args.tail
                if stop_at is not None and step >= stop_at:
                    done = True
                    break
            if args.eval_every and step % args.eval_every == 0:
                tl, te = evaluate(model, xte, yte)
                evals.append(dict(step=step, test_loss=tl, test_err=te))
            if step >= args.max_steps:
                done = True
                break
        epoch += 1

    final = full_eval(model, xtr, ytr, xte, yte, step)
    np.savez_compressed(run_dir / "log.npz",
                        loss=losses[:step].cpu().numpy(), acc=accs[:step].cpu().numpy())
    meta = dict(run_id=run_id, arch=arch, setting=setting, p=p, seed=args.seed,
                n_params=count_params(model), steps_run=step, steps_per_epoch=spe,
                first_cross_step=first_cross, snapshot=snap, final=final, evals=evals,
                wall_seconds=time.time() - t_start, torch=torch.__version__, device=str(dev),
                config=dict(batch_size=bs, tau=args.tau, window=WINDOW, check_every=CHECK_EVERY,
                            max_steps=args.max_steps, early_stop=args.early, tail=args.tail,
                            lr0=lr0, lr_decay=LR_DECAY, momentum=MOMENTUM,
                            perm_over=args.perm_over, train_subset=args.train_subset))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  done {run_id}: steps={step} cross={first_cross} "
          f"train_err={final['train_err']:.4f} test_err={final['test_err']:.4f}")
    del xtr, ytr, xte, yte, model, opt
    if dev.type == "mps":
        torch.mps.empty_cache()


def resolve_args(args):
    smoke = args.grid == "smoke"
    if args.max_steps is None:
        args.max_steps = 600 if smoke else 25000
    if args.tau is None:
        args.tau = 0.3 if smoke else 0.05
    if smoke and not args.train_subset:
        args.train_subset = 2000
    args.early = {"on": True, "off": False, "auto": args.grid != "full"}[args.early_stop]


def make_jobs(args):
    jobs = []
    for arch in [a for a in ("mlp1x512", "alexnet", "inception") if a in args.archs]:
        if args.grid == "smoke":
            ps = [0.0, 1.0]
        elif args.grid == "reduced":
            ps = P_REDUCED if arch == "inception" else P_FULL
        else:
            ps = P_FULL
        jobs += [dict(arch=arch, setting="labels", p=p) for p in ps]
        if arch == "inception":
            jobs += [dict(arch=arch, setting=s, p=0.0)
                     for s in ("shuffled_pixels", "random_pixels", "gaussian")]
    return jobs


def cmd_train(args):
    resolve_args(args)
    dev = get_device(args.device)
    out = Path(args.out)
    base = load_base(args)
    jobs = make_jobs(args)
    print(f"device={dev} jobs={len(jobs)} max_steps={args.max_steps} tau={args.tau} "
          f"early_stop={args.early} batch={args.batch_size}")
    for k, job in enumerate(jobs):
        print(f"[{k + 1}/{len(jobs)}] {job}")
        run_one(job, base, args, dev, out)


def cmd_bench(args):
    dev = get_device(args.device)
    for arch in args.archs:
        model = build_model(arch).to(dev)
        opt = torch.optim.SGD([q for q in model.parameters() if q.requires_grad],
                              lr=ARCH[arch]["lr"], momentum=MOMENTUM)
        x = torch.randn(args.batch_size, 3, 28, 28, device=dev)
        y = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=dev)
        for i in range(args.warmup + args.iters):
            if i == args.warmup:
                sync(dev)
                t0 = time.time()
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sync(dev)
        ms = (time.time() - t0) / args.iters * 1000
        print(f"{arch:10s} {ms:7.1f} ms/step  -> 25k steps = {ms * 25000 / 60000:6.1f} min")


# ---------------------------------------------------------------- plotting
def rolling_mean(x, w):
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    t = np.arange(1, len(x) + 1)
    lo = np.maximum(t - w, 0)
    return (c[t] - c[lo]) / (t - lo)


def draw_series(ax, xs, ys, censored, label, color, marker):
    ax.plot(xs, ys, color=color, lw=1.2, label=label)
    for x, y, c in zip(xs, ys, censored):
        ax.plot([x], [y], marker=marker, ms=6, color=color, ls="none",
                mfc="none" if c else color)


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    runs = {}
    for mp in sorted((out / "runs").glob("*/meta.json")):
        meta = json.loads(mp.read_text())
        if meta["seed"] != args.seed:
            continue
        loss = np.load(mp.parent / "log.npz")["loss"]
        runs[(meta["arch"], meta["setting"], round(meta["p"], 2))] = (meta, loss)
    if not runs:
        raise SystemExit("no finished runs found")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # (a) learning curves
    series = [("true labels", ("labels", 0.0), "tab:blue", "s"),
              ("random labels", ("labels", 1.0), "tab:red", "o"),
              ("shuffled pixels", ("shuffled_pixels", 0.0), "tab:green", "*"),
              ("random pixels", ("random_pixels", 0.0), "m", None),
              ("gaussian", ("gaussian", 0.0), "k", "D")]
    ax = axes[0]
    for name, key, color, marker in series:
        k = ("inception",) + key
        if k not in runs:
            print(f"panel (a): missing run for {name}")
            continue
        loss = runs[k][1]
        y = rolling_mean(loss, args.plot_window)
        x = np.arange(1, len(loss) + 1) / 1000.0
        ax.plot(x, y, color=color, marker=marker, markevery=max(1, len(x) // 25),
                ms=4, lw=1.2, label=name)
    ax.set_xlabel("thousand steps")
    ax.set_ylabel(f"average_loss (rolling mean, {args.plot_window} steps)")
    ax.set_title("(a) learning curves")
    ax.legend()

    # (b), (c)
    styles = {"inception": ("Inception", "k", "s"), "alexnet": ("AlexNet", "g", "o"),
              "mlp1x512": ("MLP 1x512", "b", "*")}
    rows = []
    for arch, (label, color, marker) in styles.items():
        items = sorted(((k[2], v) for k, v in runs.items() if k[0] == arch and k[1] == "labels"),
                       key=lambda t: t[0])
        if not items:
            continue
        tau = args.tau if args.tau is not None else items[0][1][0]["config"]["tau"]
        t0 = next((first_crossing(l, tau) for p, (m, l) in items if p == 0.0), None)
        if t0 is None:
            print(f"{arch}: p=0 run never reached tau={tau}; cannot normalise panel (b)")
        pb, rb, cb, pc, ec, cc = [], [], [], [], [], []
        for p, (meta, loss) in items:
            tc = first_crossing(loss, tau)
            use = meta["snapshot"] if meta["snapshot"] else meta["final"]
            rel = None if t0 is None else (tc if tc is not None else len(loss)) / t0
            if rel is not None:
                pb.append(p); rb.append(rel); cb.append(tc is None)
            pc.append(p); ec.append(use["test_err"]); cc.append(meta["snapshot"] is None)
            rows.append(dict(arch=arch, p=p, steps_run=meta["steps_run"], cross_step=tc,
                             rel_time=rel, censored_b=tc is None, test_err=use["test_err"],
                             train_err=use["train_err"], converged_c=meta["snapshot"] is not None))
        if pb:
            draw_series(axes[1], pb, rb, cb, label, color, marker)
        draw_series(axes[2], pc, ec, cc, label, color, marker)

    axes[1].set_xlabel("label corruption"); axes[1].set_ylabel("time to overfit (relative to p=0)")
    axes[1].set_title("(b) convergence slowdown"); axes[1].legend()
    axes[2].set_xlabel("label corruption"); axes[2].set_ylabel("test_error")
    axes[2].set_title("(c) generalization error growth"); axes[2].legend()
    if not args.no_ref_line:   # analytic reference (1 - 1/classes), not a measurement
        axes[2].axhline(1 - 1 / NUM_CLASSES, color="r", ls="--", lw=1)
    fig.text(0.5, -0.02, "Hollow markers: run did not reach the convergence threshold "
             "(b: lower bound; c: final-step values).", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure1.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "figure1.pdf", bbox_inches="tight")
    with open(out / "figure1_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)
    print(f"wrote {out / 'figure1.png'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default="fig1_runs")
        p.add_argument("--device", default="auto")
        p.add_argument("--data_root", default="./data")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument("--archs", nargs="+", default=list(ARCH), choices=list(ARCH))

    pb = sub.add_parser("bench"); common(pb)
    pb.add_argument("--warmup", type=int, default=20)
    pb.add_argument("--iters", type=int, default=100)

    pt = sub.add_parser("train"); common(pt)
    pt.add_argument("--grid", choices=["smoke", "reduced", "full"], default="reduced")
    pt.add_argument("--max_steps", type=int, default=None)
    pt.add_argument("--tau", type=float, default=None)
    pt.add_argument("--early_stop", choices=["auto", "on", "off"], default="auto")
    pt.add_argument("--tail", type=int, default=1000)
    pt.add_argument("--eval_every", type=int, default=2500)
    pt.add_argument("--train_subset", type=int, default=0)
    pt.add_argument("--perm_over", choices=["spatial", "all"], default="spatial")

    pp = sub.add_parser("plot"); common(pp)
    pp.add_argument("--tau", type=float, default=None)
    pp.add_argument("--plot_window", type=int, default=WINDOW)
    pp.add_argument("--no_ref_line", action="store_true")

    args = ap.parse_args()
    {"bench": cmd_bench, "train": cmd_train, "plot": cmd_plot}[args.cmd](args)


if __name__ == "__main__":
    main()

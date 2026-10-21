PDF
I've attached the paper "Understanding deep learning requires rethinking generalization" (Zhang et al., ICLR 2017). I want to reproduce Figure 1 (panels a, b, and c) from scratch in PyTorch. Please do not search the web or use any existing implementation — work only from the paper.
First, summarize the exact experimental setup the paper specifies for Figure 1: dataset preprocessing, architectures (small Inception, AlexNet, MLP 1x512), optimizer, learning-rate schedule, batch size, number of steps, and how each data modification (true labels, random labels, shuffled pixels, random pixels, Gaussian, partially corrupted labels with probability p) is constructed. Point out anything the paper leaves unspecified.
Then write a complete, runnable PyTorch implementation (a single Python script or notebook cells) that produces all three panels:
(a) training loss vs. thousand steps for the Inception model under the five settings,
(b) time-to-overfit (relative convergence time) vs. label corruption for the three architectures,
(c) test error vs. label corruption for the three architectures.
I'm running on an Apple M4 Pro laptop (PyTorch MPS backend, 24 GB RAM), so tell me the expected runtime and, if needed, propose a reduced grid (fewer corruption levels or architectures) — but clearly list every deviation from the paper.
Do not hardcode or fabricate any values in the plots; every number must come from actual training runs logged to disk.

# Reproducing Figure 1 of Zhang et al. (2017) in PyTorch

I worked only from the attached paper and did not search the web. I have not executed the script below, because I can't run code in this conversation. The first things to check when you run it are the parameter-count assertions and the `bench` timings.

## 1. What the paper specifies for Figure 1, and what it leaves out

**Specified**

| Item | Paper's statement (location) |
|---|---|
| Data | CIFAR10, 50,000 train and 10,000 "validation" images, 10 classes. The paper uses the held-out set as the test set (Appendix A). |
| Preprocessing | Divide by 255, center-crop to 28x28, then per-image whitening (TensorFlow `per_image_whitening`: subtract each image's own mean, divide by its own adjusted std). Appendix A. |
| Regularization | None. The paper trains with no weight decay, dropout or augmentation unless stated (Appendix A). |
| Small Inception | Figure 3. Conv module = conv, BatchNorm, ReLU. Inception module = 1x1 and 3x3 conv branches (stride 1) concatenated. Downsample module = 3x3 stride-2 conv concatenated with 3x3 stride-2 max-pool. Stack: Conv 96, Inc 32+32, Inc 32+48, Down 80, Inc 112+48, Inc 96+64, Inc 80+80, Inc 48+96, Down 96, Inc 176+160, Inc 176+160, global 7x7 mean pool, FC 10. |
| Small AlexNet | Two (conv 5x5, max-pool 3x3, LRN) modules, then FC 384, FC 192, linear 10. |
| MLP 1x512 | One hidden layer of 512 ReLU units, then a 10-way linear layer. |
| Optimizer | SGD, momentum 0.9. Initial LR 0.1 (Inception) or 0.01 (AlexNet, MLP). Decay factor 0.95 per training epoch. |
| Steps | The x-axis of 1(a) runs to 25 thousand steps. |
| True labels | Unmodified dataset. |
| Partially corrupted labels | With probability p, independently per image, the label is replaced by a uniformly random class. |
| Random labels | All labels replaced by random ones (p = 1). They are fixed and identical across epochs (Appendix A). |
| Shuffled pixels | One random pixel permutation, applied to every image in both train and test. |
| Random pixels | A different random permutation per image. |
| Gaussian | Pixels drawn from a Gaussian with the dataset's mean and variance, per image. |
| Panels (b), (c) | p runs from 0 to 1 and the training set is always fit perfectly. (b) is convergence time relative to p = 0. (c) is test error, which equals generalization error because training error is 0, and approaches 90% as p goes to 1. |

**Inferences from Table 1.** I checked the architecture details against the parameter counts in Table 1, and all three can be matched exactly by hand:

- **Inception, 1,649,402 parameters.** This is reproduced exactly (1,647,690 weights and FC bias, plus 1,712 BatchNorm shifts) only if convs have no bias and BatchNorm has a shift but no learnable scale. The script asserts this count.
- **AlexNet, 1,387,786 parameters.** This matches exactly with 64 filters in both conv layers, SAME padding, pool stride 2, so a 7x7x64 flatten.
- **MLP 1x512, 1,209,866 parameters.** Matches exactly. MLP 3x512 (1,735,178) matches too.

**Not specified, so I had to choose**

- Batch size, and therefore steps per epoch. This matters because LR decay is per epoch.
- The loss (cross-entropy is inferred, since the initial loss on the axis is about ln 10), weight initialization, and BatchNorm epsilon and momentum.
- LRN hyperparameters, pooling padding and strides.
- What "average_loss" in (a) averages over.
- The convergence criterion behind "time to overfit" in (b), and the number of steps used in (b) and (c).
- The corruption grid (the markers look roughly evenly spaced, but this is not stated) and the number of seeds.
- Pixel-permutation details: spatial positions only or all channel values, per-channel or global Gaussian statistics, and whether the modification comes before or after whitening.
- Whether training error is measured in train or eval mode.

## 2. Implementation

The script has three subcommands: `bench` (timing), `train`, and `plot`.

- **Logging.** Every training run writes per-step loss and accuracy to `runs/<id>/log.npz` and a `meta.json`. The plotting step only reads these files.
- **Hardcoded values.** The only constant drawn without a training run is the dashed 0.9 line in (c), a reference computed as 1 - 1/10 and disabled with `--no_ref_line`.
- **Convergence rule.** A run "converges" when the mean minibatch loss over the last 200 steps drops below `tau` (checked every 100 steps). At that moment the script snapshots full-train-set and test errors.
- **Censored runs.** Runs that never converge are drawn with hollow markers. In (b) the hollow value is a lower bound.
- **Shared runs.** Inception with true labels and with random labels serve both as p = 0 and p = 1 in (b) and (c) and as two of the curves in (a).
- **Label corruption.** All corruption levels use the same random draw, so the corrupted sets are nested across p.
- **Data source.** `torchvision` is used only to download and decode CIFAR10. No model or training code comes from it.

```python
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
```

**Suggested workflow**

1. `python zhang_fig1.py bench` to measure real ms/step on your machine.
2. `python zhang_fig1.py train --grid smoke --out smoke_runs`, then `plot --out smoke_runs`, to check that everything runs end to end. The numbers from this step are meaningless: it uses 2,000 images and 600 steps.
3. `python zhang_fig1.py train --grid reduced`, then `python zhang_fig1.py plot`.

Runs are resumable, since finished runs are skipped. A run interrupted mid-training restarts from scratch.

**Sanity check without plotting targets.** Table 1 gives test accuracies for the no-regularization rows: about 85.75% (Inception), 76.07% (AlexNet) and 50.51% (MLP 1x512). Your p = 0 runs should land in that neighborhood. If they are far off, suspect a bug before suspecting the setup.

## 3. Runtime on an M4 Pro, and a reduced grid

These are FLOP-count estimates, not measurements, so run `bench` first. Small Inception is about 0.2 GMAC per image forward, so roughly 150 GFLOP per step at batch 128.

| Model | Estimated step time | 25k steps |
|---|---|---|
| Small Inception | 60-150 ms | 25-60 min |
| Small AlexNet | 8-20 ms | 3-9 min |
| MLP 1x512 | 2-6 ms | 1-3 min |

- **Full grid.** This is 14 Inception runs (11 corruption levels plus 3 pixel settings) at 25k steps each, plus 11 runs each for AlexNet and MLP. Estimated total is about 7-16 hours.
- **Reduced grid (default).** This takes about 2-4.5 hours, mostly Inception. It rests on two things:
  - **Inception grid.** 9 runs: p in {0, 0.2, 0.4, 0.6, 0.8, 1.0} plus the 3 pixel settings.
  - **Early stopping.** Each run stops 1,000 steps after convergence. My step counts come from reading Figures 1(a) and 1(b), so treat them as rough.
- **Memory.** About 0.6 GB for the data on-device plus a couple of GB of activations, well within 24 GB.

## Every deviation from the paper

- **Batch size 128** with sampling without replacement and the last partial batch dropped. The paper does not specify it, and because the LR decay is per epoch, it affects the schedule.
- **Reduced corruption grid for Inception.** The reduced grid uses p in {0, 0.2, ..., 1} instead of a finer grid. AlexNet and MLP keep p in steps of 0.1.
- **One seed per configuration**, no error bars. MPS is not bitwise deterministic.
- **Early stopping in the reduced grid.** Curves in (a) end 1,000 steps after convergence instead of running to 25k. Use `--early_stop off` for full-length curves. The `full` grid runs to the full step count (and turns early stopping off).
- **Convergence definition (mine).** The rolling mean minibatch loss over 200 steps falls below `tau = 0.05`, checked every 100 steps. Panel (b) is T(p)/T(0) per architecture under that rule, and the plot step accepts `--tau` to change it from the logs. Panel (c) uses a snapshot taken at the first crossing, so it is tied to the `tau` used in training.
- **Loss in (a)** is a rolling mean of minibatch training loss in train mode, not a full-pass loss.
- **Eval-mode errors.** Train and test errors are computed in eval mode with BatchNorm running statistics, which can differ slightly from train-mode behavior.
- **Architecture details inferred.** BatchNorm shift-only and no conv biases in Inception, 64-filter AlexNet convs, and LRN settings (size 9, alpha 0.001/9, beta 0.75, k 1) are inferred or chosen. The first two are pinned by Table 1's parameter counts.
- **Initialization.** Glorot uniform for convs and linear layers, with PyTorch default BatchNorm epsilon and momentum.
- **Data modifications.**
  - Pixel permutations act on the 784 spatial positions, with channels moving together (`--perm_over all` permutes all 2,352 values).
  - Gaussian noise uses per-channel mean and std of the whitened training set, and is generated after whitening, so it is not itself per-image whitened.
  - All three input modifications are applied to both train and test inputs and keep the original labels.
- **Hardware and precision.** Single-device fp32 on MPS.
- **Test set.** The official 10,000 CIFAR10 test images, which the paper calls "validation."
- **Reference line.** The dashed line at 0.9 in (c) is computed as 1 - 1/10 and is not a training result.

If the MLP or AlexNet random-label runs do not reach `tau` within 25k steps (the 0.95 per-epoch decay shrinks the LR by roughly 25x by then), they will appear as hollow markers rather than being hidden. Raising `--max_steps` is the first thing to try in that case.

I ran your script. Results so far:
- `bench` passed all three parameter-count assertions. Timings on my M4 Pro (MPS, batch 128): Inception 81.4 ms/step (~34 min per 25k steps), AlexNet 15.4 ms/step, MLP 0.8 ms/step.
- `--grid smoke` ran end to end: all 9 runs reached 0 train error and the plot was written.

Reviewing the code against the paper, I want these changes:

1. Panel (a): the reduced grid's early stopping cuts the learning curves short, but the paper's curves run the full x-axis (~25k steps). Run the five Inception panel-(a) settings (true labels, random labels, shuffled pixels, random pixels, gaussian) to the full 25k steps without early stopping. Keep early stopping for the other corruption levels, which only feed (b) and (c).
2. Gaussian setting: the paper says the noise matches "the mean and variance of the original image dataset". You sampled in whitened space using per-channel stats of the whitened train set. Instead, sample in raw [0,1] pixel space using the original dataset's per-channel mean and std, then apply the same center crop and per-image whitening as the real images.
3. Panel (b): make the plot step able to redraw (b) at several thresholds (tau = 0.02, 0.05, 0.1) from the saved logs, so I can show the result doesn't hinge on the threshold choice.
4. Add a short `--grid` option or flag that runs only the five panel-(a) Inception settings, so I can schedule them separately.

Keep everything else unchanged, and give me the full updated script (not a diff), plus a short list of exactly what changed.

I haven't run this version. The Gaussian and threshold-sweep paths are new code, so do a fresh smoke run first (see "Before you run it").

```python
#!/usr/bin/env python3
"""
Figure 1 of Zhang et al. (ICLR 2017), written from the paper text only.

  python zhang_fig1.py bench
  python zhang_fig1.py train --grid smoke      # plumbing check only
  python zhang_fig1.py train --grid panel_a    # only the five Inception panel-(a) runs, full length
  python zhang_fig1.py train --grid reduced    # recommended
  python zhang_fig1.py train --grid full       # paper-like grid
  python zhang_fig1.py plot
  python zhang_fig1.py plot --taus 0.02 0.05 0.1   # also redraw panel (b) at several thresholds
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


def center_crop_whiten(x):
    """32x32 raw [0,1] images -> center 28x28 crop -> per-image whitening."""
    return per_image_standardization(x[:, :, 2:30, 2:30]).contiguous()


def load_base(args):
    from torchvision.datasets import CIFAR10   # only used to download/decode raw data
    tr = CIFAR10(args.data_root, train=True, download=True)
    te = CIFAR10(args.data_root, train=False, download=True)

    def to_raw(arr):
        return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)

    raw_tr, raw_te = to_raw(tr.data), to_raw(te.data)
    # Per-channel mean/std of the original (raw [0,1], uncropped 32x32) training images.
    # Computed on the full 50k training set, before any --train_subset.
    stats = dict(mean=raw_tr.mean(dim=(0, 2, 3), keepdim=True),
                 std=raw_tr.std(dim=(0, 2, 3), keepdim=True, unbiased=False))
    xtr, ytr = center_crop_whiten(raw_tr), torch.tensor(tr.targets, dtype=torch.long)
    xte, yte = center_crop_whiten(raw_te), torch.tensor(te.targets, dtype=torch.long)
    del raw_tr, raw_te
    if args.train_subset:
        xtr, ytr = xtr[:args.train_subset], ytr[:args.train_subset]
    return xtr, ytr, xte, yte, stats


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
    xtr, ytr, xte, yte, stats = base
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
        # Sample in raw [0,1] pixel space with the original training set's per-channel
        # mean/std (no clipping), then apply the same center crop and per-image
        # whitening as the real images. Test inputs use the same (train) statistics.
        mu, sd = stats["mean"], stats["std"]
        raw_tr = mu + sd * torch.randn((xtr.shape[0], 3, 32, 32), generator=g)
        raw_te = mu + sd * torch.randn((xte.shape[0], 3, 32, 32), generator=g)
        xtr, xte = center_crop_whiten(raw_tr), center_crop_whiten(raw_te)
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
    # panel-(a) runs always go to max_steps; early stopping only applies to the other runs
    early = args.early and not job.get("panel_a", False)
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
                    if early:
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
                            max_steps=args.max_steps, early_stop=early, tail=args.tail,
                            lr0=lr0, lr_decay=LR_DECAY, momentum=MOMENTUM,
                            perm_over=args.perm_over, train_subset=args.train_subset,
                            gaussian="raw01_train_stats_then_crop_whiten"))
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
    args.early = {"on": True, "off": False,
                  "auto": args.grid not in ("full", "panel_a")}[args.early_stop]


def panel_a_jobs():
    """The five Inception settings drawn in panel (a). Also the p=0 / p=1 points of (b), (c)."""
    return [dict(arch="inception", setting="labels", p=0.0, panel_a=True),
            dict(arch="inception", setting="labels", p=1.0, panel_a=True),
            dict(arch="inception", setting="shuffled_pixels", p=0.0, panel_a=True),
            dict(arch="inception", setting="random_pixels", p=0.0, panel_a=True),
            dict(arch="inception", setting="gaussian", p=0.0, panel_a=True)]


def make_jobs(args):
    if args.grid == "panel_a":
        return panel_a_jobs()                   # ignores --archs
    jobs = []
    for arch in [a for a in ("mlp1x512", "alexnet", "inception") if a in args.archs]:
        if args.grid == "smoke":
            ps = [0.0, 1.0]
        elif args.grid == "reduced":
            ps = P_REDUCED if arch == "inception" else P_FULL
        else:
            ps = P_FULL
        jobs += [dict(arch=arch, setting="labels", p=p,
                      panel_a=(arch == "inception" and p in (0.0, 1.0))) for p in ps]
        if arch == "inception":
            jobs += [dict(arch=arch, setting=s, p=0.0, panel_a=True)
                     for s in ("shuffled_pixels", "random_pixels", "gaussian")]
    return jobs


def cmd_train(args):
    resolve_args(args)
    dev = get_device(args.device)
    out = Path(args.out)
    base = load_base(args)
    jobs = make_jobs(args)
    n_full = sum(1 for j in jobs if j.get("panel_a"))
    print(f"device={dev} jobs={len(jobs)} ({n_full} panel-(a) runs, never early-stopped) "
          f"max_steps={args.max_steps} tau={args.tau} "
          f"early_stop(other runs)={args.early} batch={args.batch_size}")
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


def arch_items(runs, arch):
    return sorted(((k[2], v) for k, v in runs.items() if k[0] == arch and k[1] == "labels"),
                  key=lambda t: t[0])


def panel_b_points(items, tau):
    """Panel (b) for one architecture at threshold tau, computed from saved loss logs only.
    Returns (t0, p_list, rel_time_list, censored_list, table_rows)."""
    t0 = next((first_crossing(l, tau) for p, (m, l) in items if p == 0.0), None)
    pb, rb, cb, table = [], [], [], []
    for p, (meta, loss) in items:
        tc = first_crossing(loss, tau)
        rel = None if t0 is None else (tc if tc is not None else len(loss)) / t0
        if rel is not None:
            pb.append(p); rb.append(rel); cb.append(tc is None)
        table.append(dict(p=p, steps_run=meta["steps_run"], cross_step=tc,
                          rel_time=rel, censored_b=tc is None))
    return t0, pb, rb, cb, table


def warn_if_censoring_possible(runs, taus):
    low = min(taus)
    n = sum(1 for (arch, setting, p), (m, _) in runs.items()
            if setting == "labels" and m["config"]["early_stop"] and m["config"]["tau"] > low)
    if n:
        print(f"note: {n} early-stopped runs were trained with tau > {low}; their logs may end "
              f"before the loss reaches {low}, so those points can appear censored (hollow). "
              f"To resolve them, delete those run folders and re-train with --tau {low} "
              f"(finished runs are skipped).")


def plot_b_sweep(plt, runs, styles, taus, out):
    fig, axs = plt.subplots(1, len(taus), figsize=(5.2 * len(taus), 4.4),
                            sharey=True, squeeze=False)
    rows = []
    for ax, tau in zip(axs[0], taus):
        for arch, (label, color, marker) in styles.items():
            items = arch_items(runs, arch)
            if not items:
                continue
            t0, pb, rb, cb, table = panel_b_points(items, tau)
            rows += [dict(tau=tau, arch=arch, **r) for r in table]
            if t0 is None:
                print(f"tau={tau} {arch}: p=0 run never reached the threshold; series skipped")
            elif pb:
                draw_series(ax, pb, rb, cb, label, color, marker)
        ax.set_title(f"(b) convergence slowdown, tau = {tau}")
        ax.set_xlabel("label corruption")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    axs[0][0].set_ylabel("time to overfit (relative to p=0)")
    fig.text(0.5, -0.02, "Hollow markers: run did not reach the threshold in its log "
             "(lower bound on the relative time).", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure1_b_thresholds.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "figure1_b_thresholds.pdf", bbox_inches="tight")
    if rows:
        with open(out / "figure1_b_thresholds.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"wrote {out / 'figure1_b_thresholds.png'}")


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
        items = arch_items(runs, arch)
        if not items:
            continue
        tau = args.tau if args.tau is not None else items[0][1][0]["config"]["tau"]
        t0, pb, rb, cb, btable = panel_b_points(items, tau)
        if t0 is None:
            print(f"{arch}: p=0 run never reached tau={tau}; cannot normalise panel (b)")
        pc, ec, cc = [], [], []
        for (p, (meta, loss)), brow in zip(items, btable):
            use = meta["snapshot"] if meta["snapshot"] else meta["final"]
            pc.append(p); ec.append(use["test_err"]); cc.append(meta["snapshot"] is None)
            rows.append(dict(arch=arch, p=p, steps_run=brow["steps_run"],
                             cross_step=brow["cross_step"], rel_time=brow["rel_time"],
                             censored_b=brow["censored_b"], test_err=use["test_err"],
                             train_err=use["train_err"],
                             converged_c=meta["snapshot"] is not None))
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

    if args.taus:
        warn_if_censoring_possible(runs, args.taus)
        plot_b_sweep(plt, runs, styles, args.taus, out)


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
    pt.add_argument("--grid", choices=["smoke", "reduced", "full", "panel_a"], default="reduced")
    pt.add_argument("--max_steps", type=int, default=None)
    pt.add_argument("--tau", type=float, default=None)
    pt.add_argument("--early_stop", choices=["auto", "on", "off"], default="auto")
    pt.add_argument("--tail", type=int, default=1000)
    pt.add_argument("--eval_every", type=int, default=2500)
    pt.add_argument("--train_subset", type=int, default=0)
    pt.add_argument("--perm_over", choices=["spatial", "all"], default="spatial")

    pp = sub.add_parser("plot"); common(pp)
    pp.add_argument("--tau", type=float, default=None)
    pp.add_argument("--taus", type=float, nargs="+", default=None,
                    help="also redraw panel (b) at these thresholds, e.g. --taus 0.02 0.05 0.1")
    pp.add_argument("--plot_window", type=int, default=WINDOW)
    pp.add_argument("--no_ref_line", action="store_true")

    args = ap.parse_args()
    {"bench": cmd_bench, "train": cmd_train, "plot": cmd_plot}[args.cmd](args)


if __name__ == "__main__":
    main()
```

## What changed

1. **Panel (a) runs go full length.**
   - Jobs now carry a `panel_a` flag: Inception with true labels (p=0), random labels (p=1), shuffled pixels, random pixels, and gaussian.
   - `run_one` never early-stops a flagged job, whatever `--early_stop` says. They run to `--max_steps` (25,000), and the first-crossing snapshot for (c) is still taken on the way.
   - All other jobs keep early stopping under the same rules as before.
   - `meta.json` now records the effective `early_stop` per run.
2. **Gaussian setting.**
   - `load_base` now computes per-channel mean and std from the raw [0,1], uncropped 32x32 training images (all 50k, before any `--train_subset`) and returns them as a fifth element. `build_data` unpacks it.
   - Gaussian images are drawn at 32x32 from N(mean, std) per channel, with no clipping. They then go through the same center crop and per-image whitening as real images, via a new shared `center_crop_whiten` that `load_base` also uses.
   - Test inputs use the same train statistics. The old whitened-space sampling is gone.
3. **Threshold sweep in `plot`.**
   - `--taus 0.02 0.05 0.1` additionally writes `figure1_b_thresholds.png/.pdf`, one panel (b) per threshold on a shared y-axis, plus `figure1_b_thresholds.csv`.
   - Everything is recomputed from `log.npz`. The panel (b) logic moved into `panel_b_points`, and the main figure uses it too, so `figure1.png` and `figure1_data.csv` should come out the same as before.
   - Without `--taus` the plot step behaves as before.
4. **`--grid panel_a`.** This runs only the five settings above and ignores `--archs`. The header printed by `train` now shows how many runs are exempt from early stopping.

## Before you run it

- **Stale runs are skipped.** Finished runs are skipped by run id only, so old results block the new behavior.
  - If your smoke run wrote to the default `fig1_runs`, delete it or use a separate `--out` for smoke.
  - If any early-stopped Inception p=0, p=1 or pixel runs already exist, delete those folders to get full-length logs.
  - Any earlier gaussian run used the old whitened-space method.
- **Match settings across invocations.** If you schedule `--grid panel_a` separately from `--grid reduced`, pass the same `--tau`, `--batch_size`, `--seed` and `--out` to both. Panel (c) takes its snapshot at the first crossing of the training `--tau`, and mismatched runs would silently mix.
- **The sweep can be censored.**
  - An early-stopped run's log ends `--tail` (1,000) steps after the training-tau crossing.
  - With the default training `--tau 0.05`, the thresholds 0.05 and 0.1 are always resolvable, but 0.02 may not be. Those points show as hollow markers, and `plot` prints a note when this can happen.
  - To resolve 0.02 for the early-stopped runs, train with `--tau 0.02`. The catch is that the (c) snapshot then moves to the 0.02 crossing, so test errors can shift slightly. Runs that never reach 0.02 are drawn hollow in (c), using final-step values.
  - The five panel-(a) runs are unaffected, because their logs always go to 25k steps.
- **Time impact.** On your measured 81.4 ms/step, each of the five Inception runs is about 34 minutes, roughly 2.8 hours for all five. The other four Inception corruption levels (0.2 to 0.8) in the reduced grid are bounded above by 34 minutes each, and early stopping should shorten them.
- **First step.** Re-run `--grid smoke` into a fresh `--out` and check that the gaussian run builds and `plot --taus 0.02 0.05 0.1` writes the extra figure. The smoke numbers are meaningless, since it trains on 2,000 images for 600 steps.

Phase 3 version runs correctly — smoke test passed, and the non-gaussian smoke runs are bit-identical to the previous version. Full Inception runs are training now.

But the full MLP runs (25k steps, batch 128, tau=0.05) show a problem for panel (b):
- p=0.0 crossed tau at step 14200; p=0.1 at 23200; every p >= 0.2 never crossed tau within 25k steps.
- Final train error was small but nonzero: 0.2% (p=0.2), 0.4% (p=0.4), 0.7% (p=0.6), 1.0% (p=0.8), 1.15% (p=1.0).
- By 25k steps the LR has decayed to 0.95^64 ≈ 0.037x, so training longer barely helps.
- Table 1 of the paper reports MLP 1x512 reaching only 99.34% train accuracy on random labels, so the paper's MLP apparently never reached near-zero loss either. A loss threshold can't be what defines "time to overfit" for the MLP.

Please add an accuracy-based convergence criterion: the first step at which the rolling mean of the logged per-step minibatch training accuracy (same 200-step window, checked every 100 steps) reaches a threshold (default 0.99). Requirements:
1. It must be computed at plot time from the existing log.npz `acc` arrays, so I don't have to retrain anything.
2. Add a `--criterion {loss,acc}` option to `plot`, plus `--acc_thresholds` for a sweep (e.g. 0.95 0.99 0.995), mirroring the `--taus` sweep figure.
3. Panel (c) should keep using the existing snapshot/final values; only panel (b) changes.
4. Keep everything else unchanged. Give me the full updated script and a short list of what changed.

I haven't run this version. The accuracy path is new code, so do a quick `plot` on your existing runs and check the printed censoring lines (see "Before you rely on it").

```python
#!/usr/bin/env python3
"""
Figure 1 of Zhang et al. (ICLR 2017), written from the paper text only.

  python zhang_fig1.py bench
  python zhang_fig1.py train --grid smoke      # plumbing check only
  python zhang_fig1.py train --grid panel_a    # only the five Inception panel-(a) runs, full length
  python zhang_fig1.py train --grid reduced    # recommended
  python zhang_fig1.py train --grid full       # paper-like grid
  python zhang_fig1.py plot
  python zhang_fig1.py plot --taus 0.02 0.05 0.1               # also redraw (b) at several loss thresholds
  python zhang_fig1.py plot --criterion acc --acc_threshold 0.99   # panel (b) from training accuracy
  python zhang_fig1.py plot --acc_thresholds 0.95 0.99 0.995       # also redraw (b) at several accuracy thresholds
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


def center_crop_whiten(x):
    """32x32 raw [0,1] images -> center 28x28 crop -> per-image whitening."""
    return per_image_standardization(x[:, :, 2:30, 2:30]).contiguous()


def load_base(args):
    from torchvision.datasets import CIFAR10   # only used to download/decode raw data
    tr = CIFAR10(args.data_root, train=True, download=True)
    te = CIFAR10(args.data_root, train=False, download=True)

    def to_raw(arr):
        return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)

    raw_tr, raw_te = to_raw(tr.data), to_raw(te.data)
    # Per-channel mean/std of the original (raw [0,1], uncropped 32x32) training images.
    # Computed on the full 50k training set, before any --train_subset.
    stats = dict(mean=raw_tr.mean(dim=(0, 2, 3), keepdim=True),
                 std=raw_tr.std(dim=(0, 2, 3), keepdim=True, unbiased=False))
    xtr, ytr = center_crop_whiten(raw_tr), torch.tensor(tr.targets, dtype=torch.long)
    xte, yte = center_crop_whiten(raw_te), torch.tensor(te.targets, dtype=torch.long)
    del raw_tr, raw_te
    if args.train_subset:
        xtr, ytr = xtr[:args.train_subset], ytr[:args.train_subset]
    return xtr, ytr, xte, yte, stats


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
    xtr, ytr, xte, yte, stats = base
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
        # Sample in raw [0,1] pixel space with the original training set's per-channel
        # mean/std (no clipping), then apply the same center crop and per-image
        # whitening as the real images. Test inputs use the same (train) statistics.
        mu, sd = stats["mean"], stats["std"]
        raw_tr = mu + sd * torch.randn((xtr.shape[0], 3, 32, 32), generator=g)
        raw_te = mu + sd * torch.randn((xte.shape[0], 3, 32, 32), generator=g)
        xtr, xte = center_crop_whiten(raw_tr), center_crop_whiten(raw_te)
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


def first_window_crossing(x, thr, below, window=WINDOW, every=CHECK_EVERY):
    """First t (a multiple of `every`, t >= window) at which the mean of x[t-window:t]
    is < thr (below=True) or >= thr (below=False). None if it never happens."""
    csum = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    for t in range(window, len(x) + 1, every):
        m = (csum[t] - csum[t - window]) / window
        if (m < thr) if below else (m >= thr):
            return t
    return None


def first_crossing(loss, tau, window=WINDOW, every=CHECK_EVERY):
    """Loss criterion: rolling mean of minibatch training loss drops below tau."""
    return first_window_crossing(loss, tau, True, window, every)


def first_crossing_acc(acc, thr, window=WINDOW, every=CHECK_EVERY):
    """Accuracy criterion: rolling mean of minibatch training accuracy reaches thr."""
    return first_window_crossing(acc, thr, False, window, every)


def run_one(job, base, args, dev, out_dir):
    arch, setting, p = job["arch"], job["setting"], job["p"]
    # panel-(a) runs always go to max_steps; early stopping only applies to the other runs
    early = args.early and not job.get("panel_a", False)
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
                    if early:
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
                            max_steps=args.max_steps, early_stop=early, tail=args.tail,
                            lr0=lr0, lr_decay=LR_DECAY, momentum=MOMENTUM,
                            perm_over=args.perm_over, train_subset=args.train_subset,
                            gaussian="raw01_train_stats_then_crop_whiten"))
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
    args.early = {"on": True, "off": False,
                  "auto": args.grid not in ("full", "panel_a")}[args.early_stop]


def panel_a_jobs():
    """The five Inception settings drawn in panel (a). Also the p=0 / p=1 points of (b), (c)."""
    return [dict(arch="inception", setting="labels", p=0.0, panel_a=True),
            dict(arch="inception", setting="labels", p=1.0, panel_a=True),
            dict(arch="inception", setting="shuffled_pixels", p=0.0, panel_a=True),
            dict(arch="inception", setting="random_pixels", p=0.0, panel_a=True),
            dict(arch="inception", setting="gaussian", p=0.0, panel_a=True)]


def make_jobs(args):
    if args.grid == "panel_a":
        return panel_a_jobs()                   # ignores --archs
    jobs = []
    for arch in [a for a in ("mlp1x512", "alexnet", "inception") if a in args.archs]:
        if args.grid == "smoke":
            ps = [0.0, 1.0]
        elif args.grid == "reduced":
            ps = P_REDUCED if arch == "inception" else P_FULL
        else:
            ps = P_FULL
        jobs += [dict(arch=arch, setting="labels", p=p,
                      panel_a=(arch == "inception" and p in (0.0, 1.0))) for p in ps]
        if arch == "inception":
            jobs += [dict(arch=arch, setting=s, p=0.0, panel_a=True)
                     for s in ("shuffled_pixels", "random_pixels", "gaussian")]
    return jobs


def cmd_train(args):
    resolve_args(args)
    dev = get_device(args.device)
    out = Path(args.out)
    base = load_base(args)
    jobs = make_jobs(args)
    n_full = sum(1 for j in jobs if j.get("panel_a"))
    print(f"device={dev} jobs={len(jobs)} ({n_full} panel-(a) runs, never early-stopped) "
          f"max_steps={args.max_steps} tau={args.tau} "
          f"early_stop(other runs)={args.early} batch={args.batch_size}")
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


def arch_items(runs, arch):
    """Runs of one architecture with setting 'labels', sorted by p.
    Each item is (p, (meta, loss_array, acc_array))."""
    return sorted(((k[2], v) for k, v in runs.items() if k[0] == arch and k[1] == "labels"),
                  key=lambda t: t[0])


def panel_b_points(items, thr, criterion):
    """Panel (b) for one architecture at one threshold, computed from saved logs only.
    criterion 'loss': first step where the rolling mean minibatch loss is < thr.
    criterion 'acc' : first step where the rolling mean minibatch accuracy is >= thr.
    Returns (t0, p_list, rel_time_list, censored_list, table_rows)."""
    cross = first_crossing if criterion == "loss" else first_crossing_acc
    sig = (lambda l, a: l) if criterion == "loss" else (lambda l, a: a)
    t0 = next((cross(sig(l, a), thr) for p, (m, l, a) in items if p == 0.0), None)
    pb, rb, cb, table = [], [], [], []
    for p, (meta, loss, acc) in items:
        s = sig(loss, acc)
        tc = cross(s, thr)
        rel = None if t0 is None else (tc if tc is not None else len(s)) / t0
        if rel is not None:
            pb.append(p); rb.append(rel); cb.append(tc is None)
        # log ended early because of early stopping (not because the budget ran out)
        cut = bool(meta["config"].get("early_stop")) and meta["first_cross_step"] is not None
        table.append(dict(p=p, steps_run=meta["steps_run"], cross_step=tc,
                          rel_time=rel, censored_b=tc is None, early_stopped_log=cut))
    return t0, pb, rb, cb, table


def report_censoring(arch, tag, table):
    ran_out = [r["p"] for r in table if r["censored_b"] and not r["early_stopped_log"]]
    cut = [r["p"] for r in table if r["censored_b"] and r["early_stopped_log"]]
    if ran_out:
        print(f"{arch} [{tag}]: threshold not reached within the logged steps for p = {ran_out}")
    if cut:
        print(f"{arch} [{tag}]: log was cut by early stopping before the threshold for p = {cut} "
              f"(retrain those runs with --early_stop off to resolve)")


def warn_if_censoring_possible(runs, taus):
    low = min(taus)
    n = sum(1 for (arch, setting, p), (m, _l, _a) in runs.items()
            if setting == "labels" and m["config"]["early_stop"] and m["config"]["tau"] > low)
    if n:
        print(f"note: {n} early-stopped runs were trained with tau > {low}; their logs may end "
              f"before the loss reaches {low}, so those points can appear censored (hollow). "
              f"To resolve them, delete those run folders and re-train with --tau {low} "
              f"(finished runs are skipped).")


def plot_b_sweep(plt, runs, styles, thresholds, out, criterion):
    is_loss = criterion == "loss"
    key = "tau" if is_loss else "acc_threshold"
    stem = "figure1_b_thresholds" if is_loss else "figure1_b_acc_thresholds"
    fig, axs = plt.subplots(1, len(thresholds), figsize=(5.2 * len(thresholds), 4.4),
                            sharey=True, squeeze=False)
    rows = []
    for ax, thr in zip(axs[0], thresholds):
        for arch, (label, color, marker) in styles.items():
            items = arch_items(runs, arch)
            if not items:
                continue
            t0, pb, rb, cb, table = panel_b_points(items, thr, criterion)
            rows += [{key: thr, "arch": arch, **r} for r in table]
            if t0 is None:
                print(f"{key}={thr} {arch}: p=0 run never reached the threshold; series skipped")
            elif pb:
                draw_series(ax, pb, rb, cb, label, color, marker)
            report_censoring(arch, f"{key}={thr}", table)
        ax.set_title(f"(b) convergence slowdown, tau = {thr}" if is_loss
                     else f"(b) convergence slowdown, train acc >= {thr}")
        ax.set_xlabel("label corruption")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    axs[0][0].set_ylabel("time to overfit (relative to p=0)")
    fig.text(0.5, -0.02, "Hollow markers: run did not reach the threshold in its log "
             "(lower bound on the relative time).", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    if rows:
        with open(out / f"{stem}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"wrote {out / (stem + '.png')}")


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
        with np.load(mp.parent / "log.npz") as z:
            loss, acc = z["loss"], z["acc"]
        runs[(meta["arch"], meta["setting"], round(meta["p"], 2))] = (meta, loss, acc)
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
        items = arch_items(runs, arch)
        if not items:
            continue
        if args.criterion == "loss":
            thr = args.tau if args.tau is not None else items[0][1][0]["config"]["tau"]
            desc = f"loss<{thr}"
        else:
            thr = args.acc_threshold
            desc = f"acc>={thr}"
        t0, pb, rb, cb, btable = panel_b_points(items, thr, args.criterion)
        if t0 is None:
            print(f"{arch}: p=0 run never reached {desc}; cannot normalise panel (b)")
        report_censoring(arch, desc, btable)
        pc, ec, cc = [], [], []
        for (p, (meta, loss, acc)), brow in zip(items, btable):
            use = meta["snapshot"] if meta["snapshot"] else meta["final"]   # panel (c): unchanged
            pc.append(p); ec.append(use["test_err"]); cc.append(meta["snapshot"] is None)
            rows.append(dict(arch=arch, p=p, steps_run=brow["steps_run"],
                             cross_step=brow["cross_step"], rel_time=brow["rel_time"],
                             censored_b=brow["censored_b"], test_err=use["test_err"],
                             train_err=use["train_err"],
                             converged_c=meta["snapshot"] is not None,
                             criterion=args.criterion, threshold=thr,
                             early_stopped_log=brow["early_stopped_log"]))
        if pb:
            draw_series(axes[1], pb, rb, cb, label, color, marker)
        draw_series(axes[2], pc, ec, cc, label, color, marker)

    axes[1].set_xlabel("label corruption"); axes[1].set_ylabel("time to overfit (relative to p=0)")
    axes[1].set_title("(b) convergence slowdown" if args.criterion == "loss"
                      else f"(b) convergence slowdown (train acc >= {args.acc_threshold})")
    axes[1].legend()
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

    if args.taus:
        warn_if_censoring_possible(runs, args.taus)
        plot_b_sweep(plt, runs, styles, args.taus, out, "loss")
    if args.acc_thresholds:
        plot_b_sweep(plt, runs, styles, args.acc_thresholds, out, "acc")


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
    pt.add_argument("--grid", choices=["smoke", "reduced", "full", "panel_a"], default="reduced")
    pt.add_argument("--max_steps", type=int, default=None)
    pt.add_argument("--tau", type=float, default=None)
    pt.add_argument("--early_stop", choices=["auto", "on", "off"], default="auto")
    pt.add_argument("--tail", type=int, default=1000)
    pt.add_argument("--eval_every", type=int, default=2500)
    pt.add_argument("--train_subset", type=int, default=0)
    pt.add_argument("--perm_over", choices=["spatial", "all"], default="spatial")

    pp = sub.add_parser("plot"); common(pp)
    pp.add_argument("--tau", type=float, default=None)
    pp.add_argument("--taus", type=float, nargs="+", default=None,
                    help="also redraw panel (b) at these loss thresholds, e.g. --taus 0.02 0.05 0.1")
    pp.add_argument("--criterion", choices=["loss", "acc"], default="loss",
                    help="convergence criterion for panel (b) of the main figure")
    pp.add_argument("--acc_threshold", type=float, default=0.99,
                    help="accuracy threshold used when --criterion acc")
    pp.add_argument("--acc_thresholds", type=float, nargs="+", default=None,
                    help="also redraw panel (b) at these accuracy thresholds, "
                         "e.g. --acc_thresholds 0.95 0.99 0.995")
    pp.add_argument("--plot_window", type=int, default=WINDOW)
    pp.add_argument("--no_ref_line", action="store_true")

    args = ap.parse_args()
    {"bench": cmd_bench, "train": cmd_train, "plot": cmd_plot}[args.cmd](args)


if __name__ == "__main__":
    main()
```

## What changed

1. **Accuracy criterion, computed at plot time.**
   - `first_window_crossing` is a new generic rolling-window detector, and `first_crossing` (loss, unchanged behavior) now calls it.
   - `first_crossing_acc` is the new accuracy version. It uses the same 200-step window, the same 100-step check grid, and `>=` against the threshold.
   - Both read the existing `log.npz` arrays, so nothing needs retraining.
2. **Plot options.**
   - `--criterion {loss,acc}` (default `loss`) selects what panel (b) of `figure1.png` uses.
   - `--acc_threshold` (default 0.99) is the threshold for that main figure.
   - `--acc_thresholds` writes `figure1_b_acc_thresholds.png/.pdf/.csv`, mirroring the `--taus` figure (which is unchanged).
   - `--taus` always means loss thresholds and `--acc_thresholds` always means accuracy thresholds, whatever `--criterion` says, and you can pass both.
3. **Panel (c) is untouched.** It still uses the first-crossing snapshot (from the training-time loss `--tau`) or the final values.
4. **Data plumbing.**
   - `runs[...]` now stores `(meta, loss, acc)`.
   - `arch_items`, `panel_b_points` and `warn_if_censoring_possible` were updated to match.
   - `panel_b_points` takes a `criterion` argument.
5. **Censoring diagnostics.**
   - A new `report_censoring` prints, per architecture and threshold, which p values never reached the threshold within their logged steps.
   - It reports separately which ones were cut off by early stopping.
   - The CSVs gain an `early_stopped_log` column, and `figure1_data.csv` also gains `criterion` and `threshold` columns, appended at the end.
6. **Titles.** With `--criterion acc`, the panel (b) title and the sweep titles say "train acc >= thr". Loss-mode output is otherwise identical to the previous version.

## Before you rely on it

- **Your final errors suggest which thresholds will resolve for the MLP.**
  - These are my expectations, not results. Minibatch accuracy is measured before each update, so it should sit at or slightly below the end-of-training eval accuracy.
  - From your final train errors (about 98.85% at p=1.0, 99.0% at p=0.8, 99.3% at p=0.6), 0.99 will probably leave p=1.0 censored and may leave p=0.8 censored.
  - 0.995 will probably censor p >= 0.6.
  - 0.95 should resolve every p. I'd run `--acc_thresholds 0.95 0.98 0.99` rather than including 0.995.
- **Comparison with the paper.** Your MLP random-label run lands about half a point below the paper's 99.34% in Table 1. That is not unusual given the unspecified batch size and initialization, but it does mean a 0.99 criterion sits right at the edge of what your MLP reaches.
- **The criterion is a proxy.**
  - Training accuracy here is minibatch accuracy in train mode (BatchNorm uses batch statistics for Inception), measured before each update.
  - The 200-step window is about half an epoch (390 steps per epoch at batch 128), so the detected step is biased slightly late.
  - Relative times from the accuracy criterion should not be compared in absolute terms with the loss-criterion ones. Compare their shape across p.
- **Early-stopped Inception runs.** Their logs end 1,000 steps after the loss crossing. An accuracy threshold of 0.99 or lower should normally be reached by then, but 0.995 may not be. The new diagnostics will tell you if so.
- **Panel (c) for the MLP.** For p >= 0.2, where the loss never crossed, (c) uses final-step values and hollow markers, while (b) now uses the accuracy criterion. This is consistent with your requirement 3, but the two panels use different notions of "converged" for the MLP.
- **Suggested check.** `python zhang_fig1.py plot --criterion acc --acc_threshold 0.99 --acc_thresholds 0.95 0.98 0.99`, then read the printed censoring lines before looking at the figure.

web search has been off this entire chat correct?

Yes, as far as my own actions go. I made no tool calls in this chat, so I never ran a web search or fetched a page. Whether the search toggle was on or off in your settings, I can't see, but I didn't use it either way.

Everything came from the attached paper plus my own background knowledge, and a few details came from the latter rather than the paper:

- **Whitening floor.** The `1/sqrt(n)` lower bound on the std is how I remember TensorFlow's per-image standardization working. The paper only names the function and says "adjusted standard deviation."
- **LRN settings.** Size 9, alpha 0.001/9, beta 0.75, k 1 are my choices, as I flagged earlier.
- **Timing estimates.** The step-time estimates were FLOP-based guesses, which your `bench` run replaced with real numbers.
- **Parameter counts.** These I checked against the paper's Table 1 by hand, and your assertions passing confirms them.

can i share this chat even though it is incognito?

I don't know how sharing works for incognito chats, so I'll check Anthropic's support documentation rather than guess.

Yes, I just ran a web search for this, so search was used this turn (not earlier in the chat).

The documentation doesn't say whether incognito chats can be shared. Incognito chats are temporary conversations that aren't saved to your chat history or to Claude's memory, and once closed they cannot be reopened. They also can't be converted to a regular chat or saved to your history. The sharing article describes a Share button in the chat's upper right corner that creates a snapshot of the chat at the moment you share it, but it never mentions incognito chats. I can't confirm whether that button appears or works here.

- **Look for a Share button.** If there is one in the upper right of this chat, you can try it. If there isn't, sharing isn't available for this chat.
- **Save what you need before closing.** Since a closed incognito chat can't be reopened, copy the final script and the change lists into your own files now, whether or not sharing works.
- **Sharing is like publishing.** If you do share, the Share feature creates a public web link to a snapshot of a conversation. Some shared chats were reportedly searchable on Google. The email-invite option described in the support article is the more private way to share. You can turn sharing off later from the share dialog.

Nothing in this chat looks sensitive. The paper is public and the code is yours.

i dont see a share button

Then sharing isn't available for this chat. That fits the fact that incognito chats aren't saved anywhere, so there's nothing for a share link to point to.

Before you close the chat, copy out what you want to keep:

- **The script.** Copy the latest version (the one from the accuracy-criterion turn) into a file. It's the only complete one, so you don't need the earlier two.
- **The change lists and "Before you rely on it" notes.** Paste these into a text file next to the script. They record the deviations from the paper, which you'll want when writing up results.
- **The deviation list from my first reply.** This one is the most useful if you're comparing your figure to the paper's.

To show someone the conversation, you could paste the text into a document or message yourself. Another option is to start a regular (non-incognito) chat, paste the script and notes into it, and share that one. Regular chats do have the Share button. Anything you paste there would no longer be incognito, which is fine here since nothing in this chat is sensitive.

is this ok to say for the assignment 2 can we use markdown documents of the conversation if we do not want to share the LLM conversation?

I can't tell, because it depends on your course's policy, and I haven't seen the assignment text. The safest move is to ask your instructor or TA before you submit, rather than assume.

- **Verbatim transcript vs. summary.** If the assignment asks you to disclose your LLM use, a full, unedited transcript saved as markdown is usually a reasonable substitute for a share link. A summary or a trimmed version might not satisfy it, since the point of disclosure is usually that graders can see what you asked and what the model produced.
- **Don't edit it.** Copy the whole conversation as it appeared, including the parts where you pushed back or the code needed fixing. Removing parts or tidying wording could look like misrepresentation even if that wasn't your intent.
- **Explain the format.** This chat is incognito, so it has no share link. Say so in your submission, so it doesn't look like you're avoiding disclosure.
- **Check the other requirements.** Some courses also want a statement of what you used the model for and what you verified yourself. If so, note that you ran the code, checked the parameter counts against Table 1, and made the design decisions listed in the change lists.

Here is a short message you could send:

> Hi, for Assignment 2 I used an LLM in an incognito session, so there's no shareable link. Would a full, unedited transcript saved as markdown files be acceptable as the disclosure? Happy to provide it in whatever format you prefer.

If they say no, you can still paste the conversation into a regular chat and share that, as I mentioned before.
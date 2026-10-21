I've attached the paper "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (Keskar et al., ICLR 2017). I want to reproduce Figure 3, panels F1, C1, and C2, from scratch in PyTorch. Please do not search the web or use any existing implementation — work only from the paper.

1. First, summarize the exact setup the paper specifies for these panels: which network and dataset each panel uses (check the network table), the layer-by-layer architecture, the small-batch and large-batch sizes, the optimizer and its hyperparameters, the number of epochs, preprocessing, the range of α, and what is plotted on each axis. Point out anything the paper leaves unspecified.
2. Then write a complete, runnable PyTorch implementation (a single Python script or notebook cells) that trains the small-batch and large-batch solutions for each panel, evaluates train and test cross-entropy loss (and accuracy, if the figure shows it) along the 1-D linear interpolation between the two solutions, and plots the three panels in the style of the paper's figure.
3. Be explicit about two choices and justify them: how batch-norm statistics are computed at the interpolated weights (if the networks use batch norm), and whether the two runs share an initialization.
4. I'm running on an Apple M4 Pro laptop (PyTorch MPS backend, 24 GB RAM), so tell me the expected runtime and clearly list every deviation from the paper.

Do not hardcode or fabricate any values in the plots; every number must come from actual training runs logged to disk.

# 1. What the paper specifies for panels F1, C1, C2

**Network and dataset (Table 1 and Appendices A, B)**

| Panel | Network | Dataset | Train / test size |
|---|---|---|---|
| F1 | Fully connected (B.1) | MNIST | 60000 / 10000 |
| C1 | Shallow conv, AlexNet-like (B.3) | CIFAR-10 | 50000 / 10000 |
| C2 | Deep conv, VGG-like (B.4) | CIFAR-10 | 50000 / 10000 |

**Layer by layer**
- **F1:** 784 inputs, then 5 batch-normalized layers of 512 neurons with ReLU, then a 10-way softmax output.
- **C1:** Two repeats of [64 filters, 5x5, stride 2] followed by MaxPool(3), then dense layers of 384 and 192, then a 10-way output. Batch norm on all layers, ReLU, and dropout "of 0.5 retention probability" on the two dense layers (so the drop rate is 0.5 either way).
- **C2:** Conv stacks of 2x64, 2x128, 3x256, 3x512, 3x512 (3x3, stride 1), with MaxPool(2) after each stack, then a 512-unit dense layer and a 10-way output. Activation, batch norm and dropout are "as in B.3". The text says "C3 network" in B.4, which is a typo for C2.

**Training**
- **Batch sizes:** Large batch (LB) is 10% of the training set (6000 for MNIST, 5000 for CIFAR-10). Small batch (SB) is 256.
- **Optimizer:** ADAM for both regimes. The paper gives no hyperparameters.
- **Loss:** Mean cross-entropy.
- **Epochs:** There is no number for Figure 3. Section 2.2 says training ran "without any budget or limits, until the loss function ceased to improve". Figure 2 and Section 3 use 100 epochs.
- **Preprocessing:** Appendix A says MNIST and CIFAR-10 are "used without any pre-processing". No augmentation is used in the baseline.
- **Trials:** Table 2 uses 5 runs from different uniformly distributed random starting points. Figure 3 uses one randomly chosen SB/LB pair from those 5 trials.

**The plot**
- The curve is f(αx_l + (1-α)x_s) for α in [-1, 2]. α=0 is the SB solution and α=1 is the LB solution.
- The x-axis is α.
- The left y-axis (blue) is cross-entropy loss. Solid is train and dashed is test.
- The right y-axis (red) is classification accuracy in percent, with the same solid/dashed convention.

**Left unspecified**
- Learning rate, betas, epsilon and weight decay.
- The exact initialization distribution (only "uniformly distributed random").
- Whether the SB and LB runs of a trial share an initialization.
- How batch-norm statistics are handled at interpolated weights, and whether dropout is active when evaluating.
- Conv padding and pool stride in C1, and the conv/BN/ReLU order.
- Whether dropout is applied to the single dense layer of C2.
- Whether pixels are scaled to [0,1].
- The number of α points.
- Whether BN scale and shift are part of the interpolated "weights".
- The epoch count for the figure.

# 2. Implementation

Save this as `repro_fig3.py`. I have not run it on MPS, so see section 4 for the risks. Datasets are fetched through `torchvision.datasets`, and all model, training and interpolation code is written from the paper.

```python
#!/usr/bin/env python3
"""Reproduce Figure 3 (panels F1, C1, C2) of Keskar et al., ICLR 2017, from the paper text.

Stages (all results are written to disk, the plot reads only from disk):
  train  : trains SB and LB solutions, logs per-epoch metrics (jsonl), saves weights
  interp : evaluates the 1-D linear interpolation, writes curve.json
  plot   : reads curve.json for each panel and draws the figure
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DATASET = {"F1": "mnist", "C1": "cifar10", "C2": "cifar10"}


# ----------------------------------------------------------------- utilities
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def free_mem(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_data(name, root, device):
    from torchvision import datasets
    if name == "mnist":
        tr = datasets.MNIST(root, train=True, download=True)
        te = datasets.MNIST(root, train=False, download=True)
        prep = lambda d: (d.data.float().div(255).reshape(len(d.data), -1),
                          d.targets.clone().long())
    else:
        tr = datasets.CIFAR10(root, train=True, download=True)
        te = datasets.CIFAR10(root, train=False, download=True)
        prep = lambda d: (torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous()
                          .float().div(255), torch.tensor(d.targets).long())
    Xtr, ytr = prep(tr)
    Xte, yte = prep(te)
    return [t.to(device) for t in (Xtr, ytr, Xte, yte)]


# -------------------------------------------------------------------- models
def conv(i, o, k, s, p):
    return [nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.ReLU()]


def dense(i, o, dropout):
    layers = [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return layers


class Net(nn.Module):
    """segments: list of Sequentials (optionally gradient-checkpointed), then a head."""
    def __init__(self, segments, head):
        super().__init__()
        self.segments = nn.ModuleList(segments)
        self.head = head
        self.ckpt = False

    def forward(self, x):
        for seg in self.segments:
            if self.ckpt and self.training and torch.is_grad_enabled():
                # segments contain no dropout, so RNG state need not be preserved
                x = checkpoint(seg, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = seg(x)
        return self.head(x)


def build_net(name):
    if name == "F1":                                   # Appendix B.1
        layers, d = [], 784
        for _ in range(5):
            layers += dense(d, 512, dropout=False)
            d = 512
        return Net([nn.Sequential(*layers)], nn.Linear(512, 10))
    if name == "C1":                                   # Appendix B.3 (assumed padding/pool stride)
        s1 = nn.Sequential(*conv(3, 64, 5, 2, 2), nn.MaxPool2d(3, 2, 1))    # 32 -> 16 -> 8
        s2 = nn.Sequential(*conv(64, 64, 5, 2, 2), nn.MaxPool2d(3, 2, 1))   # 8 -> 4 -> 2
        head = nn.Sequential(nn.Flatten(), *dense(256, 384, True),
                             *dense(384, 192, True), nn.Linear(192, 10))
        return Net([s1, s2], head)
    if name == "C2":                                   # Appendix B.4
        segs, cin = [], 3
        for cout, n in [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]:
            for j in range(n):
                layers = conv(cin, cout, 3, 1, 1)
                cin = cout
                if j == n - 1:
                    layers.append(nn.MaxPool2d(2))     # 32 -> 16 -> 8 -> 4 -> 2 -> 1
                segs.append(nn.Sequential(*layers))
        head = nn.Sequential(nn.Flatten(), *dense(512, 512, True), nn.Linear(512, 10))
        return Net(segs, head)
    raise ValueError(name)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, X, y, bs):
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs])
        loss_sum += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return loss_sum / len(X), 100.0 * correct / len(X)


def evaluate_both(model, data, bs):
    Xtr, ytr, Xte, yte = data
    trl, tra = evaluate(model, Xtr, ytr, bs)
    tel, tea = evaluate(model, Xte, yte, bs)
    return dict(train_loss=trl, train_acc=tra, test_loss=tel, test_acc=tea)


@torch.no_grad()
def recompute_bn(model, X, bs):
    """Re-estimate BN running stats for the CURRENT weights from the full training set.
    Dropout stays off (model.eval()); only BN layers are put in train mode, with a
    cumulative (equal-weight) average over chunks."""
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for i in range(0, len(X), bs):
        model(X[i:i + bs])
    for m in bns:
        m.eval()
        m.momentum = 0.1


# ------------------------------------------------------------------ training
def train_run(model, data, bs, epochs, lr, run_seed, log_path, eval_every, eval_bs, device):
    Xtr, ytr, Xte, yte = data
    N = len(Xtr)
    torch.manual_seed(run_seed)
    g = torch.Generator().manual_seed(run_seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    with open(log_path, "w") as f:
        for ep in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            perm = torch.randperm(N, generator=g).to(device)
            run_loss = torch.zeros((), device=device)
            run_corr = torch.zeros((), device=device)
            for i in range(0, N, bs):
                idx = perm[i:i + bs]
                x, y = Xtr[idx], ytr[idx]
                out = model(x)
                loss = F.cross_entropy(out, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run_loss += loss.detach() * len(idx)
                run_corr += (out.argmax(1) == y).sum()
            rec = dict(epoch=ep, batch_size=bs,
                       train_loss_running=run_loss.item() / N,
                       train_acc_running=100.0 * run_corr.item() / N,
                       epoch_seconds=time.time() - t0)
            if ep % eval_every == 0 or ep == epochs:
                rec.update(evaluate_both(model, data, eval_bs))   # eval mode, running BN stats
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  ep {ep:4d} bs {bs:5d} loss {rec['train_loss_running']:.4f} "
                  f"acc {rec['train_acc_running']:.2f} "
                  f"{'test_acc %.2f' % rec['test_acc'] if 'test_acc' in rec else ''} "
                  f"({rec['epoch_seconds']:.1f}s)", flush=True)


def stage_train(args, panel, data, device):
    d = os.path.join(args.out, panel)
    os.makedirs(d, exist_ok=True)
    N = len(data[0])
    for tag in ("sb", "lb"):
        if os.path.exists(f"{d}/{tag}.pt"):
            print(f"[{panel}] {tag}.pt exists, skipping")
            continue
        init_seed = args.seed if (args.init == "shared" or tag == "sb") else args.seed + 1000
        torch.manual_seed(init_seed)
        model = build_net(panel)                     # built on CPU: init is device independent
        torch.save(model.state_dict(), f"{d}/init_{tag}.pt")
        model.to(device)
        bs = args.sb_bs if tag == "sb" else int(round(args.lb_frac * N))
        epochs = (args.sb_epochs if tag == "sb" else args.lb_epochs) or args.epochs
        model.ckpt = (tag == "lb") and (args.ckpt == "on" or (args.ckpt == "auto" and panel == "C2"))
        print(f"[{panel}] training {tag}: bs={bs}, epochs={epochs}, ckpt={model.ckpt}", flush=True)
        t0 = time.time()
        train_run(model, data, bs, epochs, args.lr, args.seed * 10 + (1 if tag == "sb" else 2),
                  f"{d}/log_{tag}.jsonl", args.eval_every, args.eval_bs, device)
        model.ckpt = False
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f"{d}/{tag}.pt")
        with open(f"{d}/run_{tag}.json", "w") as f:
            json.dump(dict(args=vars(args), batch_size=bs, epochs=epochs, device=str(device),
                           torch=torch.__version__, wall_seconds=time.time() - t0), f, indent=1)
        del model
        free_mem(device)


# ------------------------------------------------------------- interpolation
def param_vector(sd, pnames):
    return torch.cat([sd[n].flatten().double() for n in pnames])


def stage_interp(args, panel, data, device):
    d = os.path.join(args.out, panel)
    sd_s = torch.load(f"{d}/sb.pt", map_location="cpu")
    sd_l = torch.load(f"{d}/lb.pt", map_location="cpu")
    model = build_net(panel).to(device)
    model.ckpt = False
    pnames = [n for n, _ in model.named_parameters()]

    # distance-from-init ratio (the paper reports 3 to 10 for ||xs-x0|| / ||xl-x0||)
    x0s = param_vector(torch.load(f"{d}/init_sb.pt", map_location="cpu"), pnames)
    x0l = param_vector(torch.load(f"{d}/init_lb.pt", map_location="cpu"), pnames)
    dist_ratio = (param_vector(sd_s, pnames) - x0s).norm().item() / \
                 (param_vector(sd_l, pnames) - x0l).norm().item()

    # endpoints evaluated with their own trained running BN stats (consistency check)
    native = {}
    for tag, sd in (("sb", sd_s), ("lb", sd_l)):
        model.load_state_dict(sd)
        native[tag] = evaluate_both(model, data, args.eval_bs)

    alphas = np.linspace(-1.0, 2.0, args.n_alpha)
    keys = ("train_loss", "train_acc", "test_loss", "test_acc")
    res = {k: [] for k in keys}
    t0 = time.time()
    for n, a in enumerate(alphas):
        a = float(a)
        sd = {}
        for k in sd_s:
            if sd_s[k].is_floating_point():
                sd[k] = a * sd_l[k] + (1.0 - a) * sd_s[k]   # params (incl. BN gamma, beta) and buffers
            else:
                sd[k] = sd_s[k]                              # num_batches_tracked
        model.load_state_dict(sd)
        if args.bn_mode == "recompute":
            recompute_bn(model, data[0], args.eval_bs)
        r = evaluate_both(model, data, args.eval_bs)
        for k in keys:
            res[k].append(r[k])
        print(f"[{panel}] alpha {a:+.3f} train_loss {r['train_loss']:.4f} "
              f"test_loss {r['test_loss']:.4f} ({n + 1}/{len(alphas)}, {time.time() - t0:.0f}s)",
              flush=True)
    with open(f"{d}/curve.json", "w") as f:
        json.dump(dict(panel=panel, alpha=[float(a) for a in alphas], bn_mode=args.bn_mode,
                       init=args.init, native_endpoints=native, dist_ratio_sb_over_lb=dist_ratio,
                       **res), f, indent=1)


# ---------------------------------------------------------------------- plot
def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [p for p in args.panels if os.path.exists(f"{args.out}/{p}/curve.json")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.4), squeeze=False)
    for i, (ax, p) in enumerate(zip(axes[0], panels)):
        c = json.load(open(f"{args.out}/{p}/curve.json"))
        a = np.array(c["alpha"])
        ax.plot(a, c["train_loss"], "b-", label="Train")
        ax.plot(a, c["test_loss"], "b--", label="Test")
        ax.set_ylim(bottom=0)
        ax.set_xlim(-1, 2)
        ax.set_xlabel("alpha")
        ax.set_ylabel("Cross Entropy", color="b")
        ax.tick_params(axis="y", colors="b")
        ax.grid(True, linestyle=":")
        ax.legend(loc="upper left")
        ax2 = ax.twinx()
        ax2.plot(a, c["train_acc"], "r-")
        ax2.plot(a, c["test_acc"], "r--")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Accuracy", color="r")
        ax2.tick_params(axis="y", colors="r")
        ax.set_title(f"({chr(97 + i)}) {p}")
    fig.tight_layout()
    fig.savefig(f"{args.out}/figure3_reproduction.png", dpi=200)
    print("saved", f"{args.out}/figure3_reproduction.png")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "interp", "plot", "all"])
    ap.add_argument("--panels", nargs="+", default=["F1", "C1", "C2"])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)            # unspecified in paper
    ap.add_argument("--sb_bs", type=int, default=256)
    ap.add_argument("--lb_frac", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--sb_epochs", type=int, default=None)
    ap.add_argument("--lb_epochs", type=int, default=None)
    ap.add_argument("--init", choices=["shared", "independent"], default="shared")
    ap.add_argument("--bn_mode", choices=["recompute", "interp"], default="recompute")
    ap.add_argument("--n_alpha", type=int, default=61)
    ap.add_argument("--eval_bs", type=int, default=2500)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--ckpt", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    print("device:", device, "torch:", torch.__version__)

    if args.stage in ("train", "interp", "all"):
        for panel in args.panels:
            data = load_data(DATASET[panel], args.data_root, device)
            if args.stage in ("train", "all"):
                stage_train(args, panel, data, device)
            if args.stage in ("interp", "all"):
                stage_interp(args, panel, data, device)
            del data
            free_mem(device)
    if args.stage in ("plot", "all"):
        stage_plot(args)


if __name__ == "__main__":
    main()
```

Usage:
```
# smoke test first, to measure per-epoch time on your machine
python repro_fig3.py all --epochs 2 --n_alpha 5 --out runs_smoke
# full run
export PYTORCH_ENABLE_MPS_FALLBACK=1
python repro_fig3.py all
```
Training is resumable per run (`sb.pt` and `lb.pt` are skipped if present). Everything in the plot comes from `runs/<panel>/curve.json`, which `interp` writes from actual evaluations. Per-epoch training logs are in `log_sb.jsonl` and `log_lb.jsonl`.

# 3. The two explicit choices

**Batch-norm statistics at interpolated weights (default `--bn_mode recompute`).** All three networks use batch norm. A point on the interpolation line is a new network whose BN statistics have never been estimated. Each endpoint's running statistics belong to a different network, and linearly mixing running means and variances does not give the statistics of the mixed network, because variance is nonlinear in the weights. Evaluating with batch statistics would make the loss depend on batch size and composition, and for test data it would use test-set information.

So at every α, including the endpoints, I reset the BN buffers and re-estimate them with one forward pass over the full training set (dropout off, equal-weight average over chunks of 2500). I then evaluate both train and test in eval mode with those fixed statistics. This makes the loss a deterministic function of the weights alone, and the same procedure applies everywhere on the line. BN scale and shift are interpolated like any other weight. I don't know what the authors did, so two safeguards are built in. The curve file stores each endpoint evaluated with its own trained running statistics (`native_endpoints`), so you can see how much recomputation changes the endpoints. `--bn_mode interp` gives the alternative, linear interpolation of the buffers.

**Shared initialization (default `--init shared`).** The paper does not say. I chose shared for two reasons:
- Section 3 reports the ratio of ||x_s - x0|| to ||x_l - x0|| for the trials. A single x0 per trial is the most natural reading of that, though this is an inference rather than an explicit statement.
- Sharing isolates the effect of batch size. With independent inits, even two runs at the same batch size would typically sit in unrelated basins and show a barrier between them, which would confound the sharp-minima claim.

The caveat is that a shared start may make the two solutions closer than in the paper, which could change the barrier height. `--init independent` runs the alternative. The script logs the distance ratio to `curve.json` as a check against the paper's 3 to 10.

# 4. Runtime and deviations

**Expected runtime on an M4 Pro.** These are estimates from FLOP counts and typical MPS efficiency. I have not measured them, and the uncertainty is wide, so run the smoke test and extrapolate.
- **F1:** About 23k SB steps and 1k LB steps. Roughly 3 to 8 minutes, dominated by per-step overhead.
- **C1:** Roughly 10 to 25 minutes in total.
- **C2:** Per epoch, a VGG16-style net on 50k CIFAR images is about 47 TFLOP of compute. At a realistic 2 to 4 TFLOPS, that is about 12 to 25 s per epoch, so 40 to 80 minutes for both runs. The LB run pays roughly 30% extra for gradient checkpointing, and the 61-point interpolation adds about 10 to 15 minutes. Total about 1.2 to 2.5 hours.
- **Overall:** Roughly 1.5 to 3 hours for all three panels, for one seed.

**Memory risk for C2 LB.** Batch 5000 through this VGG without tricks needs on the order of 16 GB of activations, which is too close to the usable MPS limit on 24 GB. The script therefore applies gradient checkpointing per conv layer for the C2 LB run. This is mathematically identical to normal training, including BN statistics over the full 5000-sample batch. It trades about 30% extra compute for a peak that I estimate at roughly 6 to 10 GB. If it still runs out of memory, the only fix I'd consider changes the method (smaller virtual batches change BN behavior), so I'd rather you tell me than have it done silently.

**Other risks.** I haven't run this on MPS. Op-support or numerics problems are possible, which is why I suggest setting `PYTORCH_ENABLE_MPS_FALLBACK=1`.

**Choices where the paper is silent**
- ADAM with lr 1e-3 and PyTorch defaults for the other settings, with the same lr for both batch sizes.
- PyTorch default initialization, which is uniform-based.
- Pixels scaled to [0,1] with no mean subtraction or augmentation.
- Linear, then BN, then ReLU, then dropout ordering.
- Dropout 0.5 on all dense hidden layers of C1 and C2, none in F1.
- C1 uses conv padding 2 and an overlapping MaxPool (kernel 3, stride 2, padding 1), which gives a 256-dimensional flattened feature.
- C2 uses padding 1 on its 3x3 convs, which is implied because otherwise the spatial size would collapse before the last pool.
- Evaluation is deterministic, with dropout off.

**Deviations from the paper**
- **Epochs:** The paper trains "until the loss ceased to improve". I use a fixed 100 epochs for both batch sizes, which is the budget used in Figures 2 and 5. LB runs get only 1000 updates at that budget, so they may be under-trained. Compare the `train_acc` in the logs against the paper's Table 2 training accuracies (F1 99.92%, C1 99.66%, C2 99.99% for LB). If LB is short, raise `--lb_epochs`.
- **Trials:** The paper picks one pair at random out of 5 trials. The default here is a single seed. Running `--seed 1` through `--seed 4` with separate `--out` directories gets you five pairs, at five times the cost.
- **BN handling and initialization:** As in section 3, a necessary choice rather than a verified match.
- **α grid:** 61 points (step 0.05). Narrow features near α=1 could be undersampled. Use `--n_alpha 121` for a finer grid, which doubles the interpolation cost.
- **Intermediate evaluation:** Train and test metrics are logged every 5 epochs, not every epoch, to save time. Running train-mode loss is logged every epoch.
- **Hardware:** MPS float32, non-deterministic, and not the authors' hardware.
- **Data:** MNIST and CIFAR-10 are downloaded through `torchvision.datasets`. Point `--data_root` at existing copies if you are offline.

Don't expect a pixel-for-pixel match with the paper's figure. It shows one random pair, so the qualitative claim to check is that the LB side of the line shows a sharper loss rise than the SB side, not the exact curve shapes.

I ran your script. Smoke test passed for all panels. Full 100-epoch results for F1 and C1 (shared init, BN recompute):

F1: train acc SB 99.99% / LB 100.00%, test acc SB 98.69% / LB 98.61%. Distance ratio 14.2. The interpolation curve is flat from alpha=0 to 1 and rises sharply just past alpha=1 — qualitatively matches the paper.

C1 has two problems:
- Test acc SB 69.48% / LB 65.86% vs Table 2's 80.04% / 77.26%; train acc SB 97.81% / LB 96.80% vs 99.89% / 99.66%.
- Training loss was still decreasing at epoch 100 (running train loss SB 0.099 at epoch 75 -> 0.076 at 100; LB 0.129 -> 0.058). The paper says the networks were trained "until the loss function ceased to improve", so a fixed 100-epoch cap undertrains.
- The interpolation curve also shows a train-loss bump near alpha=0.85, between the two solutions.

Please make these changes:
1. Replace the fixed epoch count with a stopping rule matching the paper: stop when the full-training-set loss (eval mode) hasn't improved by a relative amount for a patience window, with a configurable per-run maximum epoch cap so C2 stays affordable on my laptop (C2 SB measured ~86 s/epoch, LB ~116 s/epoch, though the GPU was shared at the time). Log the stopping epoch and the reason.
2. For C1, your downsampling (stride-2 conv + stride-2 pool, twice) collapses 32x32 to 2x2, a 256-d flatten. Reconsider what "2 sets of [64, 5, 5, 2]–MaxPool(3)" most plausibly means, explain your reasoning, and add a flag so I can run both your original reading and the alternative as an ablation.
3. Keep everything else unchanged. Give me the full updated script and a short list of what changed.

# C1 architecture reasoning

The paper defines n×[a,b,c,d] as a stack of n convs with a filters, a b×c kernel and stride d. Read literally, "2 sets of [64,5,5,2]-MaxPool(3)" gives stride-2 convs, which is what I implemented. That reading has three problems:

- **It over-downsamples.** Two stride-2 convs plus two stride-2 pools take 32x32 to 2x2. Your run at 69.5% test accuracy against the paper's 80.0% is consistent with that bottleneck, though it does not prove it.
- **The pool convention gives no help.** In B.4, five MaxPool(2) layers reduce 32 to 1 and feed a 512-d dense layer, so there MaxPool(k) has stride k. Applying that to MaxPool(3) with stride-2 convs gives a 64-d flatten, which is worse.
- **Another notation fits better.** The paper calls C1 a "modified AlexNet", and AlexNet uses overlapping 3x3 stride-2 pooling. A plausible reading is that the trailing "2" is that pool stride, attached to the conv tuple by mistake. That gives stride-1 5x5 convs and 3x3 stride-2 pools, so 32 becomes 8x8 and the flatten is 4096-d.

Outside the paper, the 384/192 dense sizes match a small CIFAR-10 reference network I remember, which had stride-1 convs and 3x3 stride-2 pools. That is memory, not something I looked up, so treat it as a hint, not evidence.

**Flag:** `--c1_variant alexnet` is my best guess and is now the default. `collapse` is the original reading, and `both` runs both and plots them as separate panels. If `alexnet` reaches about 80% test and about 99.9% train under the new stopping rule, that supports it. If it doesn't, the gap could still come from lr, preprocessing or dropout placement, so the ablation won't fully settle it. To try another reading, add a line to `C1_VARIANTS`.

# Updated script

```python
#!/usr/bin/env python3
"""Reproduce Figure 3 (panels F1, C1, C2) of Keskar et al., ICLR 2017, from the paper text.

Stages (all results are written to disk, the plot reads only from disk):
  train  : trains SB and LB solutions until the full-train-set loss (eval mode) stops
           improving, or a per-run epoch cap is hit; logs per-epoch metrics (jsonl)
  interp : evaluates the 1-D linear interpolation, writes curve.json
  plot   : reads curve.json for each run and draws the figure
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DATASET = {"F1": "mnist", "C1": "cifar10", "C2": "cifar10"}

# Two readings of "2 sets of [64,5,5,2]-MaxPool(3)" (Appendix B.3).
C1_VARIANTS = {
    "collapse": dict(conv_stride=2, pool_stride=2),   # original reading: 32 -> 2x2, 256-d flatten
    "alexnet":  dict(conv_stride=1, pool_stride=2),   # overlapping 3x3 stride-2 pool: 32 -> 8x8, 4096-d flatten
}


# ----------------------------------------------------------------- utilities
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def free_mem(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def expand_runs(args):
    """Panels -> list of (panel, variant). Only C1 has variants."""
    runs = []
    for p in args.panels:
        if p == "C1":
            vs = ["alexnet", "collapse"] if args.c1_variant == "both" else [args.c1_variant]
            runs += [(p, v) for v in vs]
        else:
            runs.append((p, None))
    return runs


def run_key(panel, variant):
    return panel if variant is None else f"{panel}_{variant}"


def run_label(panel, variant):
    return panel if variant is None else f"{panel} ({variant})"


def parse_caps(items):
    caps = {}
    for s in items:
        try:
            key, val = s.split("=")
            panel, tag = key.split(":")
            assert panel in DATASET and tag in ("sb", "lb")
            caps[(panel, tag)] = int(val)
        except Exception:
            raise ValueError(f"bad --cap '{s}', expected PANEL:sb=N or PANEL:lb=N, e.g. C2:lb=40")
    return caps


def load_data(name, root, device):
    from torchvision import datasets
    if name == "mnist":
        tr = datasets.MNIST(root, train=True, download=True)
        te = datasets.MNIST(root, train=False, download=True)
        prep = lambda d: (d.data.float().div(255).reshape(len(d.data), -1),
                          d.targets.clone().long())
    else:
        tr = datasets.CIFAR10(root, train=True, download=True)
        te = datasets.CIFAR10(root, train=False, download=True)
        prep = lambda d: (torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous()
                          .float().div(255), torch.tensor(d.targets).long())
    Xtr, ytr = prep(tr)
    Xte, yte = prep(te)
    return [t.to(device) for t in (Xtr, ytr, Xte, yte)]


# -------------------------------------------------------------------- models
def conv(i, o, k, s, p):
    return [nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.ReLU()]


def dense(i, o, dropout):
    layers = [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return layers


def feat_dim(segments, in_shape):
    """Flattened feature size after the conv segments, found by a throwaway forward pass."""
    with torch.no_grad():
        x = torch.zeros(2, *in_shape)
        for s in segments:
            x = s.eval()(x)
    return x[0].numel()


class Net(nn.Module):
    """segments: list of Sequentials (optionally gradient-checkpointed), then a head."""
    def __init__(self, segments, head):
        super().__init__()
        self.segments = nn.ModuleList(segments)
        self.head = head
        self.ckpt = False

    def forward(self, x):
        for seg in self.segments:
            if self.ckpt and self.training and torch.is_grad_enabled():
                # segments contain no dropout, so RNG state need not be preserved
                x = checkpoint(seg, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = seg(x)
        return self.head(x)


def build_net(name, variant=None):
    if name == "F1":                                   # Appendix B.1
        layers, d = [], 784
        for _ in range(5):
            layers += dense(d, 512, dropout=False)
            d = 512
        return Net([nn.Sequential(*layers)], nn.Linear(512, 10))
    if name == "C1":                                   # Appendix B.3, see C1_VARIANTS
        v = C1_VARIANTS[variant or "alexnet"]
        cs, ps = v["conv_stride"], v["pool_stride"]
        s1 = nn.Sequential(*conv(3, 64, 5, cs, 2), nn.MaxPool2d(3, ps, 1))
        s2 = nn.Sequential(*conv(64, 64, 5, cs, 2), nn.MaxPool2d(3, ps, 1))
        d = feat_dim([s1, s2], (3, 32, 32))
        head = nn.Sequential(nn.Flatten(), *dense(d, 384, True),
                             *dense(384, 192, True), nn.Linear(192, 10))
        return Net([s1, s2], head)
    if name == "C2":                                   # Appendix B.4
        segs, cin = [], 3
        for cout, n in [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]:
            for j in range(n):
                layers = conv(cin, cout, 3, 1, 1)
                cin = cout
                if j == n - 1:
                    layers.append(nn.MaxPool2d(2))     # 32 -> 16 -> 8 -> 4 -> 2 -> 1
                segs.append(nn.Sequential(*layers))
        head = nn.Sequential(nn.Flatten(), *dense(512, 512, True), nn.Linear(512, 10))
        return Net(segs, head)
    raise ValueError(name)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, X, y, bs):
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs])
        loss_sum += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return loss_sum / len(X), 100.0 * correct / len(X)


def evaluate_both(model, data, bs):
    Xtr, ytr, Xte, yte = data
    trl, tra = evaluate(model, Xtr, ytr, bs)
    tel, tea = evaluate(model, Xte, yte, bs)
    return dict(train_loss=trl, train_acc=tra, test_loss=tel, test_acc=tea)


@torch.no_grad()
def recompute_bn(model, X, bs):
    """Re-estimate BN running stats for the CURRENT weights from the full training set.
    Dropout stays off (model.eval()); only BN layers are put in train mode, with a
    cumulative (equal-weight) average over chunks."""
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for i in range(0, len(X), bs):
        model(X[i:i + bs])
    for m in bns:
        m.eval()
        m.momentum = 0.1


# ------------------------------------------------------------------ training
def train_run(model, data, bs, max_epochs, run_seed, log_path, args, device):
    """Train until the full-train-set loss (eval mode) stops improving, or max_epochs.

    Stopping rule: every `stop_eval_every` epochs, compute the full-training-set loss in eval
    mode. It counts as an improvement only if it is below (1 - rel_tol) * (reference loss),
    where the reference is the loss at the last counted improvement. Stop when `patience`
    epochs have passed since the last counted improvement. The final iterate is kept.
    """
    Xtr, ytr, Xte, yte = data
    N = len(Xtr)
    torch.manual_seed(run_seed)
    g = torch.Generator().manual_seed(run_seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_epoch, stop_reason, ep, last = float("inf"), 0, "max_epochs", 0, {}
    with open(log_path, "w") as f:
        for ep in range(1, max_epochs + 1):
            model.train()
            t0 = time.time()
            perm = torch.randperm(N, generator=g).to(device)
            run_loss = torch.zeros((), device=device)
            run_corr = torch.zeros((), device=device)
            for i in range(0, N, bs):
                idx = perm[i:i + bs]
                x, y = Xtr[idx], ytr[idx]
                out = model(x)
                loss = F.cross_entropy(out, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run_loss += loss.detach() * len(idx)
                run_corr += (out.argmax(1) == y).sum()
            rec = dict(epoch=ep, batch_size=bs,
                       train_loss_running=run_loss.item() / N,
                       train_acc_running=100.0 * run_corr.item() / N,
                       epoch_seconds=time.time() - t0)

            check = (ep % args.stop_eval_every == 0)
            log_now = (ep % args.eval_every == 0)
            is_last = (ep == max_epochs)
            if check or log_now or is_last:
                rec["train_loss"], rec["train_acc"] = evaluate(model, Xtr, ytr, args.eval_bs)
            stop = False
            if check:
                L = rec["train_loss"]
                if not np.isfinite(L):
                    stop_reason, stop = "nonfinite_loss", True
                else:
                    if L < best * (1.0 - args.rel_tol):
                        best, best_epoch = L, ep
                    if ep - best_epoch >= args.patience:
                        stop_reason, stop = "patience", True
                rec["best_ref_loss"] = best if np.isfinite(best) else None
                rec["best_ref_epoch"] = best_epoch
            if log_now or is_last or stop:
                rec["test_loss"], rec["test_acc"] = evaluate(model, Xte, yte, args.eval_bs)

            f.write(json.dumps(rec) + "\n")
            f.flush()
            last = rec
            print(f"  ep {ep:4d} bs {bs:5d} run_loss {rec['train_loss_running']:.4f} "
                  f"run_acc {rec['train_acc_running']:.2f} "
                  + (f"| eval_train_loss {rec['train_loss']:.5f} " if "train_loss" in rec else "")
                  + (f"test_acc {rec['test_acc']:.2f} " if "test_acc" in rec else "")
                  + f"({rec['epoch_seconds']:.1f}s)", flush=True)
            if stop:
                break

        info = dict(stop_epoch=ep, stop_reason=stop_reason, max_epochs=max_epochs,
                    rel_tol=args.rel_tol, patience=args.patience,
                    stop_eval_every=args.stop_eval_every,
                    best_ref_loss=(best if np.isfinite(best) else None), best_ref_epoch=best_epoch,
                    final_train_loss=last.get("train_loss"), final_train_acc=last.get("train_acc"),
                    final_test_loss=last.get("test_loss"), final_test_acc=last.get("test_acc"))
        f.write(json.dumps(dict(event="stop", **info)) + "\n")
    print(f"  stopped at epoch {ep}, reason: {stop_reason}", flush=True)
    return info


def stage_train(args, panel, variant, data, device):
    key = run_key(panel, variant)
    d = os.path.join(args.out, key)
    os.makedirs(d, exist_ok=True)
    N = len(data[0])
    for tag in ("sb", "lb"):
        pt, rj = f"{d}/{tag}.pt", f"{d}/run_{tag}.json"
        if os.path.exists(pt):
            prev = json.load(open(rj)) if os.path.exists(rj) else {}
            if "stop_reason" not in prev:
                raise RuntimeError(f"{pt} was produced by the old fixed-epoch script. Move it aside "
                                   f"or use a new --out so protocols are not mixed.")
            print(f"[{key}] {tag}.pt exists (stopped at epoch {prev['stop_epoch']}, "
                  f"{prev['stop_reason']}), skipping")
            continue
        init_seed = args.seed if (args.init == "shared" or tag == "sb") else args.seed + 1000
        torch.manual_seed(init_seed)
        model = build_net(panel, variant)            # built on CPU: init is device independent
        torch.save(model.state_dict(), f"{d}/init_{tag}.pt")
        model.to(device)
        bs = args.sb_bs if tag == "sb" else int(round(args.lb_frac * N))
        max_epochs = args.caps.get((panel, tag), args.max_epochs)
        model.ckpt = (tag == "lb") and (args.ckpt == "on" or (args.ckpt == "auto" and panel == "C2"))
        print(f"[{key}] training {tag}: bs={bs}, max_epochs={max_epochs}, ckpt={model.ckpt}", flush=True)
        t0 = time.time()
        info = train_run(model, data, bs, max_epochs, args.seed * 10 + (1 if tag == "sb" else 2),
                         f"{d}/log_{tag}.jsonl", args, device)
        model.ckpt = False
        # run json first, weights last: the .pt file is the completion marker
        with open(rj, "w") as f:
            json.dump(dict(args={k: v for k, v in vars(args).items() if k != "caps"},
                           key=key, batch_size=bs, device=str(device), torch=torch.__version__,
                           wall_seconds=time.time() - t0, **info), f, indent=1)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, pt)
        del model
        free_mem(device)


# ------------------------------------------------------------- interpolation
def param_vector(sd, pnames):
    return torch.cat([sd[n].flatten().double() for n in pnames])


def stage_interp(args, panel, variant, data, device):
    key = run_key(panel, variant)
    d = os.path.join(args.out, key)
    sd_s = torch.load(f"{d}/sb.pt", map_location="cpu")
    sd_l = torch.load(f"{d}/lb.pt", map_location="cpu")
    model = build_net(panel, variant).to(device)
    model.ckpt = False
    pnames = [n for n, _ in model.named_parameters()]

    # distance-from-init ratio (the paper reports 3 to 10 for ||xs-x0|| / ||xl-x0||)
    x0s = param_vector(torch.load(f"{d}/init_sb.pt", map_location="cpu"), pnames)
    x0l = param_vector(torch.load(f"{d}/init_lb.pt", map_location="cpu"), pnames)
    dist_ratio = (param_vector(sd_s, pnames) - x0s).norm().item() / \
                 (param_vector(sd_l, pnames) - x0l).norm().item()

    # endpoints evaluated with their own trained running BN stats (consistency check)
    native = {}
    for tag, sd in (("sb", sd_s), ("lb", sd_l)):
        model.load_state_dict(sd)
        native[tag] = evaluate_both(model, data, args.eval_bs)

    keep = ("batch_size", "max_epochs", "stop_epoch", "stop_reason", "best_ref_epoch",
            "final_train_loss", "final_train_acc", "final_test_loss", "final_test_acc")
    training = {}
    for tag in ("sb", "lb"):
        info = json.load(open(f"{d}/run_{tag}.json"))
        training[tag] = {k: info[k] for k in keep if k in info}

    alphas = np.linspace(-1.0, 2.0, args.n_alpha)
    keys = ("train_loss", "train_acc", "test_loss", "test_acc")
    res = {k: [] for k in keys}
    t0 = time.time()
    for n, a in enumerate(alphas):
        a = float(a)
        sd = {}
        for k in sd_s:
            if sd_s[k].is_floating_point():
                sd[k] = a * sd_l[k] + (1.0 - a) * sd_s[k]   # params (incl. BN gamma, beta) and buffers
            else:
                sd[k] = sd_s[k]                              # num_batches_tracked
        model.load_state_dict(sd)
        if args.bn_mode == "recompute":
            recompute_bn(model, data[0], args.eval_bs)
        r = evaluate_both(model, data, args.eval_bs)
        for k in keys:
            res[k].append(r[k])
        print(f"[{key}] alpha {a:+.3f} train_loss {r['train_loss']:.4f} "
              f"test_loss {r['test_loss']:.4f} ({n + 1}/{len(alphas)}, {time.time() - t0:.0f}s)",
              flush=True)
    with open(f"{d}/curve.json", "w") as f:
        json.dump(dict(panel=key, label=run_label(panel, variant), variant=variant,
                       alpha=[float(a) for a in alphas], bn_mode=args.bn_mode, init=args.init,
                       native_endpoints=native, dist_ratio_sb_over_lb=dist_ratio,
                       training=training, **res), f, indent=1)


# ---------------------------------------------------------------------- plot
def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [(p, v) for p, v in expand_runs(args)
            if os.path.exists(f"{args.out}/{run_key(p, v)}/curve.json")]
    if not runs:
        print("no curve.json files found, nothing to plot")
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5.4 * len(runs), 4.4), squeeze=False)
    for i, (ax, (p, v)) in enumerate(zip(axes[0], runs)):
        c = json.load(open(f"{args.out}/{run_key(p, v)}/curve.json"))
        a = np.array(c["alpha"])
        ax.plot(a, c["train_loss"], "b-", label="Train")
        ax.plot(a, c["test_loss"], "b--", label="Test")
        ax.set_ylim(bottom=0)
        ax.set_xlim(-1, 2)
        ax.set_xlabel("alpha")
        ax.set_ylabel("Cross Entropy", color="b")
        ax.tick_params(axis="y", colors="b")
        ax.grid(True, linestyle=":")
        ax.legend(loc="upper left")
        ax2 = ax.twinx()
        ax2.plot(a, c["train_acc"], "r-")
        ax2.plot(a, c["test_acc"], "r--")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Accuracy", color="r")
        ax2.tick_params(axis="y", colors="r")
        ax.set_title(f"({chr(97 + i)}) {c['label']}")
    fig.tight_layout()
    fig.savefig(f"{args.out}/figure3_reproduction.png", dpi=200)
    print("saved", f"{args.out}/figure3_reproduction.png")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "interp", "plot", "all"])
    ap.add_argument("--panels", nargs="+", default=["F1", "C1", "C2"])
    ap.add_argument("--c1_variant", choices=["alexnet", "collapse", "both"], default="alexnet")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)            # unspecified in paper
    ap.add_argument("--sb_bs", type=int, default=256)
    ap.add_argument("--lb_frac", type=float, default=0.10)
    # stopping rule
    ap.add_argument("--max_epochs", type=int, default=300, help="default per-run epoch cap")
    ap.add_argument("--cap", action="append", default=[],
                    help="per-run cap override, PANEL:sb=N or PANEL:lb=N (repeatable)")
    ap.add_argument("--rel_tol", type=float, default=0.01,
                    help="relative drop in full-train loss that counts as an improvement")
    ap.add_argument("--patience", type=int, default=10,
                    help="stop after this many epochs without a counted improvement")
    ap.add_argument("--stop_eval_every", type=int, default=1,
                    help="epochs between full-train-set loss checks (each costs a forward pass)")
    ap.add_argument("--init", choices=["shared", "independent"], default="shared")
    ap.add_argument("--bn_mode", choices=["recompute", "interp"], default="recompute")
    ap.add_argument("--n_alpha", type=int, default=61)
    ap.add_argument("--eval_bs", type=int, default=2500)
    ap.add_argument("--eval_every", type=int, default=5, help="epochs between train+test logging")
    ap.add_argument("--ckpt", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()
    if args.max_epochs < 1:
        ap.error("--max_epochs must be >= 1")
    try:
        args.caps = parse_caps(args.cap)
    except ValueError as e:
        ap.error(str(e))
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    print("device:", device, "torch:", torch.__version__)

    if args.stage in ("train", "interp", "all"):
        for panel, variant in expand_runs(args):
            data = load_data(DATASET[panel], args.data_root, device)
            if args.stage in ("train", "all"):
                stage_train(args, panel, variant, data, device)
            if args.stage in ("interp", "all"):
                stage_interp(args, panel, variant, data, device)
            del data
            free_mem(device)
    if args.stage in ("plot", "all"):
        stage_plot(args)


if __name__ == "__main__":
    main()
```

Usage:
```
# the stopping rule changes the training protocol, so set the old results aside first
mv runs runs_fixed100

# smoke test
python repro_fig3.py all --max_epochs 2 --n_alpha 5 --out runs_smoke

# F1 and both C1 readings
export PYTORCH_ENABLE_MPS_FALLBACK=1
python repro_fig3.py all --panels F1 C1 --c1_variant both

# C2 with explicit caps and sparser stopping checks (about 1/3 of an epoch of compute per check)
python repro_fig3.py all --panels C2 --cap C2:sb=40 --cap C2:lb=40 --stop_eval_every 5

# combined figure
python repro_fig3.py plot --panels F1 C1 C2
```

# What changed

- **Stopping rule.** `--epochs`, `--sb_epochs` and `--lb_epochs` are gone. New flags are `--max_epochs` (default 300), `--cap PANEL:sb|lb=N`, `--rel_tol` (default 0.01), `--patience` (in epochs, default 10) and `--stop_eval_every` (default 1).
- **Stop logging.** The stopping epoch and reason (`patience`, `max_epochs` or `nonfinite_loss`) go into a final `{"event":"stop"}` line in `log_*.jsonl`, into `run_*.json`, and into `curve.json` under `training`. The final train/test loss and accuracy are recorded there too, so you can compare against Table 2 directly.
- **C1 variants.** `--c1_variant {alexnet,collapse,both}`, with the default now `alexnet`. The flatten size is computed by a dummy forward pass. The `collapse` init draws are unchanged from your earlier run.
- **Directories and plot.** C1 results now live in `runs/C1_<variant>/`, and plot titles show the variant.
- **Safety.** The script refuses to skip weights left by the old fixed-epoch script, and it writes the weights file last so it marks a completed run.
- **Unchanged.** Models other than C1, data, Adam settings, initialization, BN recompute, interpolation and plot style.

# Things to watch

- **The rule may not fire.** An overparameterized net on separable data keeps lowering cross-entropy slowly, so with a strict `rel_tol` the cap will often be what stops training. Check `stop_reason`. The paper gives no tolerance, so `rel_tol` and `patience` are my operationalization of "ceased to improve". Raise `rel_tol` to stop earlier.
- **LB stopping signal.** LB does 10 updates per epoch, so the eval-mode BN running statistics lag more and the patience window contains far fewer updates than for SB. If LB stops early with train accuracy below the paper's, raise `--patience`. Compare `train_loss` (eval mode) against `train_loss_running` in the log if the signal looks erratic.
- **C2 will be capped.** At about 86 and 116 s per epoch, a cap of 40 means roughly 1 and 1.3 hours of training, plus checks and interpolation. C2 likely will not meet the stopping rule in 40 epochs. Treat that as a documented deviation and look at `final_train_acc`.
- **C1 `alexnet` cost.** Its second conv works on 16x16 maps instead of 4x4, so conv compute is roughly 10x the `collapse` version, and LB activations are several times larger. My estimate is a peak of about 6 to 9 GB for LB. I haven't measured it. If it runs out of memory, use `--ckpt on`. The smoke test will give you epoch times.
- **A correction to my earlier note on checkpointing.** Gradients and batch statistics are identical, but BN runs again during the recompute in backward, so running statistics get updated twice per step (effective EMA momentum about 0.19 instead of 0.1). That only affects eval-mode numbers such as the stopping loss for C2 LB, not training dynamics and not the interpolation curves, since those recompute BN.
- **The bump near alpha=0.85.** None of these changes target it. Because BN is recomputed at every alpha, it is not a stale-statistics artifact. A barrier region between two distinct solutions is not by itself a bug, and the retrained endpoints will shift it. Compare it to the paper's C1 panel by eye, and rerun interpolation with `--bn_mode interp` as a cross-check.


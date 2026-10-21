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

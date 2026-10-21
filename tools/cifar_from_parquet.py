"""Convert the Hugging Face CIFAR-10 parquet files into torchvision's cifar-10-batches-py layout.

The official Toronto mirror was too slow (~2.5 KB/s), so the data comes from
huggingface.co/datasets/uoft-cs/cifar10 instead. Same images and labels, but the
pickles are not byte-identical to the originals, so torchvision's md5 check fails;
run training through tools/run_patched.py, which skips only that check.

  python tools/cifar_from_parquet.py data/hf data
"""
import io, pickle, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load(path):
    t = pq.read_table(path).to_pydict()
    imgs = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
    data = imgs.transpose(0, 3, 1, 2).reshape(len(imgs), -1)     # (N, 3072), CHW like the originals
    return data.astype(np.uint8), list(t["label"])


def main(src, dst):
    out = Path(dst) / "cifar-10-batches-py"
    out.mkdir(parents=True, exist_ok=True)
    xtr, ytr = load(Path(src) / "train.parquet")
    xte, yte = load(Path(src) / "test.parquet")
    assert xtr.shape == (50000, 3072) and xte.shape == (10000, 3072)
    for i in range(5):
        s = slice(i * 10000, (i + 1) * 10000)
        with open(out / f"data_batch_{i + 1}", "wb") as f:
            pickle.dump({"data": xtr[s], "labels": ytr[s]}, f)
    with open(out / "test_batch", "wb") as f:
        pickle.dump({"data": xte, "labels": yte}, f)
    with open(out / "batches.meta", "wb") as f:
        pickle.dump({"label_names": CLASSES}, f)
    print("train label counts", np.bincount(ytr), "test", np.bincount(yte))


if __name__ == "__main__":
    main(*sys.argv[1:3])

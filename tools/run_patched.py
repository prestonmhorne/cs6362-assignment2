"""Run a script with torchvision's CIFAR-10 md5 check disabled (data converted by cifar_from_parquet.py).

  python tools/run_patched.py zhang/phase1.py train --grid smoke --data_root data
"""
import runpy, sys

import torchvision.datasets.cifar as cifar
from torchvision.datasets import CIFAR10

cifar.check_integrity = lambda fpath, md5=None: __import__('os').path.isfile(fpath)

CIFAR10._check_integrity = lambda self: True
CIFAR10.download = lambda self: None

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")

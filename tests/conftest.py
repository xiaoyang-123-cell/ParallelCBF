from __future__ import annotations

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy
import torch


_ = numpy.__version__
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

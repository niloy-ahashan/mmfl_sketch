"""
N-modality Independent Compression training: :class:`FedMultiModalN` +
:class:`IndependentCompressionN`.

Same training as ``mm_train_independent_4mod_v2.py`` (summation fusion, one
Count Sketch of the full gradient, compact table, console teed to
``result_updated/IC_<dataset>_mm_<cum_up>MB_..._seed<N>.txt``), with the number
of modalities taken from ``prepare_stats.json``. Used for 12-modality PTB-XL.

Requires ``dataset_dir/prepare_stats.json`` (``mod_dims``, ``num_classes``) and
the client cache from the dataset's prepare script, e.g.
``prepare_ptbxl_12mod_mm.py``.
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import models
from models.independent_compression_net_n import IndependentCompressionN

# So ``parse_args --model IndependentCompressionN`` is a valid choice
# without editing models/__init__.py (mirrors mm_train_independent.py).
models.IndependentCompressionN = IndependentCompressionN

import mm_train as mm_base
from CommEfficient.utils import get_grad
from data_utils import FedSampler
from data_utils.fed_multimodal_n import FedMultiModalN
from fed_aggregator import FedModel, FedOptimizer
from mm_train_independent import _Tee, _result_updated_dir, _txt_stem
from mm_train_independent_4mod import _warn_if_unused_flags
from utils import (
    PiecewiseLinear,
    TableLogger,
    Timer,
    make_logdir,
    parse_args,
    steps_per_epoch,
)

import torch.multiprocessing as multiprocessing

_ce_criterion = torch.nn.CrossEntropyLoss(reduction="mean")


def compute_loss(model, batch, args):
    del args  # IC loss has no missing-modality / MFM / similarity terms
    *mods, targets = batch
    pred, _H = model(*mods)
    loss = _ce_criterion(pred, targets)
    accuracy = (pred.max(dim=1)[1] == targets).float().mean()
    return loss, accuracy


def _load_prepare_stats(args):
    path = os.path.join(args.dataset_dir, "prepare_stats.json")
    with open(path, encoding="utf-8") as f:
        st = json.load(f)
    args.mod_dims = [int(x) for x in st["mod_dims"]]
    if args.num_classes is None:
        args.num_classes = int(st["num_classes"])


def get_data_loaders(args):
    train_dataset = FedMultiModalN(
        args.dataset_dir, args.dataset_name, transform=None,
        do_iid=args.do_iid, num_clients=args.num_clients,
        train=True, download=True,
    )
    test_dataset = FedMultiModalN(
        args.dataset_dir, args.dataset_name, transform=None,
        train=False, download=False,
    )
    train_sampler = FedSampler(
        train_dataset, args.num_workers, args.local_batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.train_dataloader_workers,
        pin_memory=True,
    )
    test_batch_size = args.valid_batch_size * args.num_workers
    n_te = len(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.val_dataloader_workers,
        pin_memory=True,
    )
    print(
        f"Validation drop_last=False: evaluating all {n_te} test samples "
        f"(batch_size={test_batch_size})"
    )
    print(len(train_loader), len(test_loader))
    return train_loader, test_loader


def _run(args):
    _warn_if_unused_flags(args)
    _load_prepare_stats(args)
    print("MY PID:", os.getpid())

    timer = Timer()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, test_loader = get_data_loaders(args)
    print("Detected single-label dataset — using CrossEntropyLoss")

    model_config = {
        "mod_dims": tuple(args.mod_dims),
        "feat_dim": args.feat_dim,
        "num_classes": args.num_classes,
        "dropout": args.mm_dropout,
    }
    print(f"IndependentCompressionN config: {model_config}")
    print(
        f"Fusion: summation (sum of {len(args.mod_dims)} refined modality "
        "features).  Gradient compression: original FetchSGD Count Sketch."
    )
    model = IndependentCompressionN(**model_config)

    param_groups = model.parameters()
    opt = optim.SGD(param_groups, lr=1)

    model = FedModel(model, compute_loss, args, compute_loss)
    opt = FedOptimizer(opt, args)

    if args.mode != "fedavg":
        lr_schedule = PiecewiseLinear(
            [0, args.pivot_epoch, args.num_epochs],
            [0, args.lr_scale, 0],
        )
        spe = steps_per_epoch(
            args.local_batch_size, train_loader.dataset, args.num_workers,
        )
        lr_scheduler = LambdaLR(
            opt, lr_lambda=lambda step: lr_schedule(step / spe),
        )
    else:
        lr_scheduler = None

    log_dir = make_logdir(args)
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    print("Finished initializing in {:.2f} seconds".format(timer()))

    _spe0 = steps_per_epoch(
        args.local_batch_size, train_loader.dataset, args.num_workers,
    )
    _mle = max(float(getattr(args, "mm_local_epochs", 1.0)), 1.0)
    print(
        "Federated train: steps_per_epoch(spe)={:.0f} (CommEfficient "
        "rounds/epoch; LR divisor=spe). mm_local_epochs={:.3g} is local "
        "SGD per round, not extra uploads.".format(_spe0, _mle)
    )

    grad = get_grad(model, args)
    print("Grad size:", grad.numel())
    print("Total params:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))
    print("Hyperparameters (resolved):")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    summary = mm_base.train(
        model, opt, lr_scheduler, train_loader, test_loader, args,
        writer, loggers=(TableLogger(),), timer=timer,
        compact_console=True,
    )
    model.finalize()
    if summary and "cum_up_MB" in summary:
        return summary["cum_up_MB"]
    return None


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = parse_args()
    args.model = "IndependentCompressionN"
    # Retrieval MAP only exists for the 2-modality extract_fused API.
    args.skip_map = True

    log_buf = io.StringIO()
    stdout_orig = sys.stdout
    sys.stdout = _Tee(stdout_orig, log_buf)
    cum_up_mb = None
    try:
        cum_up_mb = _run(args)
    finally:
        sys.stdout = stdout_orig
        updated_dir = _result_updated_dir()
        os.makedirs(updated_dir, exist_ok=True)
        txt_path = os.path.join(updated_dir, _txt_stem(args, cum_up_mb) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")

"""PMFH + federated head with a **logit-adjusted** head loss ("PMFHla").

Same run as ``main_uci_har_v2.py --variant fedhead`` (official PMFH training, Linear
head on ``nec_vec.detach()``, head FedAvg'd with the global model, test_acc = head
argmax, 8-sample shard mean), except the head's CrossEntropy during local training is
logit-adjusted with the client's own label prior (Menon et al., ICLR 2021; FedLC-style
label-skew correction):

    CE(logits + tau * log pi_client, y)          (training)
    argmax(logits)                               (test: raw logits)

Under Dirichlet(0.1) most clients are dominated by one class; with plain CE each
client's head learns its own prior and the unweighted FedAvg of those heads predicts
the global majority class (PTB-XL: NORM, 56.33% for every seed). The adjustment moves
the prior out of the head's weights.

Unchanged: PMFH's losses and parameters (the head input stays detached), the model's
state_dict (``log_prior`` is a plain attribute, so upload bytes are identical),
FedAvg, prototypes, data, metric. With ``--la_tau 0`` the run is PMFHfh exactly.

Nothing in ``main_uci_har_v2.py`` is edited: this file imports it and swaps three
names in memory (``LayersFedHead``, ``LocalUpdateUCI``, the log/npz prefix), then
calls its ``main()``. Logs: ``result_updated/PMFHla_<dataset>_mm_<MB>MB_dirichlet.._tau<t>_it.._seed<s>.txt``.
"""

import argparse
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
from torch import nn

import main_uci_har_v2 as base
from layers_uci import LayersFedHead

LA_EPS = 1e-3  # pseudo-count so classes a client lacks get a finite log prior


class LAHead(nn.Linear):
    """``nn.Linear`` that adds ``tau * log_prior`` to its logits in training mode only."""

    def __init__(self, in_features, out_features, tau):
        super().__init__(in_features, out_features)
        self.tau = float(tau)
        self.log_prior = None  # plain attribute: not in state_dict, never uploaded

    def forward(self, x):
        out = super().forward(x)
        if self.training and self.log_prior is not None and self.tau != 0.0:
            out = out + self.tau * self.log_prior.to(out.device, out.dtype)
        return out


class LayersFedHeadLA(LayersFedHead):
    """``LayersFedHead`` whose head is an ``LAHead`` with the same initial weights."""

    def __init__(self, args):
        super().__init__(args)
        old = self.head
        self.head = LAHead(old.in_features, old.out_features, args.la_tau)
        with torch.no_grad():
            self.head.weight.copy_(old.weight)
            self.head.bias.copy_(old.bias)


class LocalUpdateLA(base.LocalUpdateUCI):
    """``LocalUpdateUCI`` with the client's label prior set on the head during its update."""

    def __init__(self, args, id, dataset, idxs, idcs_retrieval, idcs_query):
        super().__init__(args, id, dataset, idxs, idcs_retrieval, idcs_query)
        counts = np.asarray(dataset[0])[np.asarray(idxs, dtype=np.int64)].sum(axis=0).astype(np.float64)
        prior = (counts + LA_EPS) / (counts.sum() + LA_EPS * len(counts))
        self.log_prior = torch.tensor(np.log(prior), dtype=torch.float32)

    def update_weights_UCCH_new(self, args, global_protos, global_avg_protos, model_img, private_modal):
        model_img.head.log_prior = self.log_prior
        try:
            return super().update_weights_UCCH_new(args, global_protos, global_avg_protos, model_img, private_modal)
        finally:
            model_img.head.log_prior = None


_orig_args_parser = base.args_parser
_orig_txt_stem = base._txt_stem
_orig_run = base._run


def _args_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--la_tau", type=float, default=1.0,
                   help="logit-adjustment strength tau for the head CE (0 = plain PMFHfh)")
    la, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = _orig_args_parser()
    if args.variant != "fedhead" or args.head_grad != "detached":
        raise SystemExit("main_pmfh_lahead.py needs --variant fedhead --head_grad detached")
    args.la_tau = la.la_tau
    return args


def _txt_stem(args, cum_up_mb, n_active):
    stem = _orig_txt_stem(args, cum_up_mb, n_active)
    return stem.replace(f"_it{args.rounds}_", f"_tau{args.la_tau:g}_it{args.rounds}_", 1)


def _run(args):
    print(f"Head loss: logit-adjusted CE, tau={args.la_tau:g} (client label prior, eps={LA_EPS:g}; "
          f"raw logits at test)")
    return _orig_run(args)


base.LayersFedHead = LayersFedHeadLA
base.LocalUpdateUCI = LocalUpdateLA
base.args_parser = _args_parser
base._prefix = lambda args: "PMFHla"
base._txt_stem = _txt_stem
base._run = _run


if __name__ == "__main__":
    base.main()

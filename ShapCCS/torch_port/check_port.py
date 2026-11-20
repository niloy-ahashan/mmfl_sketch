"""Check that gpsv.py computes what the released ShapCCS code computes.

Imports the vendored upstream modules (ShapCCS/upstream) with a stand-in for their ``args.py``, feeds the
upstream and ported functions identical inputs and random seeds, and compares the outputs:
  1. train_client_weight: local SGD result (state dict)
  2. cal_gpsv: per-client GPSV, exact Gray-code branch (2 and 5 clients) and GTG branch (16 clients)
  3. select_clients: round-robin pretraining picks and random picks afterwards
  4. prune_by_sv: kept clients, mean GPSV and the samples each client keeps (moderate and random, both
     prune fractions used by the runners)
  5. multimodal smoke test: the ported functions on tuple inputs with ShapCCSNet
  6. source diff: every line of gpsv.py that differs from upstream must carry a ``# PORT:`` marker
No dataset is needed. Run:  MFedMC/.venv/bin/python ShapCCS/torch_port/check_port.py
"""

import argparse
import copy
import difflib
import inspect
import os
import sys
import types

HERE = os.path.abspath(os.path.dirname(__file__))
UPSTREAM = os.path.abspath(os.path.join(HERE, "..", "upstream"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ARGS = argparse.Namespace(
    device="cpu", dataset="fmnist", lr=0.05, rounds=100, num_epochs=1, pre_train_round=10,
    client_selected_ratio=0.2, prune_frac=0.7, train_ratio=1.0, num_clients_remain=2, ml_method="moderate",
    num_classes=4, dirichlet_alpha=0.1, noisy_client_ratio=0.0, batch_size=8, test_batchsize=8,
)

# upstream modules do `from args import args`
_args_mod = types.ModuleType("args")
_args_mod.args = ARGS
sys.modules["args"] = _args_mod
sys.path.insert(0, UPSTREAM)
from coreset_selection import shap_ccs as up_shap  # noqa: E402
from coreset_selection.utils import moderate_utils as up_mod  # noqa: E402
from coreset_selection.utils import sv_utils as up_sv  # noqa: E402
from train import client as up_client  # noqa: E402
from train import server as up_server  # noqa: E402

sys.path.pop(0)
sys.path.insert(0, HERE)
import gpsv  # noqa: E402
from dataset_mm import PrunableDataset  # noqa: E402

gpsv.configure(ARGS)
torch.set_default_dtype(torch.float32)
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


class TinyNet(nn.Module):
    """No-argument constructor, as upstream's copy_global_model requires."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(6, 16), nn.ReLU())
        self.output = nn.Linear(16, 4)

    def forward(self, x):
        return self.output(self.backbone(x))

    def get_backbone(self, x):
        return self.backbone(x)


class Pairs(Dataset):
    def __init__(self, x, y):
        self.x, self.targets = x, y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], int(self.targets[i])


def make_loaders(sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = sum(sizes)
    pooled = Pairs(torch.randn(n, 6, generator=g), torch.randint(0, 4, (n,), generator=g))
    loaders, start = [], 0
    for s in sizes:
        loaders.append(DataLoader(PrunableDataset(pooled, range(start, start + s)), batch_size=8, shuffle=True))
        start += s
    return loaders


def states_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def seeded(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)


# 1. local training. Upstream copies the model with __class__(), which runs a fresh weight init and so
# consumes random numbers before the DataLoader shuffles; the port's deepcopy does not. The batch order
# would differ for that reason alone, so this check uses an unshuffled loader to compare the computation.
torch.manual_seed(0)
net = TinyNet()
loader = make_loaders([37])[0]
loader = DataLoader(loader.dataset, batch_size=8, shuffle=False)
w = torch.ones(len(loader.dataset))
seeded()
s_up = up_client.train_client_weight(loader, net, 3, w)
seeded()
s_po = gpsv.train_client_weight(loader, net, 3, w)
check("train_client_weight", states_equal(s_up, s_po))
check("get_weights", np.allclose(up_server.get_weights([0, 1], make_loaders([37, 90])),
                                 gpsv.get_weights([0, 1], make_loaders([37, 90]))))

# 2. GPSV
for n_sel in (2, 5, 16):
    torch.manual_seed(n_sel)
    net = TinyNet()
    base = net.state_dict()
    sel = np.arange(n_sel)
    weights = np.random.RandomState(n_sel).rand(n_sel)
    weights = weights / weights.sum()
    client_states = {int(c): {k: v + 0.05 * torch.randn_like(v) for k, v in base.items()} for c in sel}
    glob = None
    for i, c in enumerate(sel):
        glob = up_server.weight_aggregate(glob, client_states[int(c)], weights[i])
    last = {int(c): up_sv.get_last_layer_state(client_states[int(c)]) for c in sel}
    rs_up, rs_po = [[] for _ in range(n_sel)], [[] for _ in range(n_sel)]
    seeded()
    up_shap.cal_gpsv(net, glob, sel, weights.copy(), last, rs_up)
    seeded()
    gpsv.cal_gpsv(net, glob, sel, weights.copy(), last, rs_po)
    a, b = np.array(rs_up).ravel(), np.array(rs_po).ravel()
    branch = "exact" if n_sel <= 15 else "GTG"
    check(f"cal_gpsv {n_sel} clients ({branch})", np.array_equal(a, b), f"GPSV {np.round(a, 4).tolist()}")

# 3. client selection
for label, rounds in (("pretraining", range(10)), ("after selection", range(10, 14))):
    counts_up, counts_po = {c: 0 for c in range(10)}, {c: 0 for c in range(10)}
    seeded()
    picks_up = [up_sv.select_clients(counts_up, np.arange(10), 2, r).tolist() for r in rounds]
    seeded()
    picks_po = [gpsv.select_clients(counts_po, np.arange(10), 2, r).tolist() for r in rounds]
    check(f"select_clients ({label})", picks_up == picks_po, f"{picks_up}")

# 4. budget allocation, client exclusion and data-level pruning
sizes = [5, 160, 40, 300, 12, 75, 220, 8, 130, 50]
rng = np.random.RandomState(7)
round_sv = [list(rng.rand(2) - 0.2) for _ in sizes]
for ml in ("moderate", "random"):
    for pf, rem in ((0.7, 2), (0.3, 7)):
        ARGS.ml_method, ARGS.prune_frac, ARGS.num_clients_remain = ml, pf, rem
        torch.manual_seed(1)
        net = TinyNet()
        lo_up, lo_po = make_loaders(sizes), make_loaders(sizes)
        cw = {c: torch.ones(s) for c, s in enumerate(sizes)}
        seeded()
        kept_up, sv_up = up_sv.prune_by_sv(round_sv, lo_up, net, None, copy.deepcopy(cw))
        seeded()
        kept_po, sv_po = gpsv.prune_by_sv(round_sv, lo_po, net, None, copy.deepcopy(cw))
        same_idx = all(sorted(a.dataset.active_global_indices) == sorted(b.dataset.active_global_indices)
                       for a, b in zip(lo_up, lo_po))
        kept_n = {int(c): lo_po[c].dataset.get_current_length() for c in kept_po}
        check(f"prune_by_sv {ml} pf{pf} rem{rem}",
              list(map(int, kept_up)) == list(map(int, kept_po)) and sv_up == sv_po and same_idx,
              f"kept {kept_n} ({sum(kept_n.values())} of {sum(sizes)} samples, budget {round((1 - pf) * sum(sizes))})")
ARGS.ml_method, ARGS.prune_frac, ARGS.num_clients_remain = "moderate", 0.7, 2

# 5. multimodal smoke test through the port
from main import ShapCCSNet  # noqa: E402


class MM(Dataset):
    def __init__(self, n):
        g = torch.Generator().manual_seed(3)
        self.xs = [torch.randn(n, 7, generator=g), torch.randn(n, 5, generator=g), torch.randn(n, 3, generator=g)]
        self.targets = torch.randint(0, 4, (n,), generator=g)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return tuple(x[i] for x in self.xs), int(self.targets[i])


mm = MM(120)
mm_loaders = [DataLoader(PrunableDataset(mm, range(i * 40, (i + 1) * 40)), batch_size=8, shuffle=True)
              for i in range(3)]
mnet = ShapCCSNet([7, 5, 3], 4, feat_dim=8, dropout=0.0)
st = gpsv.train_client_weight(mm_loaders[0], mnet, 1, torch.ones(40))
ids = gpsv.prune_moderate(mm_loaders[1], mnet, budget=10)
check("multimodal train + Moderate", "output.weight" in st and len(ids) == 30,
      f"output layer {tuple(st['output.weight'].shape)}, Moderate prunes {len(ids)} of 40 for budget 10")

# 6. source diff
pairs = [
    (up_server.get_weights, gpsv.get_weights), (up_server.weight_aggregate, gpsv.weight_aggregate),
    (up_client.copy_global_model, gpsv.copy_global_model), (up_client.get_lr, gpsv.get_lr),
    (up_client.train_client_weight, gpsv.train_client_weight),
    (up_sv.get_last_layer_state, gpsv.get_last_layer_state), (up_sv.get_weight_delta, gpsv.get_weight_delta),
    (up_sv.get_grad_cos_sim, gpsv.get_grad_cos_sim), (up_sv.select_clients, gpsv.select_clients),
    (up_sv.prune_by_sv, gpsv.prune_by_sv), (up_mod.get_median, gpsv.get_median),
    (up_mod.get_distance, gpsv.get_distance), (up_mod.get_features, gpsv.get_features),
    (up_mod.prune_moderate, gpsv.prune_moderate), (up_shap.shap_ccs, gpsv.shap_ccs),
    (up_shap.cal_gpsv, gpsv.cal_gpsv),
]
print("\nLines of gpsv.py that differ from upstream:")
unmarked = []
for f_up, f_po in pairs:
    a, b = inspect.getsource(f_up).splitlines(), inspect.getsource(f_po).splitlines()
    plus = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("+") and not l.startswith("+++")]
    minus = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("-") and not l.startswith("---")]
    if plus or minus:
        print(f"  {f_po.__name__}: {len(minus)} upstream line(s) replaced/removed, {len(plus)} line(s) added")
        for l in plus:
            print(f"    {l}")
            if "# PORT" not in l:
                unmarked.append((f_po.__name__, l))
    else:
        print(f"  {f_po.__name__}: identical")
check("every changed line is marked # PORT", not unmarked, f"unmarked: {unmarked}" if unmarked else "")

print(f"\n{'All checks passed' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)

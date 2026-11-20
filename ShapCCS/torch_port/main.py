"""ShapCCS (Ji, Hu, He, Zhao, Li, Xu; ICML 2026) on the shared UCI HAR / mHealth split.

Paper: ShapCCS/ShapCCS.pdf ("ShapCCS: Shapley-Driven Client Coreset Selection in Federated Learning",
openreview.net/forum?id=PnabunIVDL). Code: ShapCCS/upstream = github.com/iz70778/ShapCCS @ 8a45bbb, vendored
unchanged. This port runs the released path ``scripts/prune.py --option fed_sv_cs_super --ml_method moderate``
(``ShapCCS`` with Moderate as the data-level method, the paper's default). The selection and training code is
upstream's, copied into gpsv.py (changes marked ``# PORT:``, checked against upstream by check_port.py).

Algorithm (released code):
  * FedAvg: ``ceil(client_ratio * |clients|)`` clients per round, each trains a copy of the global model for
    ``local_epochs`` with plain SGD (lr 0.05 * 0.99^round, batch 32); the server averages the returned state
    dicts weighted by each client's number of batches.
  * rounds < pre_train_round (10): clients are picked round-robin (least picked first); after each round the
    GPSV of the picked clients is computed from the change in the final ``output`` layer: projection
    lambda_i = cos(d_i, d) * |d_i| onto the aggregate change d, coalition value = sample-weighted mean of
    lambda_i clipped by psi (1.25 / 1.75 / 3 * |d|), exact Shapley over all coalitions (Gray-code order),
    normalised to sum to 1.
  * round pre_train_round: each client's mean GPSV sets its share of the budget B = (1 - prune_frac) * N
    samples (shifted by the minimum, l1-normalised); in ascending GPSV order, clients whose budget is below
    tau = B / num_clients_remain are dropped and their budget is spread over the rest; each kept client
    shrinks its data to its budget with Moderate (distance of the backbone feature to its class median).
  * later rounds: clients are drawn at random from the kept clients only; plain FedAvg on the reduced data.

Multimodal adapter (the only model change): each modality gets HybridSketchMFedMC_v2's FeaExtractor +
FeaRefiner (``--feat_dim``, default 64), the refined features are concatenated (the "backbone", which
Moderate reads), and one ``output`` Linear classifies them. Upstream's models are ``backbone -> output``
too, and GPSV reads only ``output.weight`` / ``output.bias``, so the selection code runs unchanged for any
number of modalities.

Settings that differ from upstream (FMNIST: 100 clients, 15% per round), with the reason:
  * 10 clients, Dirichlet(0.1) shared split, every client sample is training data (no 5% validation
    hold-out; the Moderate path never reads it): shared protocol.
  * --client_ratio 0.2 (2 clients per round): chosen to match Hybrid v2's mHealth 5-10 MB configs.
  * --num_clients_remain scaled with the client count: upstream 20 of 100 at prune_frac 0.7 and 70 of 100
    at 0.3 (paper Sec. 5.1: tau = B/20 and B/70) -> 2 and 7 of 10.
  * test_acc = 8-sample shard mean on the shared test set (as the baselines); upstream's whole-test-set
    accuracy is kept in the npz as ``test_acc_full``.

Paper vs. release: they agree on psi, lambda_i, the budget split, tau and the redistribution. The paper samples
pretraining clients at random (Alg. 1 line 4); the release picks them round-robin. Learning rate, batch size and
local epochs are not in the paper; they come from the release's args.py (FMNIST settings). This port follows
the release.

Communication: each picked client downloads the global model and uploads its trained model every round
(4 bytes per parameter; the model has no buffers). Dropped clients send nothing after the selection round.
The selection itself needs no extra transfers (GPSV is computed from the uploaded models). MB = 1e6 bytes.
"""

import argparse
import io
import os
import random
import sys
import time

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(ROOT, "HybridSketchMFedMC_v2"))

import numpy as np
import torch
from torch import nn

import gpsv
from dataset_mm import build_client_loaders, load_shared_split
from federated import evaluate_model_ic_shards  # HybridSketchMFedMC_v2
from models import FeaExtractor, FeaRefiner  # HybridSketchMFedMC_v2
from utils import TableLogger, Timer  # HybridSketchMFedMC_v2


def args_parser():
    p = argparse.ArgumentParser(description="ShapCCS (ICML'26) on the shared UCI HAR / mHealth split")
    p.add_argument("--dataset_dir", type=str, required=True, help="e.g. datasets/uci_har_mm or datasets/mhealth_mm")
    p.add_argument("--dataset", choices=["auto", "uci_har", "mhealth"], default="auto",
                   help="loader; auto = mhealth if 'mhealth' is in the folder name, else uci_har")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--dirichlet_alpha", type=float, default=0.1)
    # ShapCCS (released args.py / run_prune.py; FMNIST settings)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--pre_train_round", type=int, default=10, help="GPSV pretraining rounds (paper: 0.1 T)")
    p.add_argument("--client_ratio", type=float, default=0.2, help="clients per round (released: 0.15 of 100)")
    p.add_argument("--prune_frac", type=float, default=0.7, help="fraction of the data removed (1 - r)")
    p.add_argument("--num_clients_remain", type=int, default=2,
                   help="tau = B / num_clients_remain (released: 20 at prune_frac 0.7, 70 at 0.3, for 100 clients)")
    p.add_argument("--ml_method", choices=["moderate", "random"], default="moderate",
                   help="data-level method inside ShapCCS (released default: moderate)")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--local_epochs", type=int, default=1)
    # encoder (Hybrid v2's, as the FedMobile / FLASH shared ports)
    p.add_argument("--feat_dim", type=int, default=64, help="encoder width per modality")
    p.add_argument("--mm_dropout", type=float, default=0.3)
    p.add_argument("--valid_batch_size", type=int, default=8, help="test shard size (as the baselines)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--results_dir", type=str, default=os.path.join(HERE, "results"))
    return p.parse_args()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


class ShapCCSNet(nn.Module):
    """Per-modality Hybrid v2 encoder -> concat ("backbone") -> ``output`` Linear (upstream Base_Model shape).

    ``forward(xs)`` takes a list with one tensor per modality and returns logits, as upstream's models do.
    """

    def __init__(self, mod_dims, num_classes, feat_dim=64, dropout=0.3):
        super().__init__()
        self.mod_dims = [int(d) for d in mod_dims]
        self.extractors = nn.ModuleList([FeaExtractor(d, feat_dim, dropout) for d in self.mod_dims])
        self.refiners = nn.ModuleList([FeaRefiner(feat_dim) for _ in self.mod_dims])
        for m in self.modules():  # Hybrid v2's encoder init; the output layer keeps PyTorch's, as upstream
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.output = nn.Linear(feat_dim * len(self.mod_dims), num_classes)

    def get_backbone(self, xs):
        feats = []
        for x, ext, ref in zip(xs, self.extractors, self.refiners):
            f = ext(x)
            feats.append(f * ref(f))
        return torch.cat(feats, dim=1)

    def forward(self, xs):
        return self.output(self.get_backbone(xs))


class _ShardView(nn.Module):
    """Adapter for evaluate_model_ic_shards, which calls model(*mods) and expects (logits, feature)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, *mods):
        return self.net(list(mods)), None


@torch.no_grad()
def _full_test_acc(model, test, device, bs=256):
    """Upstream validate_model: accuracy over the whole test set."""
    model.eval()
    xs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in test["xs"]]
    y = torch.as_tensor(test["y"], dtype=torch.long, device=device)
    correct = 0
    for s in range(0, len(y), bs):
        correct += (model([x[s:s + bs] for x in xs]).argmax(1) == y[s:s + bs]).sum().item()
    return correct / max(len(y), 1)


def _run(args):
    timer = Timer()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print(f"Requested {args.device} but CUDA is unavailable — using {device}")

    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    print(f"Loading the shared split from {dataset_name}...")
    clients_raw, test_raw, meta = load_shared_split(args.dataset_dir, args.dataset, args.num_clients,
                                                    args.dirichlet_alpha, args.seed)
    kind = "mhealth" if "mhealth" in dataset_name else "uci_har"
    if args.dataset != "auto":
        kind = args.dataset
    modalities, num_classes, mod_dims = meta["modalities"], meta["num_classes"], meta["mod_dims"]
    client_loaders, pooled = build_client_loaders(clients_raw, args.bs)
    n_clients = len(client_loaders)

    # upstream's global args namespace, filled from our flags
    gpsv.configure(argparse.Namespace(
        device=device, dataset=kind, lr=args.lr, rounds=args.rounds, num_epochs=args.local_epochs,
        pre_train_round=args.pre_train_round, client_selected_ratio=args.client_ratio,
        prune_frac=args.prune_frac, train_ratio=1.0, num_clients_remain=args.num_clients_remain,
        ml_method=args.ml_method, num_classes=num_classes, dirichlet_alpha=args.dirichlet_alpha,
        noisy_client_ratio=0.0, batch_size=args.bs, test_batchsize=args.valid_batch_size,
    ))

    model = ShapCCSNet(mod_dims, num_classes, args.feat_dim, args.mm_dropout).to(device)
    view = _ShardView(model)
    n_params = sum(p.numel() for p in model.parameters())
    model_bytes = 4 * sum(t.numel() for t in model.state_dict().values())

    print(f"Clients: {n_clients}, {args.client_ratio:g} per round "
          f"(= {int(np.ceil(n_clients * args.client_ratio))} before selection)")
    print(f"  samples/client: {meta['samples_per_client']}")
    print(f"Model: per-modality Hybrid v2 encoder {mod_dims} -> {args.feat_dim}, concat "
          f"{args.feat_dim * len(mod_dims)} -> output -> {num_classes}; params {n_params}; bytes {model_bytes}")
    print(f"ShapCCS: pretrain {args.pre_train_round} rounds, prune_frac {args.prune_frac:g} "
          f"(budget {(1 - args.prune_frac) * len(pooled):.0f} of {len(pooled)} samples), "
          f"tau = B / {args.num_clients_remain}, data-level method {args.ml_method}")
    print("Finished initializing in {:.2f} seconds".format(timer()))
    print("Hyperparameters (resolved):")
    for key, v in sorted(vars(args).items()):
        print(f"  {key}: {v}")

    history = {k: [] for k in ("test_acc", "test_loss", "test_acc_full", "upload_bytes", "download_bytes",
                               "time_s", "n_kept", "data_kept")}
    totals = {"up": 0.0, "down": 0.0, "prev": 0.0}
    logger = TableLogger()
    train_start = time.perf_counter()

    def on_round(cur_round, selected_clients, remained_clients):
        k = len(selected_clients)
        up_bytes = down_bytes = k * model_bytes
        test_loss, test_acc = evaluate_model_ic_shards(view, test_raw, device, shard_size=args.valid_batch_size)
        acc_full = _full_test_acc(model, test_raw, device)
        elapsed = time.perf_counter() - train_start
        round_s, totals["prev"] = elapsed - totals["prev"], elapsed
        totals["up"] += up_bytes / 1e6
        totals["down"] += down_bytes / 1e6
        data_kept = sum(client_loaders[c].dataset.get_current_length() for c in remained_clients)
        logger.append({
            "iter": cur_round + 1,
            "test_acc": test_acc,
            "clients": k,
            "up_MB": up_bytes / 1e6,
            "cum_up_MB": totals["up"],
            "down_MB": down_bytes / 1e6,
            "cum_down_MB": totals["down"],
            "round_s": round_s,
            "time_s": elapsed,
        })
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["test_acc_full"].append(acc_full)
        history["upload_bytes"].append(up_bytes)
        history["download_bytes"].append(down_bytes)
        history["time_s"].append(elapsed)
        history["n_kept"].append(len(remained_clients))
        history["data_kept"].append(data_kept)
        return acc_full

    _, remained, round_sv, sv, selected = gpsv.shap_ccs(model, client_loaders, on_round)

    kept = sorted(int(c) for c in remained)
    kept_data = {c: client_loaders[c].dataset.get_current_length() for c in kept}
    mean_sv = np.array([float(np.mean(v)) if len(v) else 0.0 for v in round_sv])
    print(f"Mean GPSV per client: {np.round(mean_sv, 4).tolist()}")
    print(f"Kept clients after round {args.pre_train_round}: {kept} "
          f"(samples kept {kept_data}; {sum(kept_data.values())} of {len(pooled)})")
    total_up, total_down = totals["up"], totals["down"]
    print("Total Download (MB): {:0.2f}".format(total_down))
    print("Total Upload (MB): {:0.2f}".format(total_up))
    print("Avg Download Per Client: {:0.4f}".format(total_down / max(n_clients, 1)))
    print("Avg Upload Per Client: {:0.4f}".format(total_up / max(n_clients, 1)))
    print("Cumulative train comm (download+upload): {:.4f} MB".format(total_down + total_up))
    print(f"Final test accuracy: {float(history['test_acc'][-1]):.4f} "
          f"({100.0 * float(history['test_acc'][-1]):.2f}%)")

    os.makedirs(args.results_dir, exist_ok=True)
    file_name = os.path.join(args.results_dir, _npz_name(args, dataset_name, n_clients))
    max_sel = max(len(s) for s in selected)
    np.savez(
        file_name,
        test_loss=np.array(history["test_loss"]),
        test_acc=np.array(history["test_acc"]),
        test_acc_full=np.array(history["test_acc_full"]),
        upload_bytes=np.array(history["upload_bytes"]),
        download_bytes=np.array(history["download_bytes"]),
        time_s=np.array(history["time_s"]),  # cumulative training time per round (log's time_s)
        n_kept=np.array(history["n_kept"]),
        data_kept=np.array(history["data_kept"]),
        selected=np.array([s + [-1] * (max_sel - len(s)) for s in selected]),  # -1 = padding
        gpsv_mean=mean_sv,
        kept_clients=np.array(kept),
        kept_samples=np.array([kept_data[c] for c in kept]),
        modalities=np.array(modalities),
        dataset_dir=np.array(meta["dataset_dir"]),
        samples_per_client=np.array(meta["samples_per_client"]),
        total_download_mib=total_down,  # MB (1e6), key name as the other ports' npz
        total_upload_mib=total_up,
        n_params=n_params,
        model_bytes=model_bytes,
        client_ratio=np.array(args.client_ratio),
        prune_frac=np.array(args.prune_frac),
        num_clients_remain=np.array(args.num_clients_remain),
        feat_dim=np.array(args.feat_dim),
    )
    print(f"Results saved to {file_name}")
    return total_up


def _variant_tag(args):
    """Settings in every name: client ratio, prune fraction, tau divisor; extras only when not default."""
    tag = f"C{args.client_ratio:g}_pf{args.prune_frac:g}_rem{args.num_clients_remain}"
    if args.ml_method != "moderate":
        tag += f"_{args.ml_method}"
    if args.feat_dim != 64:
        tag += f"_fd{args.feat_dim}"
    return tag


def _npz_name(args, dataset_name, n_clients):
    return f"ShapCCS_{dataset_name}_K{n_clients}_{_variant_tag(args)}_R{args.rounds}_seed{args.seed}.npz"


def _txt_stem(args, cum_up_mb):
    dataset = os.path.basename(os.path.normpath(args.dataset_dir))
    up_str = f"{cum_up_mb:.2f}MB" if cum_up_mb is not None else "NAMB"
    return (f"ShapCCS_{dataset}_mm_{up_str}_dirichlet{args.dirichlet_alpha:g}_it{args.rounds}_"
            f"{_variant_tag(args)}_lr{args.lr:g}_clients{args.num_clients}_seed{args.seed}")


def main():
    args = args_parser()
    log_buf = io.StringIO()
    stdout_orig = sys.stdout
    sys.stdout = _Tee(stdout_orig, log_buf)
    cum_up_mb = None
    try:
        cum_up_mb = _run(args)
    finally:
        sys.stdout = stdout_orig
        updated_dir = os.path.join(HERE, "result_updated")
        os.makedirs(updated_dir, exist_ok=True)
        txt_path = os.path.join(updated_dir, _txt_stem(args, cum_up_mb) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")


if __name__ == "__main__":
    main()

import json
import os

import numpy as np

from args import args
from scripts.init import global_model, client_loaders, test_loader, validate_loader
from coreset_selection.cs_fl import fed_cs, fed_gc, fed_gc_execute
from coreset_selection.cs_ml import fed_craig, fed_el2n, fed_el2n_execute, fed_ccs, fed_moderate
from coreset_selection.cs_random import fed_random_data_cs, fed_random_data_cs_execute, fed_random_client_cs
from coreset_selection.cs_shapley import fed_other_sv_cs
from coreset_selection.shap_ccs import shap_ccs_max, shap_ccs
from train.server import fed_avg


class PruneResult:
    def __init__(self, round_accuracy, remain_frac, clients_remain_farc):
        self.round_accuracy = round_accuracy
        self.max_accuracy = np.max(round_accuracy)
        self.dirichlet_alpha = args.dirichlet_alpha
        self.pre_train_round = args.pre_train_round
        self.remain_frac = remain_frac
        self.clients_remain_farc = clients_remain_farc

    def to_json(self):
        data = {
            "remain_frac": self.remain_frac,
            "max_accuracy": self.max_accuracy,
            "last_10_rounds_mean_accuracy": np.mean(self.round_accuracy[-10:]),
            "clients_remain_farc": self.clients_remain_farc,
            "dirichlet_alpha": self.dirichlet_alpha,
            "pre_train_round": self.pre_train_round,
            "round_accuracy": self.round_accuracy,
        }

        file_path = f'results_seed_{args.seed}'

        if args.noisy_client_ratio > 0:
            file_path += f'/{args.dirichlet_alpha}/{args.dataset}_{args.noisy_client_ratio}_{args.prune_frac}'
            file_path += f'/{args.noise_ratio}/{args.option}'
        elif args.num_clients != 100:
            file_path += f'/{args.dirichlet_alpha}/{args.dataset}_hyper/{round(args.prune_frac, 1)}/{args.option}'
        elif args.prune_frac == 0.7 and args.num_clients_remain != 20:
            file_path += f'/{args.dirichlet_alpha}/{args.dataset}_tau/{round(args.prune_frac, 1)}/{args.option}'
        else:
            file_path += f'/{args.dirichlet_alpha}/{args.dataset}/{round(args.prune_frac, 1)}/{args.option}'

        if len(args.ml_method) > 0:
            file_path += f'_{args.ml_method}'

        # if args.option == 'fed_sv_cs_super':
        #     file_path += f'_{args.num_clients_remain}'

        if args.cs_reverse:
            file_path += '_reverse'

        if args.num_clients != 100:
            file_path += f'_{args.num_clients}'
        elif args.prune_frac == 0.7 and args.num_clients_remain != 20:
            file_path += f'_{args.num_clients_remain}'

        file_path += f'.txt'
        print(f'written to {file_path}')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @classmethod
    def to_obj(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(
            round_accuracy=data["round_accuracy"],
            remain_frac=data["remain_frac"],
            clients_remain_farc={int(k): v for k, v in data["clients_remain_farc"].items()}
        )


def show_prune_information(client_loaders, remained_clients=None, show_each_client=False):
    if remained_clients is None:
        remained_clients = np.arange(len(client_loaders))

    total_original = len(client_loaders[0].dataset.data)
    total_remain = 0
    clients_remain_frac = {int(client): 0.0 for client in remained_clients}
    for client in remained_clients:
        client_dataset = client_loaders[client].dataset
        cur_len = client_dataset.get_current_length()
        total_remain += cur_len
        ori_len = client_dataset.get_original_length()
        clients_remain_frac[client] = round(1 - cur_len / ori_len, 4)
        if show_each_client:
            print(f'client {client} prune {clients_remain_frac[client]:.4g}')

    total_remain_frac = total_remain / total_original

    print(f'method {args.option} '
          f'prune {args.prune_frac} '
          f'dirichlet {args.dirichlet_alpha} '
          f'data remain {total_remain_frac:.4g} '
          f'clients remain {len(remained_clients)} '
          f'diff {1 - args.prune_frac - total_remain_frac:.4g} '
          f'{args.ml_method if len(args.ml_method) > 0 else ""}'
          )

    return total_remain_frac, clients_remain_frac


if __name__ == '__main__':
    remained_clients = None
    if args.option == "fed_avg":
        round_accuracy = fed_avg(global_model, client_loaders, test_loader)
    elif args.option == 'random_data':
        round_accuracy = fed_random_data_cs(global_model, client_loaders, test_loader)
    elif args.option == "fed_cs":
        round_accuracy = fed_cs(global_model, client_loaders, test_loader)
    elif args.option == "fed_gc":
        round_accuracy = fed_gc(global_model, client_loaders, test_loader, validate_loader)
    elif args.option == "fed_craig":
        round_accuracy = fed_craig(global_model, client_loaders, test_loader)
    elif args.option == "fed_el2n":
        round_accuracy = fed_el2n(global_model, client_loaders, test_loader)
    elif args.option == "fed_ccs":
        round_accuracy = fed_ccs(global_model, client_loaders, test_loader)
    elif args.option == "fed_moderate":
        round_accuracy = fed_moderate(global_model, client_loaders, test_loader)

    elif args.option == 'random_client':
        round_accuracy, remained_clients = fed_random_client_cs(global_model, client_loaders, test_loader)
    elif args.option == 'random_data_execute':
        round_accuracy, remained_clients = fed_random_data_cs_execute(global_model, client_loaders, test_loader)
    elif args.option == "fed_gc_execute":
        round_accuracy, remained_clients = fed_gc_execute(global_model, client_loaders, test_loader, validate_loader)
    elif args.option == "fed_el2n_execute":
        round_accuracy, remained_clients = fed_el2n_execute(global_model, client_loaders, test_loader)
    elif args.option == "fed_sv_cs":
        round_accuracy, remained_clients = shap_ccs_max(global_model, client_loaders, test_loader)
    elif args.option == "fed_sv_cs_super":
        round_accuracy, remained_clients = shap_ccs(global_model, client_loaders, test_loader, validate_loader)
    elif args.option in ["fed_cssv_cs", "fed_gtgsv_cs", "fed_nsv_cs", "fed_cgsv_cs", "fed_cgsv_sample_cs"]:
        round_accuracy, remained_clients = fed_other_sv_cs(global_model, client_loaders, test_loader)
    else:
        raise 'method not correct'

    total_remain_frac, clients_remain_frac = show_prune_information(client_loaders, remained_clients,
                                                                    show_each_client=True)

    res = PruneResult(round_accuracy, total_remain_frac, clients_remain_frac)
    res.to_json()

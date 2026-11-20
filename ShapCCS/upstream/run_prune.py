from itertools import product

from plyer import notification

from scripts.running import run_all_tasks

if __name__ == "__main__":

    seeds = [
        0,
        # 1,
        # 10,
        # 100,
        # 1000
    ]

    methods = [
        # 'fed_avg',
        # 'random_data',
        # 'fed_craig',
        # 'fed_cs',
        # 'fed_el2n',
        # 'fed_ccs',
        # 'fed_moderate',
        # 'fed_gc',
        # 'fed_sv_cs', # ShapCCS*
        'fed_sv_cs_super', # ShapCCS

        # 'random_client',

        # 'fed_cssv_cs',
        # 'fed_gtgsv_cs',
        # 'fed_cgsv_cs',
        # 'fed_cgsv_sample_cs',
        # 'fed_nsv_cs',
    ]

    # data level methods that can be combined with ShapCCS
    ml_methods = [
        # 'random',
        # 'fed_cs',
        # 'el2n',
        # 'craig',
        # 'gcfl',
        # 'ccs',
        'moderate'
    ]

    datasets = [
        # 'cifar10',
        # 'cifar100'
        # 'cinic10',
        'fmnist',
        # 'mnist',
    ]

    dirichlet_alphas = [
        0.1,
        # float('inf'),
        # 0.5,
    ]

    prune_fracs = [
        # 0.3,
        # 0.5,
        0.7,
    ]

    noisy_configs = {
        0.0: [0.0],
        # 1.0: [i / 10 for i in range(1, 7)],
        # 0.3: [i / 10 for i in range(1, 7)],
    }

    num_clients_remain_config = {
        0.7: [20],
        # 0.7: [5, 10, 40, 80, 160, 320, 640, 1280, 2560],
        0.5: [50],
        0.3: [70]
    }

    # num_clients_configs = [i * 10 for i in range(2, 21, 2) if i != 10]
    num_clients_configs = [100]

    tasks = []

    base_combinations = product(
        prune_fracs,
        methods,
        dirichlet_alphas,
        datasets,
        noisy_configs.items(),
        num_clients_configs
    )
    for (prune_frac, method, dirichlet_alpha, dataset,
         (noisy_client_ratio, noise_ratios), num_clients) in base_combinations:
        for noise_ratio in noise_ratios:
            for num_clients_remain in num_clients_remain_config[prune_frac]:
                cmd = [
                    "python", "scripts/prune.py",
                    "--option", method,
                    "--dataset", dataset,
                    "--dirichlet_alpha", str(dirichlet_alpha),
                    "--prune_frac", str(prune_frac),
                    "--noisy_client_ratio", str(noisy_client_ratio),
                    "--noise_ratio", str(noise_ratio),
                    "--num_clients_remain", str(num_clients_remain),
                    "--num_clients", str(num_clients)
                ]

                if method == 'fed_sv_cs_super':
                    for ml_method in ml_methods:
                        tasks.append((cmd + ["--ml_method", ml_method], True))
                else:
                    tasks.append((cmd, True))

    run_all_tasks(tasks, num_workers=8)

    notification.notify(
        title="111",
        message="111",
        timeout=5,
    )

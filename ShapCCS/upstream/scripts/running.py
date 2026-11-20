import multiprocessing as mp
import os
import subprocess

from tqdm import tqdm


def run_cmd(cmd, print_cmd=True):
    if print_cmd:
        print("Running:", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        env=env
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def run_cmd_serial(cmd, print_cmd=True):
    if print_cmd:
        print("Running:", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = '0'

    subprocess.run(cmd, check=True, env=env)


def run_single_task(args):
    cmd, print_cmd = args
    run_cmd(cmd, print_cmd=print_cmd)


def run_all_tasks(tasks, num_workers=3):
    with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        try:
            list(tqdm(
                pool.imap(run_single_task, tasks),
                total=len(tasks),
                desc="Processing",
                unit="task"
            ))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            exit(1)

    print("All tasks completed.")


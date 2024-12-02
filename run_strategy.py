"""
Strategy can viewed as a full pipeline/meta system that can go through multiple training/eval stages
and able to hold multiple systems together.
"""
import os
import argparse
import yaml

from src.utils.tool import seed_everything
from src.strategies.load import get_strategy_cls
from src.tasks.load import get_task


def get_data_obj(task_name: str):
    if task_name == "none":
        return None
    elif task_name == "cv-val100":
        return get_task("cv-val100")
    else:
        raise NotImplementedError


def create_config(args):
    """ Create a dictionary for full configuration """
    res = {
        "strategy_name": args.strategy_name,
        "task_name": args.task_name,
    }
    res["strategy_config"] = {}
    for path in args.config:
        strategy_config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        res["strategy_config"].update(strategy_config)
    
    exp_name = args.exp_name
    exp_root = f"results/{exp_name}"  # maximum flexibility
    os.makedirs(exp_root, exist_ok=True)
    os.makedirs(f"{exp_root}/log", exist_ok=True)
    os.makedirs(f"{exp_root}/result", exist_ok=True)
    os.makedirs(f"{exp_root}/ckpt", exist_ok=True)
    res["output_dir"] = {
        "exp_root": exp_root,
        "log_dir": f"{exp_root}/log",
        "result_dir": f"{exp_root}/result",
        "ckpt_dir": f"{exp_root}/ckpt"
    }
    with open(f"{exp_root}/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(res, f, sort_keys=False)

    return res


def main(args):
    config = create_config(args)
    strategy = get_strategy_cls(args.strategy_name)(config)
    data_obj = get_data_obj(args.task_name)

    print("========================== Start! ==========================")
    print("Output Dir: ", config["output_dir"]["exp_root"])
    print("Strategy name: ", config["strategy_name"])
    print("Task name: ", config["task_name"])
    
    strategy.run(data_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR")
    parser.add_argument('-s', '--strategy_name', type=str, help="strategy identifier")
    parser.add_argument('-t', '--task_name', type=str, help="task identifier")  # note that this is a meta data object
    parser.add_argument('-n', '--exp_name', type=str, default="unnamed_strategy")
    parser.add_argument('--config', nargs='+', default=[])

    args = parser.parse_args()
    seed_everything(666)
    main(args)

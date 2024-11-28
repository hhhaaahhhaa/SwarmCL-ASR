"""
Strategy can viewed as a full pipeline/meta system that can go through multiple training/eval stages
and able to hold multiple systems together.
"""
import os
import argparse
import yaml

from src.utils.tool import seed_everything
from src.strategies.load import get_strategy_cls


def create_config(args):
    """ Create a dictionary for full configuration """
    res = {
        "strategy_name": args.strategy_name,
        "task_name": args.task_name,
    }
    res["strategy_config"] = {}
    for path in args.strategy_config:
        strategy_config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        res["strategy_config"].update(strategy_config)

    return res


def main(args):
    config = create_config(args)
    if args.debug:
        print("Configuration: ", config)
    
    exp_name = args.exp_name
    exp_root = f"results/benchmark/{args.strategy_name}/{args.exp_name}/{args.task_name}"
    os.makedirs(exp_root, exist_ok=True)
    config["output_dir"] = {
        "log_dir": f"{exp_root}/log",
        "result_dir": f"{exp_root}/result",
        "ckpt_dir": f"{exp_root}/ckpt"
    }
    with open(f"{exp_root}/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    strategy = get_strategy_cls(args.strategy_name)(config)
    # TODO: create specific data object
    data_obj = get_data_obj(args.task_name)

    print("========================== Start! ==========================")
    print("Exp name: ", exp_name)
    print("Strategy name: ", config["strategy_name"])
    print("Task name: ", config["task_name"])
    print("Exp directory: ", config["output_dir"])
    
    strategy.run(data_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR")
    parser.add_argument('-s', '--system_name', type=str, help="system identifier")
    parser.add_argument('-t', '--task_name', type=str, help="task identifier")  # note that this is a meta data object
    parser.add_argument('-n', '--exp_name', type=str, default="unnamed")
    parser.add_argument('--config', nargs='+', default=["config/strategy/swarm-cl.yaml"])
    parser.add_argument(
        "--logger", type=str,
        default="none",
    )
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    seed_everything(666)
    main(args)

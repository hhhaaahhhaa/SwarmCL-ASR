import os
import torch
import argparse
import yaml
import lightning as pl
from lightning.pytorch.profilers import SimpleProfiler

from src.systems.load import get_system_cls
from src.tasks.load import get_task
from src.datamodule import DataModule


def create_config(args):
    """ Create a dictionary for full configuration """
    res = {
        "system_name": args.system_name,
        "task_name": args.task_name,
    }
    if args.checkpoint is not None:
        res["checkpoint"] = args.checkpoint
    res["config"] = {}
    for path in args.config:
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        res["config"].update(config)

    exp_name = args.exp_name
    exp_root = f"results/{exp_name}"  # maximum flexibility
    os.makedirs(exp_root, exist_ok=True)
    res["config"]["output_dir"] = {
        "exp_root": exp_root,
        "log_dir": f"{exp_root}/log",
        "result_dir": f"{exp_root}/result",
        "ckpt_dir": f"{exp_root}/ckpt"
    }
    with open(f"{exp_root}/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(res, f, sort_keys=False)

    return res


def train_one_task(config: dict, debug: bool=False):
    # Init system
    system_config = config["config"]
    system_cls = get_system_cls(config["system_name"])
    if config.get("checkpoint", None) is not None:
        try:
            print(f'Load from {config["checkpoint"]}...')
            system = system_cls.load_from_checkpoint(config["checkpoint"], config=system_config)  # ONLY load weights and overwrite config
        except:
            print(f'System {config["system_name"]} fails/unsupports checkpoint loading.')
            exit()
    else:
        system = system_cls(system_config)
    if debug:
        print("System module prepared.")
        input()

    # Init task
    task = get_task(config["task_name"])
    datamodule = DataModule(
        train_dataset=task.train_dataset(),
        val_dataset=task.val_dataset(),
        test_dataset=task.test_dataset(),
        batch_size=system_config["train_config"]["per_device_train_batch_size"]
    )
    if debug:
        print("Data module prepared.")
        input()

    # TODO: Init logger
    loggers = None
    
    # Training
    train_config = system_config["train_config"]
    trainer_training_config = {
        'default_root_dir': system_config["output_dir"]["exp_root"],
        'max_epochs': train_config["num_train_epochs"],
        'log_every_n_steps': train_config["logging_steps"],
    }
    if system.automatic_optimization:
        trainer_training_config.update({
            'gradient_clip_val': train_config.get("grad_clip_thresh", 1.0),
            'accumulate_grad_batches': train_config.get("gradient_accumulation_steps", 1),
        })

    print("========================== Start Training! ==========================")
    print("Output Dir: ", system_config["output_dir"]["exp_root"])
    print("System name: ", config["system_name"])
    print("Task name: ", config["task_name"])
    print("Checkpoint Path: ", config.get("checkpoint", "none"))
    
    pl.seed_everything(42, True)
    if debug:  # Useful for debugging
        trainer_training_config.update({
            "limit_train_batches": 10,
            "limit_val_batches": 5,
            "max_epochs": 3,
            'log_every_n_steps': 1,
        })
    trainer = pl.Trainer(
        **trainer_training_config,
        accelerator="gpu" if torch.cuda.is_available() else None,
        strategy="ddp_find_unused_parameters_true",  # multigpu should use ddp
        logger=loggers,
        profiler=SimpleProfiler(system_config["output_dir"]["exp_root"], filename="profile"),
    )
    trainer.fit(system, datamodule=datamodule)


def main(args):
    config = create_config(args)
    if args.debug:
        print("Configuration: ", config)
    train_one_task(config, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR")
    parser.add_argument('-s', '--system_name', type=str, help="system identifier")
    parser.add_argument('-t', '--task_name', type=str, help="task identifier")
    parser.add_argument('-n', '--exp_name', type=str, default="unnamed")
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('--config', nargs='+', default=["config/system/base.yaml"])
    parser.add_argument(
        "--logger", type=str,
        default="none",
    )
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    main(args)

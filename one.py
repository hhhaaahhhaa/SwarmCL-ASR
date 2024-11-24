import os
import torch
import argparse
import yaml
import lightning as pl

from src.systems.load import get_system_cls
from src.tasks.load import get_task
from src.datamodule import DataModule


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # https://stackoverflow.com/questions/73747731/runtimeerror-cuda-out-of-memory-how-can-i-set-max-split-size-mb

TRAINER_CONFIG = {
    "accelerator": "gpu" if torch.cuda.is_available() else None,
    "strategy": "ddp_find_unused_parameters_true",  # multigpu should use ddp
    "profiler": 'simple',
}


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
    res["output_dir"] = {
        "log_dir": f"{exp_root}/log",
        "result_dir": f"{exp_root}/result",
        "ckpt_dir": f"{exp_root}/ckpt"
    }
    with open(f"{exp_root}/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(res, f, sort_keys=False)

    return res


def train_one_task(config: dict, debug: bool=False):
    # Init system
    system_cls = get_task(config["system_name"])(config)

    if config.get("checkpoint", None) is not None:
        try:
            print(f'Load from {config["checkpoint"]}...')
            system = system_cls.load_from_checkpoint(config["checkpoint"], config=config)  # ONLY load weights
        except:
            print(f'System {config["system_name"]} fails/unsupports checkpoint loading.')
            exit()

    if debug:
        print("System module prepared.")
        input()

    # Init task
    task = get_task(config["task_name"])
    datamodule = DataModule(
        train_dataset=task.train_dataset(),
        val_dataset=task.val_dataset(),
        test_dataset=task.test_dataset(),
        batch_size=config["train_config"]["batch_size"]
    )
    if debug:
        print("Data module prepared.")
        input()

    # TODO: Init logger
    loggers = None
    
    # Training
    train_config = config["train_config"]
    trainer_training_config = {
        'max_steps': train_config["step"]["total_step"],
        'log_every_n_steps': train_config["step"]["log_step"],
        'val_check_interval': train_config["step"]["val_step"],
        'check_val_every_n_epoch': None,
    }
    if system.automatic_optimization:
        trainer_training_config.update({
            'gradient_clip_val': train_config["optimizer"]["grad_clip_thresh"],
            'accumulate_grad_batches': train_config["optimizer"]["grad_acc_step"],
        })

    print("========================== Start Training! ==========================")
    pl.seed_everything(43, True)
    TRAINER_CONFIG = {
        "accelerator": "gpu" if torch.cuda.is_available() else None,
        "strategy": "ddp_find_unused_parameters_true",  # multigpu should use ddp
        "profiler": 'simple',
    }
    if debug:  # Useful for debugging
        TRAINER_CONFIG.update({
            "limit_train_batches": 300,
            "limit_val_batches": 50,
            "max_epochs": 3
        })
    trainer = pl.Trainer(logger=loggers, **TRAINER_CONFIG, **trainer_training_config)
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
    parser.add_argument('--config', nargs='+', default=["config/system/suta.yaml"])
    parser.add_argument(
        "--logger", type=str, help="output result path",
        default="tb",
    )
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    main(args)

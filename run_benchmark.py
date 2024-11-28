import os
import argparse
import yaml
import pickle
from tqdm import tqdm

from src.systems.load import get_system_cls
from src.tasks.load import get_task
from src.utils.tool import wer


def run_eval(system, output_dir: str):
    accents = ["aus", "eng", "ind", "ire", "sco"]
    for accent in (accents):
        tname = f"cv-{accent}"
        ds = get_task(tname).test_dataset()
        system.eval()

        long_cnt = 0
        basenames = []
        n_words = []
        errs = []
        transcriptions = []
        for sample in tqdm(ds, desc=accent):
            if len(sample["wav"]) > 320000:  # 20s
                long_cnt += 1
                continue
            n_words.append(len(sample["text"].split(" ")))
            trans = system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))
            basenames.append(sample["id"])

        results = {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "basenames": basenames,
        }
        log_results(results, f"{output_dir}/{accent}")


def calc_wer(results: dict) -> float:
    assert len(results["n_words"]) == len(results["wers"])  # make sure correctness
    err = 0
    for i in range(len(results["n_words"])):
        err += results["wers"][i] * results["n_words"][i]
    denom = sum(results["n_words"])
    return err / denom


def log_results(results, output_dir: str):
    word_error_rate = calc_wer(results)
    print(f"WER: {word_error_rate * 100:.2f}%")

    # log
    log_dir = f"{output_dir}/log"
    result_dir = f"{output_dir}/result"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    with open(f'{output_dir}/results.txt', "w") as f:
        f.write(f"WER: {word_error_rate * 100:.2f}%\n")
    with open(f'{result_dir}/results.pkl', "wb") as f:
        pickle.dump(results, f)
    with open(f'{log_dir}/transcriptions.txt', "w") as f:
        for (orig, pred), wer, basename in zip(results["transcriptions"], results["wers"], results["basenames"]):
            f.write(f"{wer * 100:.2f}%|{basename}|{orig}|{pred}\n")


def main(args):
    output_dir = f"results/benchmark/{args.output_dir}"
    ckpt_path = "none"
    if args.exp_dir is None:  # load an empty system (usually a pretrained checkpoint)
        assert args.config is not None
        system_config = {}
        for path in args.config:
            config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
            system_config.update(config)
        exp_root = output_dir
        system_config["output_dir"] = {
            "exp_root": exp_root,
            "log_dir": f"{exp_root}/log",
            "result_dir": f"{exp_root}/result",
            "ckpt_dir": f"{exp_root}/ckpt"
        }
        system_cls = get_system_cls(args.system_name)
        system = system_cls(system_config)
    else:
        ckpt_path = f"{args.exp_dir}/{args.checkpoint}"
        config = yaml.load(open(f"{args.exp_dir}/config.yaml", "r"), Loader=yaml.FullLoader)
        system_config = config["config"]
        system_cls = get_system_cls(config["system_name"])
        system = system_cls.load_from_checkpoint(ckpt_path, config=system_config)

    print("========================== Start! ==========================")
    print("Output Dir: ", output_dir)
    print("System name: ", args.system_name)
    print("Checkpoint Path: ", ckpt_path)

    run_eval(system, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR")
    parser.add_argument('-o', '--output_dir', type=str, help="path for evaluated results")
    parser.add_argument('-s', '--system_name', type=str, help="system identifier")
    parser.add_argument('-n', '--exp_dir', type=str, default=None)
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('--config', nargs='+', default=["config/system/base.yaml"])

    args = parser.parse_args()
    main(args)

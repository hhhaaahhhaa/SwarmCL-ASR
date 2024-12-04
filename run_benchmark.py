import os
import argparse
import yaml
import pickle
from tqdm import tqdm

from src.tasks.load import get_task
from src.utils.tool import wer
from one import load_system


def run_eval(system, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    system.eval()
    system.cuda()
    accents = ["aus", "eng", "ind", "ire", "sco"]
    for accent in (accents):
        tname = f"cv-{accent}"
        ds = get_task(tname).test_dataset()

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
    if args.loader == "torch":
        system_config = {}
        for path in args.config:
            config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
            system_config.update(config)
    else:
        system_config = None
    system = load_system(
        system_name=args.system_name,
        system_config=system_config,
        checkpoint=args.checkpoint,
        loader=args.loader
    )

    print("========================== Start! ==========================")
    print("Output Dir: ", output_dir)
    print("System name: ", args.system_name)
    print("Checkpoint Path: ", args.checkpoint)

    run_eval(system, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR")
    parser.add_argument('-o', '--output_dir', type=str, help="path for evaluated results")
    parser.add_argument('-s', '--system_name', type=str, help="system identifier")
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('--config', nargs='+', default=["config/system/base.yaml"])
    parser.add_argument("--loader", type=str, default="torch")

    args = parser.parse_args()
    main(args)

import numpy as np
import os
import librosa
from datasets import load_dataset
from tqdm import tqdm
from scipy.io import wavfile
import json


def commonvoice_preprocess():
    cache_dir = "_cache/CommonVoice"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(f"{cache_dir}/wav", exist_ok=True)
    os.makedirs(f"{cache_dir}/text", exist_ok=True)

    src_dataset = load_dataset(
        "mozilla-foundation/common_voice_16_1", "en",
        split="test",
        trust_remote_code=True,
    )

    data_info = []
    for idx, instance in tqdm(enumerate(src_dataset)):
        audio = instance.pop("audio")
        wav = librosa.resample(
            audio["array"],
            orig_sr=src_dataset.features["audio"].sampling_rate,
            target_sr=16000
        )
        
        basename = f"{idx:07d}"
        wavfile.write(f"{cache_dir}/wav/{basename}.wav", 16000, (wav * 32767).astype(np.int16))
        with open(f"{cache_dir}/text/{basename}.txt", "w", encoding="utf-8") as f:
            f.write(instance["sentence"])
        data_info.append({
            "basename": basename,
            "length": len(wav),
            "text": instance["sentence"],
            "metadata": instance
        })
    with open(f"{cache_dir}/data_info.json", "w", encoding="utf-8") as f:
        json.dump(data_info, f, indent=4)


def commonvoice_accent():
    src_dataset = load_dataset(
        "mozilla-foundation/common_voice_16_1", "en",
        trust_remote_code=True,
    )
    
    filename2accent = {}
    accents = []
    root = "preprocess/cv-accent-split/test"
    for accent in os.listdir(root):
        accents.append(accent)
        cache_dir = f"_cache/CommonVoice-accent/{accent}"
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(f"{cache_dir}/wav", exist_ok=True)
        os.makedirs(f"{cache_dir}/text", exist_ok=True)
        data_info = []
        with open(f"{root}/{accent}/list_of_utterances.txt", encoding="utf-8") as f:
            for line in f:
                if line == "\n":
                    continue
                # select file from hf dataset
                [client_id, filename] = line.strip().split('-')
                filename2accent[filename] = accent
                # print(filename, accent)
    
    idxs = {x: 0 for x in accents}
    data_infos = {x: [] for x in accents}
    subsets = ["train", "validation", "test"]
    for split in subsets:
        ds = src_dataset[split]
        for instance in tqdm(ds):
            filename = instance["path"].split('/')[-1][:-4]
            accent = filename2accent.get(filename, None)
            if accent is None:
                continue

            cache_dir = f"_cache/CommonVoice-accent/{accent}"
            audio = instance.pop("audio")
            wav = librosa.resample(
                audio["array"],
                orig_sr=ds.features["audio"].sampling_rate,
                target_sr=16000
            )
            
            basename = f"{idxs[accent]:07d}"
            wavfile.write(f"{cache_dir}/wav/{basename}.wav", 16000, (wav * 32767).astype(np.int16))
            with open(f"{cache_dir}/text/{basename}.txt", "w", encoding="utf-8") as f:
                f.write(instance["sentence"])
            data_infos[accent].append({
                "basename": basename,
                "length": len(wav),
                "text": instance["sentence"],
                "metadata": instance
            })
            idxs[accent] += 1
    
    # metadata
    for accent, data_info in data_infos.items():
        cache_dir = f"_cache/CommonVoice-accent/{accent}"
        with open(f"{cache_dir}/data_info.json", "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)


def commonvoice_accent_full():
    src_dataset = load_dataset(
        "mozilla-foundation/common_voice_16_1", "en",
        trust_remote_code=True,
    )
    
    filename2accent = {}
    filename2split = {}
    accents = []
    for sp in ["train", "dev", "test"]:
        root = f"preprocess/cv-accent-split/{sp}"
        for accent in os.listdir(root):
            if sp == "train":
                accents.append(accent)
            cache_dir = f"_cache/CommonVoice-accent-full/{accent}/{sp}"
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(f"{cache_dir}/wav", exist_ok=True)
            os.makedirs(f"{cache_dir}/text", exist_ok=True)
            data_info = []
            with open(f"{root}/{accent}/list_of_utterances.txt", encoding="utf-8") as f:
                for line in f:
                    if line == "\n":
                        continue
                    # select file from hf dataset
                    [client_id, filename] = line.strip().split('-')
                    filename2accent[filename] = accent
                    filename2split[filename] = sp
                    # print(filename, accent)
    
    idxs = {x: 0 for x in accents}
    train_total_duration = {x: 0 for x in accents}
    data_infos = {x: {"train": [], "dev": [], "test": []} for x in accents}
    subsets = ["validation", "test", "train"]
    for split in subsets:
        ds = src_dataset[split]
        for instance in tqdm(ds, desc=f"{split}"):
            filename = instance["path"].split('/')[-1][:-4]
            accent = filename2accent.get(filename, None)
            sp = filename2split.get(filename, None)
            if accent is None:
                continue
            
            # smaller training set
            if sp == "train" and train_total_duration[accent] > 16000 * 3600 * 10:  # maximum 10 hours
                continue

            cache_dir = f"_cache/CommonVoice-accent-full/{accent}/{sp}"
            audio = instance.pop("audio")
            wav = librosa.resample(
                audio["array"],
                orig_sr=ds.features["audio"].sampling_rate,
                target_sr=16000
            )

            # smaller training set
            if sp == "train":
                train_total_duration[accent] += len(wav)
            
            basename = f"{idxs[accent]:07d}"
            wavfile.write(f"{cache_dir}/wav/{basename}.wav", 16000, (wav * 32767).astype(np.int16))
            with open(f"{cache_dir}/text/{basename}.txt", "w", encoding="utf-8") as f:
                f.write(instance["sentence"])
            data_infos[accent][sp].append({
                "basename": basename,
                "length": len(wav),
                "text": instance["sentence"],
                "metadata": instance
            })
            idxs[accent] += 1
    
    # metadata
    for accent in data_infos:
        for sp in data_infos[accent]:
            cache_dir = f"_cache/CommonVoice-accent-full/{accent}/{sp}"
            data_info = data_infos[accent][sp]
            with open(f"{cache_dir}/data_info.json", "w", encoding="utf-8") as f:
                json.dump(data_info, f, indent=4)


if __name__ == "__main__":
    # commonvoice_preprocess()
    # commonvoice_accent()
    commonvoice_accent_full()

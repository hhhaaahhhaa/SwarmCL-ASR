import os
from tqdm import tqdm
import json
import torchaudio


ROOT = "/livingrooms/public/LibriSpeech"  # set your local path


def filter(cache_dir, dset):
    with open(f"{cache_dir}/{dset}.json", "r") as f:
        metadata = json.load(f)
    data_info = []
    for id, instance in tqdm(metadata.items()):
        os.makedirs(f"{cache_dir}/{dset}/wav", exist_ok=True)
        os.makedirs(f"{cache_dir}/{dset}/text", exist_ok=True)

        wav_path = instance.pop("wav_path")
        tgt_wav_path = f"{cache_dir}/{dset}/wav/{id}.wav"
        wav, sample_rate = torchaudio.load(wav_path)
        assert wav.shape[0] == 1
        assert sample_rate == 16000
        duration = wav.shape[1] / sample_rate
        if not (1 <= duration < 30):  # too long/too short
            continue
        torchaudio.save(tgt_wav_path, wav, sample_rate)

        text = instance.pop("text")
        with open(f"{cache_dir}/{dset}/text/{id}.txt", "w") as f:
            f.write(text)
        
        data_info.append({
            "basename": id,
            "length": wav.shape[1],
            "text": text,
            "metadata": instance
        })

    with open(f"{cache_dir}/{dset}/data_info.json", "w") as f:
        json.dump(data_info, f, indent=4)


def main(root: str, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    dsets = ["train-clean-360", "dev-clean", "test-clean"]
    data = {}
    for dset in dsets:
        data[dset] = {}
        for speaker in tqdm(os.listdir(f"{root}/{dset}"), desc=dset):
            for chapter in os.listdir(f"{root}/{dset}/{speaker}"):
                with open(f"{root}/{dset}/{speaker}/{chapter}/{speaker}-{chapter}.trans.txt", "r") as f:
                    for line in f:
                        if line == "\n":
                            continue
                        [id, text]  = line.strip().split(" ", 1)
                        wav_path = f"{root}/{dset}/{speaker}/{chapter}/{id}.flac"
                        data[dset][id] = {
                            "id": id,
                            "speaker": speaker,
                            "chapter": chapter,
                            "wav_path": wav_path,
                            "text": text,
                        }
        with open(f"{cache_dir}/{dset}.json", "w") as f:
            json.dump(data[dset], f, indent=4)
    
    for dset in dsets:
        filter(cache_dir, dset)


if __name__ == "__main__":
    main(ROOT, cache_dir="./_cache/LibriSpeech")

from torch.utils.data import Dataset, ConcatDataset, Subset
import hashlib
import random

from .base import Task, StandardDataset
from .apply_noise import add_noise


class LibriSpeechTask(Task):
    def __init__(self) -> None:
        self._train_dataset = StandardDataset(root=f"_cache/LibriSpeech/train-clean-360")
        self._val_dataset = StandardDataset(root=f"_cache/LibriSpeech/dev-clean")
        self._test_dataset = StandardDataset(root=f"_cache/LibriSpeech/test-clean")

    def train_dataset(self) -> Dataset:
        return self._train_dataset

    def val_dataset(self) -> Dataset:
        return self._val_dataset
    
    def test_dataset(self) -> Dataset:
        return self._test_dataset


def get_seed_from_string_and_int(input_string, input_int):
    # Combine the string and integer into a single string
    combined = f"{input_string}:{input_int}"
    
    # Use a hash function to generate a consistent hash value
    hash_object = hashlib.sha256(combined.encode())
    
    # Convert the hash to an integer
    seed = int(hash_object.hexdigest(), 16) % (2**32)  # Limit seed to 32 bits
    return seed


class NoiseWrapper(Dataset):
    def __init__(self, ds: Dataset, noise_type: str, snr_level=10, name: str=None):
        self.obj = ds
        self.noise_type = noise_type
        self.snr_level = snr_level
        self.name = name
    
    def __getitem__(self, index):
        res = self.obj.__getitem__(index)
        res["wav"] = add_noise(res["wav"], noise_type=self.noise_type, snr_level=self.snr_level)
        return res
    
    def __len__(self):
        return len(self.obj)


class LibriSpeechNoiseTask(Task):
    def __init__(self, noise_type: str, snr_level=10, n_samples=5000) -> None:
        self.name = f"LS_{noise_type}_{snr_level}"
        self.noise_type = noise_type
        self.snr_level = snr_level
        self.n_samples = n_samples
        self.seed = get_seed_from_string_and_int(noise_type, snr_level)

    def train_dataset(self) -> Dataset:
        res = getattr(self, "_train_dataset", None)
        if res is None:
            ds = StandardDataset(root=f"_cache/LibriSpeech/train-clean-360")
            indices = random.sample(range(len(ds)), self.n_samples)
            self._train_dataset = NoiseWrapper(
                Subset(ds, indices=indices),
                noise_type=self.noise_type,
                snr_level=self.snr_level
            )
        return self._train_dataset

    def val_dataset(self) -> Dataset:
        res = getattr(self, "_val_dataset", None)
        if res is None:
            ds = StandardDataset(root=f"_cache/LibriSpeech/dev-clean")
            self._val_dataset = NoiseWrapper(
                Subset(ds, indices=list(range(1000))),
                noise_type=self.noise_type,
                snr_level=self.snr_level
            )
        return self._val_dataset
    
    def test_dataset(self) -> Dataset:
        res = getattr(self, "_test_dataset", None)
        if res is None:
            ds = StandardDataset(root=f"_cache/LibriSpeech/test-clean")
            self._test_dataset = NoiseWrapper(
                Subset(ds, indices=list(range(1000))),
                noise_type=self.noise_type,
                snr_level=self.snr_level
            )
        return self._test_dataset

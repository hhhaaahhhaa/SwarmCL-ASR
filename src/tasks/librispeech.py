from torch.utils.data import Dataset, ConcatDataset, Subset
import hashlib
import random

from .base import Task, StandardDataset
from .apply_noise import apply_noise_on_dataset


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


class LibriSpeechNoiseTask(Task):
    def __init__(self, noise_type: str, snr_level=10, n_samples=5000) -> None:
        self.full = LibriSpeechTask()
        seed = get_seed_from_string_and_int(noise_type, snr_level)
        random.seed(seed)
        indices = random.sample(range(len(self.full._train_dataset)), n_samples)
        self._train_dataset = Subset(self.full._train_dataset, indices=indices)
        self._val_dataset = Subset(self.full._val_dataset, indices=list(range(1000)))
        self._test_dataset = Subset(self.full._test_dataset, indices=list(range(1000)))
        apply_noise_on_dataset(self._train_dataset, noise_type, snr_level)
        apply_noise_on_dataset(self._val_dataset, noise_type, snr_level)
        apply_noise_on_dataset(self._test_dataset, noise_type, snr_level)

    def train_dataset(self) -> Dataset:
        return self._train_dataset

    def val_dataset(self) -> Dataset:
        return self._val_dataset
    
    def test_dataset(self) -> Dataset:
        return self._test_dataset

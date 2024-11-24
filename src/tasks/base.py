from typing import Optional
from torch.utils.data import Dataset

from src.corpus.corpus import StandardCorpus


class Task(object):
    """ Task template """
    def train_dataset(self) -> Optional[Dataset]:
        return None

    def val_dataset(self) -> Optional[Dataset]:
        return None
    
    def test_dataset(self) -> Optional[Dataset]:
        return None
    

class StandardDataset(Dataset):
    def __init__(self, root: str) -> None:
        self.corpus = StandardCorpus(root=root)
        self.idx_seq = list(range(len(self.corpus)))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])

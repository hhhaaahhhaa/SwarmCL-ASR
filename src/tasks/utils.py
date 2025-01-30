from torch.utils.data import Dataset

from .base import Task
from .load import get_task


class Sequence(Dataset):
    """ Concat datasets and addtionally return an identifier. """
    def __init__(
        self,
        datasets: list[Dataset],
        names: list[str],
        tid_seq=None,
        tidx_seq=None,
        task_boundaries=None
    ) -> None:
        self.datasets = datasets
        self.names = names
        self.tid_seq = tid_seq
        self.tidx_seq = tidx_seq
        self.task_boundaries = task_boundaries

        if self.tid_seq is None:  # direct concat
            self.tid_seq, self.tidx_seq, self.task_boundaries = [], [], []
            for i, ds in enumerate(self.datasets):
                self.tid_seq.extend([i] * len(ds))
                self.tidx_seq.extend(list(range(len(ds))))
                self.task_boundaries.append(len(self.tid_seq))

    def __len__(self):
        return len(self.tidx_seq)
    
    def __getitem__(self, idx):
        tid = self.tid_seq[idx]
        tidx = self.tidx_seq[idx]

        return {
            "tid": self.names[tid],
            **self.datasets[tid].__getitem__(tidx)
        }


class TaskSequence(Task):

    task_names: list[str]
    tasks: list[Task]

    def __init__(
        self,
        tnames: list[str],
    ) -> None:
        self.tasks = []
        self.task_names = tnames
        for tname in tnames:
            self.tasks.append(get_task(tname))
    
    def train_dataset(self) -> Dataset:
        res = getattr(self, "_train_dataset", None)
        if res is None:
            datasets = [ts.train_dataset() for ts in self.tasks]
            self._train_dataset = Sequence(datasets, names=self.task_names)
        return self._train_dataset
    
    def val_dataset(self) -> Dataset:
        res = getattr(self, "_val_dataset", None)
        if res is None:
            datasets = [ts.val_dataset() for ts in self.tasks]
            self._val_dataset = Sequence(datasets, names=self.task_names)
        return self._val_dataset
    
    def test_dataset(self) -> Dataset:
        res = getattr(self, "_test_dataset", None)
        if res is None:
            datasets = [ts.test_dataset() for ts in self.tasks]
            self._test_dataset = Sequence(datasets, names=self.task_names)
        return self._test_dataset


class MultiTaskSequence(Dataset):
    def __init__(self, datasets: list[Dataset], tid_seq=None, tidx_seq=None, task_boundaries=None) -> None:
        self.datasets = datasets
        self.tid_seq = tid_seq
        self.tidx_seq = tidx_seq
        self.task_boundaries = task_boundaries

        if self.tid_seq is None:  # direct concat
            self.tid_seq, self.tidx_seq, self.task_boundaries = [], [], []
            for i, ds in enumerate(self.datasets):
                self.tid_seq.extend([i] * len(ds))
                self.tidx_seq.extend(list(range(len(ds))))
                self.task_boundaries.append(len(self.tid_seq))

    def __len__(self):
        return len(self.tidx_seq)
    
    def __getitem__(self, idx):
        tid = self.tid_seq[idx]
        tidx = self.tidx_seq[idx]
        ds = self.datasets[tid]

        return {
            "tid": getattr(ds, "name", None),
            **ds.__getitem__(tidx)
        }

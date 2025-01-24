from torch.utils.data import Dataset


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

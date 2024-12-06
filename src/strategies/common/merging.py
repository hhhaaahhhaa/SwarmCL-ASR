import torch


def quantile(x: torch.Tensor, q: float) -> torch.Tensor:  # https://github.com/pytorch/pytorch/issues/64947
    num = x.numel()
    if num < 16_000_000:
        threshold = torch.quantile(x, q)
    else:
        sorted, _ = torch.sort(x.reshape(-1))
        idxf = q * num
        idxi = int(idxf)
        threshold = sorted[idxi] + (sorted[idxi + 1] - sorted[idxi]) * (idxf - idxi)
    return threshold


class TIES(object):
    """ TIES merging for 1D vector. """
    def __init__(self, density=1.0) -> None:
        self.density = density
    
    def get_tv(self, ref_vector: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return vector - ref_vector
    
    def prune(self, vector: torch.Tensor) -> torch.Tensor:
        vector_abs = vector.abs()
        threshold = quantile(vector_abs, 1 - self.density)        
        mask = vector_abs >= threshold
        return vector * mask

    def resolve_sign(self, vectors: list[torch.Tensor]) -> torch.BoolTensor:
        return torch.sum(torch.stack(vectors, dim=0), dim=0) > 0
    
    def disjoint_merge(self, vectors: list[torch.Tensor], sign: torch.BoolTensor) -> torch.Tensor:
        all = torch.stack(vectors, dim=0)
        mask = torch.where(sign.unsqueeze(0), all > 0, all < 0)
        remained_entries = all * mask

        non_zero_counts = (remained_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(remained_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
        return disjoint_aggs
    
    def merge(self, ref_vector: torch.Tensor, vectors: list[torch.Tensor]) -> torch.Tensor:
        tvs = [self.get_tv(ref_vector, vector) for vector in vectors]
        tvs = [self.prune(tv) for tv in tvs]
        sign = self.resolve_sign(tvs)
        return self.disjoint_merge(tvs, sign)

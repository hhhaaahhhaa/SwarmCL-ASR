import os
import torch
import pickle

from src.systems.base import System
from src.systems.wav2vec2 import Wav2vec2System
from ..common.interface import IParticle


class ModelParticle(IParticle):

    _data: dict[str: torch.Tensor]
    SAVE_RAM = False

    def __init__(self, data: dict[str: torch.Tensor]={}) -> None:
        self._data = data
        self._cache_path = None

    @classmethod
    def dummy(cls) -> IParticle:
        return cls()
    
    def is_dummy(self) -> bool:
        return True if (not self._data and self._cache_path is None) else False

    def cache(self, root: str, tag: str) -> None:
        if not ModelParticle.SAVE_RAM:
            return
        [it, name] = tag.split(":")
        os.makedirs(f"{root}/{it}", exist_ok=True)
        cache_path = f"{root}/{it}/{name}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(self.data, f)
        self._cache_path = cache_path
        self.data = {}

    def get_data(self) -> dict[str: torch.Tensor]:
        if self._cache_path is None:
            return self._data
        # always load from disk to save RAM(but increase IO time) since the model might be very large.
        with open(self._cache_path, "rb") as f:
            data = pickle.load(f)
        return data


def linear_combination(coefficients: list[float], particles: list[ModelParticle]) -> ModelParticle:
    assert len(coefficients) == len(particles)
    keys = None
    for particle in particles:
        if not particle.is_dummy():
            keys = particle.get_data().keys()
    if keys is None:  # all dummy
        return ModelParticle.dummy()
    data = {k: 0 for k in keys}
    for (coeff, particle) in zip(coefficients, particles):
        if particle.is_dummy():
            continue
        for k, v in particle.get_data().items():
            data[k] = data[k] + coeff * v
    return ModelParticle(data)


def system2particle(system: System) -> ModelParticle:
    # assert isinstance(system, Wav2vec2System)
    data = {}
    params, names = system._collect_params()
    for (name, param) in zip(names, params):
        data[name] = param.detach().cpu().clone()
    return ModelParticle(data)


def particle2system(particle: ModelParticle, ref_system: System) -> System:
    """ require an reference system object since particle only contains weight information """
    # assert isinstance(ref_system, Wav2vec2System)
    state_dict = ref_system.model.state_dict()
    for name, param in particle.get_data().items():
        state_dict[name] = param.detach().clone().to(ref_system.device)
    ref_system.model.load_state_dict(state_dict)
    return ref_system

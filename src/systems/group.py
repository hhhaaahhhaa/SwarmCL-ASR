import os
import torch
import json
import copy

from one import load_system
from .wav2vec2 import Wav2vec2System


class Wav2vec2Group(object):

    tid2gid: dict[str, str]
    _cur: str
    _cur_system: Wav2vec2System

    def __init__(self, config: dict) -> None:
        self.config = config
        self.particles = {}
        self.tid2gid = {}

        self._cur = None
        self._cur_system = None
    
    def cuda(self):
        pass

    def eval(self):
        pass

    def _get_initial_system(self):
        return load_system(
            system_name="wav2vec2",
            system_config=copy.deepcopy(self.config)
        )
    
    def _get_system_from_raw_path(self, path):
        return load_system(
            system_name="wav2vec2",
            checkpoint=path,
            loader="lightning"
        )

    def load(self, dir: str) -> None:
        self.dir = dir
        with open(f"{dir}/info.json", "r") as f:
            self.info = json.load(f)

    @torch.no_grad()
    def inference(self, wavs, tid: str):
        gid = self.info[tid]  # gid can be a path in order to support mmodel which might have 20+ checkpoints
        if self._cur != gid:  # reduce reloading
            if self.info.get("use_raw_path", False):
                self._cur_system = self._get_system_from_raw_path(gid)
            else:
                self._cur_system = self._get_initial_system()
                self._cur_system.load(f"{self.dir}/{gid}.ckpt")
            self._cur_system.eval()
            self._cur_system.cuda()
            self._cur = gid
        return self._cur_system.inference(wavs)
    
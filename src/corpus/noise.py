import os
from pathlib import Path
import numpy as np
import torch
import json
import librosa

from . import Define


class MUSANNoiseCorpus(object):
    """ free-sound noises from MUSAN """
    def __init__(self) -> None:
        self.root = f"{Define.MUSAN}/noise/free-sound"

    def __len__(self):
        return 843
    
    def get(self, idx: int) -> np.ndarray:
        basename = f"noise-free-sound-{idx % 843:04d}"
        noise, _ = librosa.load(f"{self.root}/{basename}.wav", sr=16000)

        return noise


class MSSNSD10(object):

    type2noisefilename = {
        "AC": "AirConditioner_6",
        "AA": "AirportAnnouncements_2",
        "BA": "Babble_4",
        "CM": "CopyMachine_2",
        "MU": "Munching_3",
        "NB": "Neighbor_6",
        "SD": "ShuttingDoor_6",
        "TP": "Typing_2",
        "VC": "VacuumCleaner_1",
        "GS": None,  # Gaussian noise
    }
        
    def get(self, noise_type: str) -> np.ndarray:
        assert noise_type in self.type2noisefilename
        noise_filename = self.type2noisefilename[noise_type]
        if noise_type == "GS":
            noise = None
            # noise = np.random.randn(*clean_wav.shape)
        else:
            noise, _ = librosa.load(f"preprocess/res/{noise_filename}.wav", sr=16000)
        return noise

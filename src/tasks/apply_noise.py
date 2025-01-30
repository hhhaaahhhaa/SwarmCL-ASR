import numpy as np
import librosa
from torch.utils.data import Dataset


def snr_mixer(clean, noise, snr):
    # Normalizing to -25 dB FS
    rmsclean = (clean**2).mean()**0.5
    # print("clean(before): ", rmsclean)
    scalarclean = 10 ** (-25 / 20) / rmsclean
    # print(scalarclean)
    clean = clean * scalarclean
    rmsclean = (clean**2).mean()**0.5
    # print("clean(normalized): ", rmsclean)

    rmsnoise = (noise**2).mean()**0.5
    # print("noise(before): ", rmsnoise)
    scalarnoise = 10 ** (-25 / 20) / rmsnoise
    # print(scalarnoise)
    noise = noise * scalarnoise
    rmsnoise = (noise**2).mean()**0.5
    # print("noise(normalized): ", rmsnoise)
    # wavfile.write("normed_AC.wav", 16000, (noise * 32767).astype(np.int16))
    
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr/20)) / rmsnoise
    # print(noisescalar)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return noisyspeech


def add_noise(clean_wav, noise_type, snr_level):
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
    assert noise_type in type2noisefilename
    noise_filename = type2noisefilename[noise_type]

    if noise_type == "GS":
        np.random.seed(666)  # fixed so that the GS noise is always the same
        noise = np.random.randn(*clean_wav.shape).astype(np.float32)
    else:
        noise, _ = librosa.load(f"preprocess/res/{noise_filename}.wav", sr=16000)

    # repeat noise content if too short
    noiseconcat = noise
    while len(noiseconcat) <= len(clean_wav):
        noiseconcat = np.append(noiseconcat, noise)
    noise = noiseconcat
    if len(noise) > len(clean_wav):
        noise = noise[0:len(clean_wav)]

    return snr_mixer(clean_wav, noise, snr=snr_level)

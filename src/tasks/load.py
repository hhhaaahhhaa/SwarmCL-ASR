from src.utils.load import get_class_in_module
from .base import Task


SRC_DIR = "src/tasks"

CV_ACCENT = {
    "cv-aus": (f"{SRC_DIR}/commonvoice.py", "AUSTask"),
    "cv-eng": (f"{SRC_DIR}/commonvoice.py", "ENGTask"),
    "cv-ind": (f"{SRC_DIR}/commonvoice.py", "INDTask"),
    "cv-ire": (f"{SRC_DIR}/commonvoice.py", "IRETask"),
    "cv-sco": (f"{SRC_DIR}/commonvoice.py", "SCOTask"),
    "cv-us": (f"{SRC_DIR}/commonvoice.py", "USTask"),

    "cv-seq": (f"{SRC_DIR}/commonvoice.py", "CVSequence"),
}

EXP = {
    "long1": (f"{SRC_DIR}/long.py", "Long1Sequence"),
    "long2": (f"{SRC_DIR}/long.py", "Long2Sequence"),
    "long2a": (f"{SRC_DIR}/long.py", "Long2ASequence"),
    "long2b": (f"{SRC_DIR}/long.py", "Long2BSequence"),
    "long3a": (f"{SRC_DIR}/long.py", "Long3ASequence"),
    "long3b": (f"{SRC_DIR}/long.py", "Long3BSequence"),
    "long3c": (f"{SRC_DIR}/long.py", "Long3CSequence"),
}

TASK_MAPPING = {
    **CV_ACCENT,
    **EXP,
}


def get_task(name) -> Task:
    if name.startswith("LS_"):  # e.g. LS_AA_5
        from . import librispeech
        types = name.split("_")
        noise_type, snr_level = types[1], int(types[2])
        task = librispeech.LibriSpeechNoiseTask(noise_type, snr_level=snr_level)
        return task
    
    if name.startswith("cv-") and "_" in name:  # e.g. cv-eng_AA_5
        from . import commonvoice
        types = name.split("_")
        accent, noise_type, snr_level = types[0][3:], types[1], int(types[2])
        task = commonvoice.CVAccentNoiseTask(accent, noise_type, snr_level=snr_level)
        return task
    
    module_path, class_name = TASK_MAPPING[name]
    return get_class_in_module(class_name, module_path)()

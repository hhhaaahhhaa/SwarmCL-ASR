import typing

from src.utils.load import get_class_in_module
from .base import System


SRC_DIR = "src/systems"

BASIC = {
    "wav2vec2": (f"{SRC_DIR}/wav2vec2.py", "Wav2vec2System"),
    "wav2vec2-er": (f"{SRC_DIR}/wav2vec2.py", "Wav2vec2ERSystem"),
}

GROUP = {
    "wav2vec2-group": (f"{SRC_DIR}/group.py", "Wav2vec2Group"),  # does not inherit System...
}

SYSTEM_MAPPING = {
    **BASIC,
    **GROUP,
}


def get_system_cls(name: str) -> typing.Type[System]:
    module_path, class_name = SYSTEM_MAPPING[name]
    return get_class_in_module(class_name, module_path)

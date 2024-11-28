import typing

from src.utils.load import get_class_in_module
from .base import IStrategy


SRC_DIR = "src/strategies"

BASIC = {
    
}

CONTINUAL = {
    "swarm-cl": (f"{SRC_DIR}/swarm_cl/strategy.py", "SwarmCLStrategy"),
}

STRATEGY_MAPPING = {
    **BASIC,
    **CONTINUAL,
}


def get_strategy_cls(name: str) -> typing.Type[IStrategy]:
    module_path, class_name = STRATEGY_MAPPING[name]
    return get_class_in_module(class_name, module_path)

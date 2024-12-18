import typing

from src.utils.load import get_class_in_module
from .base import IStrategy


SRC_DIR = "src/strategies"

BASIC = {
    
}

MERGING = {
    "uniform-soup": (f"{SRC_DIR}/merging.py", "UniformSoup"),
    "ties": (f"{SRC_DIR}/merging.py", "TIES"),
    "greedy-soup": (f"{SRC_DIR}/merging.py", "GreedySoup"),
    "swarm": (f"{SRC_DIR}/merging.py", "ModelSwarm"),
}

CONTINUAL = {
    "seq-linear": (f"{SRC_DIR}/seq_linear.py", "SeqLinearStrategy"),
    # "seq-ties": (f"{SRC_DIR}/swarm_cl/strategy.py", "SwarmCLStrategy"),
    "seq-swarm": (f"{SRC_DIR}/swarm_cl/strategy.py", "SwarmCLStrategy"),
}

STRATEGY_MAPPING = {
    **BASIC,
    **MERGING,
    **CONTINUAL,
}


def get_strategy_cls(name: str) -> typing.Type[IStrategy]:
    module_path, class_name = STRATEGY_MAPPING[name]
    return get_class_in_module(class_name, module_path)

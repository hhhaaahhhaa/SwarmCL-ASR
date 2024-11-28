from tqdm import tqdm

from one import train_one_task
from src.strategies.base import IStrategy
from .utility import WERUtility


class SwarmCLStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
    
    def run(self, data_obj):
        pass

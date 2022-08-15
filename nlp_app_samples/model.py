
from nlp_app_samples.optimizer import Optimizer


class Dataset:
    name: str
    path: str

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path

    def __repr__(self):
        return repr(self.__dict__)

class Trainer:

    def __init__(self, optimizer: Optimizer, dataset: Dataset) -> None:
        self.optimizer = optimizer
        self.dataset = dataset

    def __repr__(self):
        return repr(self.__dict__)


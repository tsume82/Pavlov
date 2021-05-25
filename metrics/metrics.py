from abc import ABC, abstractmethod
from builtins import Exception

import numpy as np

_metrics = {}


class UnknownMetricException(Exception):
    def __init__(self,name):
        message = "The metric identifier \"{}\" provided does not correspond to a registered metric implementation".format(name)
        super().__init__(message)

class MetricProvider():
    @staticmethod
    def register_metric(name, clazz):
        _metrics[name] = clazz

    @staticmethod
    def get_metric(name) -> str:
        if name in _metrics.keys():
            return _metrics[name]
        else:
            raise UnknownMetricException(name)

class Metric(ABC):
    def __init__(self):
        MetricProvider.register_metric((self.name, self.__class__))

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, state: np.array, **options) -> np.array:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

# TODO
class RecentGradients(Metric):
    name = "RecentGradients"
    def compute(self, state: np.array, **options) -> np.array:
        pass

    def reset(self) -> None:
        pass
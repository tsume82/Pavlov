from abc import ABC, abstractmethod
from builtins import Exception, map
from gym.spaces import *
import warnings

import numpy as np

_metrics = {}


class UnknownMetricException(Exception):
    def __init__(self, name):
        message = "The metric identifier \"{}\" provided does not correspond to a registered metric implementation".format(
            name)
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
        MetricProvider.register_metric(self.name, self.__class__)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    # return gym.Env-usable space
    @abstractmethod
    def get_space(self):
        pass

    @abstractmethod
    def compute(self, solutions: np.array, **options) -> np.array:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class RecentGradients(Metric):
    """
        RecentGradient metric, keep track of fitness gradients, computing it with custom steps over solution list
    """

    name = "RecentGradients"

    def __init__(self, dim, max_archive=None, chunk_size=1, chunk_num=None, chunk_use_last=1):
        super().__init__()
        self.max_archive = max_archive
        self.dim = dim

        if chunk_num is not None and max_archive is not None:
            chunk_size = max(1, max_archive // chunk_num)
        else:
            chunk_size = chunk_size  # group in chunks
        self.chunk_size = chunk_size

        if chunk_use_last is None and max_archive is None:
            warnings.warn("Invalid number of chunks to consider (chunk_use_last param): cannot be automatically " +
                          "inferred with unbounded archive size (use max_archive param to set an archive bound)." +
                          "Setting chunk_use_last to 1.")
        elif chunk_use_last is None and max_archive is not None:
            chunk_use_last = max_archive // chunk_size
        self.chunk_use_last = chunk_use_last

        self.archive = None
        self.reset()

    def compute(self, fitness_list: np.array, **options) -> np.array:
        self.archive = np.vstack((self.archive, fitness_list))
        # only consider last chunks_use_last chunks of size chunk_size
        considering = self.archive[[*range(
            self.archive.shape[0] - 1,
            max(-1, self.archive.shape[0] - (self.chunk_size * (1 + self.chunk_use_last))),
            -self.chunk_size
        )]]
        # print(f"considering archive subset: {considering}")
        # TODO implement max_archive efficiently, avoid continue deletion and use rolling index similar to MemePolicy
        if self.max_archive is not None:
            oversize = self.archive.shape[0] - self.max_archive
            if oversize > 0:
                self.archive = np.delete(self.archive, range(oversize), axis=0)
        # subtract previous fitness, along each axis
        grads = considering - np.roll(considering, 1, 0)
        # delete oldest entry, whose gradient is not meaningful having no predecessor in the archive
        # print(f"pre-deletion grads: {grads}")
        grads = np.delete(grads, 0, axis=0)

        if options.get("autoreset", False):
            self.reset()

        return grads

    def reset(self) -> None:
        self.archive = np.zeros(shape=(0, self.dim))

    def get_space(self):
        space = Box(low=-np.inf, high=np.inf, shape=(self.chunk_use_last, self.dim))
        return space

from abc import ABC, abstractmethod
from builtins import Exception, map
from gym import spaces
import warnings

import numpy as np


class UnknownMetricException(Exception):
    def __init__(self, name):
        message = "The metric identifier \"{}\" provided does not correspond to a registered metric implementation".format(
            name)
        super().__init__(message)


class MetricProvider():
    _metrics = {}

    @classmethod
    def register_metric(cls, name, clazz):
        cls._metrics[name] = clazz

    @classmethod
    def get_metric(cls, name):
        if name in cls._metrics.keys():
            return cls._metrics[name]
        else:
            raise UnknownMetricException(name)

    @classmethod
    def build(cls):
        for k,v in cls._metrics.items():
            if type(v) == str:
                cls._metrics[v] = eval(v)

    @classmethod
    def combine(cls, metricsNames):
        class CombinedMetric(Metric):

            name = "_".join(metricsNames)
            metrics_types = tuple((MetricProvider.get_metric(m) for m in metricsNames))

            def __init__(self, configs):
                self.metrics = tuple((self.metrics_types[i](*configs[i]) for i in range(len(self.metrics_types))))

            def get_space(self):
                return spaces.Tuple([m.get_space() for m in self.metrics])

            def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
                # TODO may add mask attribute to compute metrics selectively
                return tuple([m.compute(solutions, fitness, **options) for m in self.metrics])

            def reset(self) -> None:
                for m in self.metrics:
                    m.reset()

        return CombinedMetric


class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    # return gym.Env-usable space
    @abstractmethod
    def get_space(self):
        pass

    @abstractmethod
    def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class RecentGradients(Metric):
    """
        RecentGradient metric, keep track of fitness gradients, computing it with custom steps over solution list
    """

    name = "RecentGradients"
    MetricProvider.register_metric(name, __qualname__)

    def __init__(self, dim, max_archive=None, chunk_size=1, chunk_num=None, chunk_use_last=1, on_fitness=True):
        super().__init__()
        self.max_archive = max_archive
        self.dim = dim
        self.on_fitness = on_fitness

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

    def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
        dataset = fitness if self.on_fitness else solutions

        self.archive = np.vstack((self.archive, dataset))
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
        if considering.shape[0] > 1:
            grads = np.delete(grads, 0, axis=0)

        if options.get("autoreset", False):
            self.reset()

        return grads

    def reset(self) -> None:
        self.archive = np.zeros(shape=(0, self.dim))

    def get_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.chunk_use_last, self.dim))

class RecentFitness(Metric):
    name = "RecentFitness"
    MetricProvider.register_metric(name, __qualname__)

    def __init__(self) -> None:
        super().__init__()

    def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
        return super().compute(solutions, fitness, **options)

    def get_space(self):
        return super().get_space()

    def reset(self) -> None:
        return super().reset()

class Best(Metric):

    name = "Best"
    MetricProvider.register_metric(name, __qualname__)

    def __init__(self, fit_dim=1, fit_index=0):
        self.fit_dim = fit_dim
        self.fit_index = fit_index
        assert 0 <= self.fit_index < self.fit_dim
        self.best = None
        self.best_sol = None
        self.reset()

    def get_space(self):
        space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1))
        return space

    def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
        indexes = np.argmin(fitness, axis=0)
        if len(indexes.shape) == 0:
            curr_best_index = indexes
            curr_best_fit = fitness[curr_best_index]
        else:
            curr_best_index = indexes[self.fit_index]
            curr_best_fit = fitness[curr_best_index, self.fit_index]
        if curr_best_fit < self.best:
            self.best = curr_best_fit
            self.best_sol = solutions[curr_best_index]
        return self.best

    def reset(self) -> None:
        self.best = np.inf
        self.best_sol = None

    def get_best(self):
        return self.best, self.best_sol


# build up MetricProvider registered metrics class types
# NB: this must be the last line of metrics.py
MetricProvider.build()

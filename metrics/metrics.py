from abc import ABC, abstractmethod
from builtins import Exception, map
from gym import spaces
from ray.rllib.utils.spaces.repeated import Repeated
import warnings

import numpy as np


class UnknownMetricException(Exception):
	def __init__(self, name):
		message = (
			'The metric identifier "{}" provided does not correspond to a registered metric implementation'.format(
				name
			)
		)
		super().__init__(message)


class MetricProvider:
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
		for k, v in cls._metrics.items():
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
				return [m.compute(solutions, fitness, **options) for m in self.metrics]

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

	def __init__(self, dim, max_archive=None, chunk_size=1, chunk_num=None, chunk_use_last=2, on_fitness=True):
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
			warnings.warn(
				"Invalid number of chunks to consider (chunk_use_last param): cannot be automatically "
				+ "inferred with unbounded archive size (use max_archive param to set an archive bound)."
				+ "Setting chunk_use_last to 1."
			)
		elif chunk_use_last is None and max_archive is not None:
			chunk_use_last = max_archive // chunk_size
		self.chunk_use_last = chunk_use_last

		self.archive = None
		self.reset()

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		dataset = fitness if self.on_fitness else solutions

		self.archive = np.vstack((self.archive, dataset))
		# only consider last chunks_use_last chunks of size chunk_size
		considering = self.archive[
			[
				*range(
					self.archive.shape[0] - 1,
					max(-1, self.archive.shape[0] - (self.chunk_size * (1 + self.chunk_use_last))),
					-self.chunk_size,
				)
			]
		]
		# print(f"considering archive subset: {considering}")
		# TODO implement max_archive efficiently, avoid continue deletion and use rolling index similar to MemePolicy
		if self.max_archive is not None:
			oversize = self.archive.shape[0] - self.max_archive
			if oversize > 0:
				self.archive = np.delete(self.archive, range(oversize), axis=0)
		# subtract previous fitness, along each axis
		grads = considering - np.roll(considering, 1, 0)
		# delete oldest entry, whose gradient is not meaningful having no predecessor in the archive
		grads = np.delete(grads, 0, axis=0)

		if options.get("autoreset", False):
			self.reset()

		return grads.tolist()  # Repeated doesn't handle numpy arrays, TODO custom Repeated handling numpy?

	def reset(self) -> None:
		self.archive = np.zeros(shape=(0, self.dim))

	def get_space(self):
		# return spaces.Box(low=-np.inf, high=np.inf, shape=(self.chunk_use_last, self.dim))
		box = spaces.Box(low=-np.inf, high=np.inf, shape=((self.dim,)))
		return Repeated(box, max_len=self.chunk_use_last)


class DifferenceOfBest(Metric):
	"""
	Get the difference of the current best fitness and the precedent best fitness
	"""

	name = "DifferenceOfBest"
	MetricProvider.register_metric(name, __qualname__)

	def __init__(self, history_max_length=1, maximize=True, fitness_dim=1, relative=True, normalize=False):
		self.prec_best = None
		self.maximize = maximize
		self.fitness_dim = fitness_dim
		self.relative = relative
		self.normalize = normalize
		self.history_max_length = history_max_length
		self.history = []

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		if self.prec_best is None:
			self.prec_best = np.nanmax(fitness, axis=0) if self.maximize else np.nanmin(fitness, axis=0)
			self.history.insert(0, np.array(0, dtype=np.float32))
		else:
			curr_best = np.nanmax(fitness, axis=0) if self.maximize else np.nanmin(fitness, axis=0)
			delta = curr_best - self.prec_best

			if self.relative:
				if self.normalize:
					delta /= abs(curr_best - self.prec_best) + abs(self.prec_best) + 1e-5
				else:
					delta /= abs(self.prec_best) + 1e-3

			self.history.insert(0, np.array(delta.item()))  # Repeated needs a list, Box needs a np.array as a scalar
			if len(self.history) > self.history_max_length:
				self.history.pop()

			self.prec_best = curr_best

		return self.history

	def reset(self) -> None:
		self.prec_best = None
		self.history = []

	def get_space(self):
		low = -np.inf
		high = np.inf

		if self.relative:
			if self.normalize:
				low = -1
				high = 1
			else:
				low = -1e6
				high = 1e6

		box = spaces.Box(low=low, high=high, shape=([]))
		return Repeated(box, self.history_max_length)


class DeltaFitPop(Metric):
	"""
	History of the normaized difference between max fitness and min fitness in the population
	"""
	name = "DeltaFitPop"
	MetricProvider.register_metric(__qualname__, __qualname__)

	def __init__(self, history_max_length=1, maximize=True):
		self.maximize = maximize
		# self.normalize = normalize
		self.history_max_length = history_max_length
		self.history = []

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		max = np.nanmax(fitness, axis=0)
		min = np.nanmin(fitness, axis=0)
		
		deltaFitPop = abs(max - min) / (abs(max - min) + abs(max if self.maximize else min) + 1e-5)
		
		self.history.insert(0, np.array(deltaFitPop.item()))  # Repeated needs a list, Box needs a np.array as a scalar
		if len(self.history) > self.history_max_length:
			self.history.pop()

		return self.history

	def reset(self) -> None:
		self.history = []

	def get_space(self):
		box = spaces.Box(low=0, high=1, shape=([]))
		return Repeated(box, self.history_max_length)

class DeltaX(Metric):
	"""
	History of deltaX intra or inter generation.

	intra-generation deltaX: vector of the difference between the maximum and the minimum value of the search space of the objective function for each dimension

	inter-generation deltaX: vector of the difference between the two best solutions in 2 genereations.
	"""
	name = "DeltaX"
	MetricProvider.register_metric(__qualname__, __qualname__)

	def __init__(self, history_max_length=1, maximize=True, intra_gen = True):
		self.maximize = maximize
		self.history_max_length = history_max_length
		self.history = []
		self.intra_gen = intra_gen
		self.compute = self._intra_deltaX if intra_gen else self._inter_deltaX

		# if domain_bounds is not None:
		# 	bounds = np.array(domain_bounds)
		# 	if len(bounds.shape) == 1: # bounds can be written as a 1D vector with 2*dim elements
		# 		bounds = bounds.reshape((bounds.shape[0]//2, 2))
		# 	self.bounds_range = np.abs(bounds.T[0] - bounds.T[1])

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		pass # this function will be _intra_deltaX or _inter_deltaX
	
	def _intra_deltaX(self, solutions: np.array, fitness: np.array, **options):
		bounds = np.array(options["bounds"])
		bounds_range = np.abs(bounds.T[0] - bounds.T[1])

		max = np.nanmax(solutions, axis=0)
		min = np.nanmin(solutions, axis=0)
		
		deltaX = np.abs(max - min) / bounds_range

		min_and_max_deltaX = np.array([np.max(deltaX), np.min(deltaX)])
		
		self.history.insert(0, min_and_max_deltaX)
		if len(self.history) > self.history_max_length:
			self.history.pop()

		return self.history

	def _inter_deltaX(self, solutions: np.array, fitness: np.array, **options):
		best_idx = np.argmax(fitness) if self.maximize else np.argmin(fitness)

		if len(self.history) == 0:
			self.history.insert(0, np.zeros(shape=[self.dim], dtype=np.float32))
			self.prec = solutions[best_idx]
		else:
			bounds = np.array(options["bounds"])
			bounds_range = np.abs(bounds.T[0] - bounds.T[1])

			deltaX = (solutions[best_idx] - self.prec) / bounds_range

			min_and_max_deltaX = np.array([np.max(deltaX), np.min(deltaX)])
			
			self.history.insert(0, min_and_max_deltaX)
			if len(self.history) > self.history_max_length:
				self.history.pop()

		return self.history

	def reset(self) -> None:
		self.history = []

	def get_space(self):
		low = 0 if self.intra_gen else -1
		box = spaces.Box(low=low, high=1, shape=[2])
		return Repeated(box, self.history_max_length)


class FitnessHistory(Metric):
	"""
	RecentFitness metric, keep track of the fitness within the last history_size steps
	most recent is in [0]
	"""

	name = "FitnessHistory"
	MetricProvider.register_metric(name, __qualname__)

	def __init__(self, dim, history_size) -> None:
		self.dim = dim
		self.history_size = history_size
		self.archive = None

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		self.archive.insert(0, fitness)

		if len(self.archive) > self.history_size:
			self.archive.pop()

		return self.archive

	def get_space(self):
		box = spaces.Box(low=-np.inf, high=np.inf, shape=((self.dim,)))
		return Repeated(box, max_len=self.history_size)

	def reset(self) -> None:
		self.archive = []


class BestsHistory(Metric):
	"""
	BestsHistory metric, history of the best fitness values
	"""

	name = "BestsHistory"
	MetricProvider.register_metric(name, __qualname__)
	default_bounds = {"max": np.inf, "min": -np.inf}

	def __init__(self, history_size, maximize=True, bounds=default_bounds, normalize=True) -> None:
		self.history_size = history_size
		self.maximize = maximize
		self.normalize = True if normalize and bounds != self.default_bounds else False
		self.bounds = bounds
		self.history = None

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		best = np.max(fitness) if self.maximize else np.min(fitness)

		if self.normalize:
			best = (best - self.bounds["min"]) / (self.bounds["max"] - self.bounds["min"])

		self.history.insert(0, best)

		if len(self.history) > self.history_size:
			self.history.pop()

		return self.history

	def get_space(self):
		box = spaces.Box(
			low=0 if self.normalize else self.bounds["min"],
			high=1 if self.normalize else self.bounds["max"],
			shape=([]),
		)
		return Repeated(box, max_len=self.history_size)

	def reset(self) -> None:
		self.history = []


class Best(Metric):

	name = "Best"
	MetricProvider.register_metric(name, __qualname__)

	def __init__(self, maximize=True, use_best_of_run=False, fit_dim=1, fit_index=0):
		self.fit_dim = fit_dim
		self.fit_index = fit_index
		self.maximize = maximize
		self.use_best_of_run = use_best_of_run
		assert 0 <= self.fit_index < self.fit_dim
		self.best = None
		self.best_sol = None
		self.reset()

	def get_space(self):
		space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1))
		return space

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		indexes = np.argmax(fitness, axis=0) if self.maximize else np.argmin(fitness, axis=0)
		if len(indexes.shape) == 0:
			curr_best_index = indexes
			curr_best_fit = fitness[curr_best_index]
		else:
			curr_best_index = indexes[self.fit_index]
			curr_best_fit = fitness[curr_best_index, self.fit_index]
		if curr_best_fit < self.best:
			self.best = curr_best_fit
			self.best_sol = solutions[curr_best_index]
		return self.best if self.use_best_of_run else curr_best_fit  # self.best or curr_best_fit?

	def reset(self) -> None:
		self.best = np.inf
		self.best_sol = None

	def get_best(self):
		return self.best, self.best_sol


class DeltaBest(Metric):
	name = "DeltaBest"
	MetricProvider.register_metric(name, __qualname__)

	def __init__(self, maximize=True, use_best_of_run=False, normalize=False):
		self.maximize = maximize
		if maximize:
			self.compare = lambda a, b: a > b
		else:
			self.compare = lambda a, b: a < b
		self.normalize = normalize
		self.use_best_of_run = use_best_of_run
		self.best = None
		self.prec_best = None
		self.best_sol = None
		self.reset()

	def get_space(self):
		if self.normalize:
			return spaces.Box(low=-1, high=1, shape=(1, 1))
		else:
			return spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1))

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		indexes = np.argmax(fitness, axis=0) if self.maximize else np.argmin(fitness, axis=0)
		if len(indexes.shape) == 0:
			curr_best_index = indexes
			curr_best_fit = fitness[curr_best_index]
		if self.compare(curr_best_fit, self.best):
			self.best = curr_best_fit
			self.best_sol = solutions[curr_best_index]
		best = self.best if self.use_best_of_run else curr_best_fit

		if self.prec_best is not None:
			if self.normalize:
				# delta_best = np.tanh((best - self.prec_best)/(abs(self.prec_best)+1e-5))
				delta_best = (best - self.prec_best)/(abs(best-self.prec_best)+abs(self.prec_best)+1e-5)
				# delta_best = np.tanh((best - self.prec_best)/(abs(best)+1e-5))
			else:
				delta_best = (best - self.prec_best) / (abs(self.prec_best) + 1e-5) # 3
		else:
			delta_best = 0

		self.prec_best = best

		return delta_best

	def reset(self) -> None:
		self.best = np.inf
		self.best_sol = None
		self.prec_best = None


class SolverState(Metric):

	name = "SolverState"
	MetricProvider.register_metric(name, __qualname__)

	def __init__(self, solver_states_bounds: dict):
		self.solver_states_bounds = solver_states_bounds

	def get_space(self):
		return spaces.Dict(
			{
				key: spaces.Box(low=np.array(value["min"]), high=np.array(value["max"]))
				for (key, value) in self.solver_states_bounds.items()
			}
		)

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		return {key: value for (key, value) in options.items() if key in self.solver_states_bounds.keys()}

	def reset(self):
		return


class SolverStateHistory(SolverState):

	name = "SolverStateHistory"
	MetricProvider.register_metric(name, __qualname__)

	def __init__(self, solver_states_bounds: dict, history_max_length=1):
		super().__init__(solver_states_bounds)
		self.history = []
		self.history_max_length = history_max_length

	def get_space(self):
		return Repeated(
			super().get_space(), max_len=self.history_max_length
		)  # is better Repeated of Dict or Dict of Repeated?

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		self.history.insert(0, super().compute(solutions, fitness, **options))

		if len(self.history) > self.history_max_length:
			self.history.pop()

		return self.history

	def reset(self) -> None:
		self.history = []


class AvgPosition(Metric):
	name = "AvgPosition"
	MetricProvider.register_metric(name, __qualname__)

	def __init__(self, dim, bounds={"max": np.inf, "min": -np.inf}, withVariance=False) -> None:
		self.dim = dim
		self.bounds = bounds
		self.withVariance = withVariance

	def get_space(self):
		if not self.withVariance:
			return spaces.Box(low=self.bounds["min"], high=self.bounds["max"], shape=([self.dim]))
		return spaces.Tuple(
			[
				spaces.Box(low=self.bounds["min"], high=self.bounds["max"], shape=([self.dim])),
				spaces.Box(low=-50, high=50, shape=([self.dim])),
			]
		)

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		if not self.withVariance:
			return np.average(solutions, axis=0)
		return np.average(solutions, axis=0), np.var(solutions, axis=0)

	def reset(self) -> None:
		pass


# build up MetricProvider registered metrics class types
# NB: this must be the last line of metrics.py
MetricProvider.build()

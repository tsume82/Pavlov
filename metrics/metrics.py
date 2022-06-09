from abc import ABC, abstractmethod
from builtins import Exception
from gym import spaces
from ray.rllib.utils.spaces.repeated import Repeated
import numpy as np

class MetricProvider:
	_metrics = {}

	class UnknownMetricException(Exception):
		def __init__(self, name):
			message = (f'The metric identifier {name} provided does not correspond to a registered metric implementation')
			super().__init__(message)

	@classmethod
	def register_metric(cls, name, clazz):
		cls._metrics[name] = clazz

	@classmethod
	def get_metric(cls, name):
		if name in cls._metrics.keys():
			return cls._metrics[name]
		else:
			raise cls.UnknownMetricException(name)

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


class InterDeltaF(Metric):
	"""
		Inter Generational Delta F: normalized difference between the best fitness of the current generation and the best fitness of the previous generation.
		It is a measure of the improvement of the optimization process.

		If "use_fitness_bound = True" , the fitness is normalized with a bound range, maintaining linearity between the two fitness 
		(requires "fitness_bound" option passed from the solver driver in step()).
		Otherwise, the normalization is a ratio proportional to the absolute value of the previous fitness.

		with fitness_bound = [min_b, max_b]:
			Inter-ΔF = |curr_best - self.prec_best} / |min_b - max_b|
		without fitness_bound:
			Inter-ΔF = |curr_best - self.prec_best| / (|curr_best - self.prec_best| + |self.prec_best| + 1e-5)
	"""

	name = "InterDeltaF"
	MetricProvider.register_metric(__qualname__, __qualname__)

	def __init__(self, use_fitness_bound=False):
		self.prec_best = None
		if use_fitness_bound:
			self.deltaF_fun = lambda c, p, b: abs(c-p) / abs(b[0]-b[1])
		else:
			self.deltaF_fun = lambda c, p, b: abs(c-p) / (abs(c-p) + abs(p) + 1e-5)

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		if self.prec_best is None:
			self.prec_best = np.nanmin(fitness, axis=0)
			return 0
		
		curr_best = np.nanmin(fitness, axis=0)
		deltaF = self.deltaF_fun(curr_best, self.prec_best, options.get('fitness_bound', None))
		self.prec_best = curr_best

		return deltaF

	def reset(self) -> None:
		self.prec_best = None

	def get_space(self):
		return spaces.Box(low=0, high=1, shape=([]))


class IntraDeltaF(Metric):
	"""
		Normalized difference between max fitness and min fitness in the population.
		It is a measure of the range of the fitness in the population.

		If "use_fitness_bound = True" , the fitness is normalized with a bound range, maintaining linearity between the two fitness
		(requires "fitness_bound" option passed from the solver driver in step()).
		Otherwise, the normalization is a ratio proportional to the absolute value of the min fitness of the population.

		max = max(fitness), min = min(fitness)
		with fitness_bound = [min_b, max_b]:
			Intra-ΔF = |max - min} / |min_b - max_b|
		without fitness_bound:
			Intra-ΔF = |max - min| / (|max - min| + |min| + 1e-5)
	"""
	name = "IntraDeltaF"
	MetricProvider.register_metric(__qualname__, __qualname__)

	def __init__(self, use_fitness_bound=False):
		if use_fitness_bound:
			self.deltaF_fun = lambda M, m, b: abs(M-m) / abs(b[0]-b[1])
		else:
			self.deltaF_fun = lambda M, m, b: abs(M-m) / (abs(M-m) + abs(m) + 1e-5)

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		max = np.nanmax(fitness, axis=0)
		min = np.nanmin(fitness, axis=0)
		
		return self.deltaF_fun(max, min, options.get('fitness_bound', None))
	
	def reset(self) -> None:
		return

	def get_space(self):
		return spaces.Box(low=0, high=1, shape=([]))


class IntraDeltaX(Metric):
	"""
		Intra generational deltaX.
		It requires "bounds" option in compute's arguments (so must be returned by the solver in step() method)

		intra-generation deltaX: vector of the difference between the maximum and the minimum value of the search space of the objective function for each dimension.
		max = max(X, axis=0), min = min(X, axis=0)
		dim-ΔX = abs(max - min) / bounds_range
		Intra-ΔX = [max(dim-ΔX), min(dim-ΔX)]
	"""
	name = "IntraDeltaX"
	MetricProvider.register_metric(__qualname__, __qualname__)

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		bounds = np.array(options["bounds"])
		bounds_range = np.abs(bounds.T[0] - bounds.T[1])

		max = np.nanmax(solutions, axis=0)
		min = np.nanmin(solutions, axis=0)
		
		deltaX = np.abs(max - min) / bounds_range

		return np.array([np.max(deltaX), np.min(deltaX)])
	
	def reset(self) -> None:
		return

	def get_space(self):
		return spaces.Box(low=0, high=1, shape=[2])


class InterDeltaX(Metric):
	"""
		inter generational deltaX.
		It requires "bounds" option in compute's arguments (so must be returned by the solver in step() method)

		inter-generation deltaX: vector of the difference between the two best solutions in 2 generetions.
		# best(X) = x_i ∈ X | f(x_i) = min(f(X)). (prec_best is the same for the previous generation)
		ΔX = (best(X) - prec_best(X)) / bounds_range
		Inter-ΔX = [max(ΔX), min(ΔX)]
	"""
	name = "InterDeltaX"
	MetricProvider.register_metric(__qualname__, __qualname__)

	def __init__(self) -> None:
		self.prec = None

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		best_idx = np.argmin(fitness)

		if self.prec is None:
			interDeltaX =  np.zeros(shape=[2], dtype=np.float32)
		else:
			bounds = np.array(options["bounds"])
			bounds_range = np.abs(bounds.T[0] - bounds.T[1])

			deltaX = (solutions[best_idx] - self.prec) / bounds_range
			interDeltaX = np.array([np.max(deltaX), np.min(deltaX)])

		self.prec = solutions[best_idx]
		return interDeltaX
	
	def reset(self) -> None:
		self.prec = None

	def get_space(self):
		return spaces.Box(low=-1, high=1, shape=[2])


class SolverState(Metric):
	"""
		It pass a dictionary with one or more states of the solver driver.
		It requires that the solver pass the state(s) with the same key in the options dictionary returned by step().
		The bounds must be specified with min and max in __init__.

		Configuration Example:
		...
		"state_metrics_names": ["SolverState"],
        "state_metrics_config": [
            ({"ps": {"min": -10, "max": 10}},),
        ],
		...
	"""

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


class MetricHistory(Metric):
	"""
	General metric History, last g generations of a given metric.
	The vector goes from the most recent generation to the first one.

	Args:
		metric (str): name of the metric to be stored
		metric_args (list): arguments of the metric
		history_max_length (int): maximum length of the history
	"""
	name = "MetricHistory"
	MetricProvider.register_metric(name, __qualname__)
	def __init__(self, metric: str, metric_args: list = [], history_max_length = 1) -> None:
		self.metric = eval(metric)(*metric_args)
		self.history_max_length = history_max_length
		self.history = []
	
	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		metric_value = self.metric.compute(solutions, fitness, **options)

		self.history.insert(0, metric_value)
		if len(self.history) > self.history_max_length:
			self.history.pop()

		return self.history

	def get_space(self):
		metric_space = self.metric.get_space()
		return Repeated(metric_space, max_len=self.history_max_length)

	def reset(self) -> None:
		self.history = []



# OLD DEPRECATED METRICS USED BY OLD MODELS ######################################################################

class DeltaBest(Metric):
	"""
		Deprecated by InterDeltaF
	"""
	name = "DeltaBest"
	MetricProvider.register_metric(name, __qualname__)

	def __init__(self, normalize = True, use_best_of_run = True):
		self.normalize = normalize
		if self.normalize:
			self.delta_best_function = lambda best, prec: (best - prec) / (abs(best - prec) + abs(prec) + 1e-5)
		else:
			self.delta_best_function = lambda best, prec: (best - prec) / (abs(prec) + 1e-5)
		
		self.use_best_of_run = use_best_of_run
		self.best = None
		self.prec_best = None
		self.reset()

	def get_space(self):
		if self.normalize:
			return spaces.Box(low=-1, high=1, shape=[])
		else:
			return spaces.Box(low=-np.inf, high=np.inf, shape=[]) # TODO: inf bounds reduce significantly the performance

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		curr_best_index = np.argmin(fitness, axis=0)
		curr_best_fit = fitness[curr_best_index]

		if curr_best_fit < self.best:
			self.best = curr_best_fit

		best = self.best if self.use_best_of_run else curr_best_fit

		if self.prec_best is not None:
			delta_best = self.delta_best_function(best, self.prec_best)
		else:
			delta_best = 0

		self.prec_best = best

		return delta_best

	def reset(self) -> None:
		self.best = np.inf
		self.prec_best = None

class SolverStateHistory(SolverState):
	"""
		Deprecated by MetricHistory
	"""
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

class DifferenceOfBest(Metric):
	"""
	Deprecated by InterDeltaF
	
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
	Deprecated by MetricHistory(IntraDeltaF)

	History of the normalized difference between max fitness and min fitness in the population
	"""
	name = "DeltaFitPop"
	MetricProvider.register_metric(__qualname__, __qualname__)

	def __init__(self, history_max_length=1, maximize=False):
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
	Deprecated by MetricHistory(IntraDeltaX) and MetricHistory(InterDeltaX)

	History of deltaX intra or inter generation.
	It requires "bounds" option in compute's arguments

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

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		pass # this function will be _intra_deltaX or _inter_deltaX
	
	def _intra_deltaX(self, solutions: np.array, fitness: np.array, **options):
		# bounds are function dependent and they can't be setted in init
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
			self.history.insert(0, np.zeros(shape=[2], dtype=np.float32))
			self.prec = solutions[best_idx]
		else:
			bounds = np.array(options["bounds"])
			bounds_range = np.abs(bounds.T[0] - bounds.T[1])

			deltaX = (solutions[best_idx] - self.prec) / bounds_range
			self.prec = solutions[best_idx]

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
	Deprecated and not recommended

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
	Deprecated and not recommended, consider to use it only for models trained with Guided Policy Search

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
	"""
		Deprecated and not recommended, consider to use it only for models trained with Guided Policy Search
	"""

	name = "Best"
	MetricProvider.register_metric(name, __qualname__)

	def __init__(self, maximize=False, use_best_of_run=False):
		self.maximize = maximize
		self.use_best_of_run = use_best_of_run
		self.best = None
		self.best_sol = None
		self.reset()

	def get_space(self):
		space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1))
		return space

	def compute(self, solutions: np.array, fitness: np.array, **options) -> np.array:
		indexes = np.argmax(fitness, axis=0) if self.maximize else np.argmin(fitness, axis=0)
		curr_best_index = indexes
		curr_best_fit = fitness[curr_best_index]

		if curr_best_fit < self.best:
			self.best = curr_best_fit
			self.best_sol = solutions[curr_best_index]

		return self.best if self.use_best_of_run else curr_best_fit

	def reset(self) -> None:
		self.best = np.inf
		self.best_sol = None

	def get_best(self):
		return self.best, self.best_sol

# build up MetricProvider registered metrics class types
# NB: this must be the last line of metrics.py
MetricProvider.build()

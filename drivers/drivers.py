import warnings
from abc import ABC, abstractmethod, ABCMeta
import pandas as pd
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from benchmarks.utils import loadFunction
COLORS = list(TABLEAU_COLORS.values())

DRIVERS = {}
def registerDriver(name, clazz):
	DRIVERS[name] = clazz
def buildRegister():
	for k, v in DRIVERS.items():
		if type(v) == str:
			DRIVERS[k] = eval(v)

class Driver(ABC):
	# TODO overloading of this method with online training/enforcing must also pass the parameter tuning config action
	@abstractmethod
	def step(self, command) -> Tuple[np.ndarray, np.ndarray, dict]:  # (evaluated solutions, fitness, solver params)
		pass

	@abstractmethod
	def reset(self, condition = {}) -> Tuple[np.ndarray, np.ndarray, dict]:  # (initialized solutions, fitness, solver params)
		pass

	@abstractmethod
	def initialized(self) -> bool:
		pass

	@abstractmethod
	def initialize(self) -> None:
		pass

	@abstractmethod
	def is_done(self) -> bool:
		return False


class SolverDriver(Driver):
	def __init__(self, objective_function={"id": 1, "instance": 0, "lib": "cma"}) -> None:
		self.__lines = []
		self.__add_lines = []
		self.__data = []
		self.__add_data = []
		self.np_rng = np.random
		self.seed = None
		self.obj_fun = self.__parse_obj_fun(objective_function)

	def render(self, curr_step, fitness, additional_params={}, block=False):
		max_fitness = np.max(fitness)
		min_fitness = np.min(fitness)
		median_fitness = np.median(fitness)
		average_fitness = np.mean(fitness)
		colors = ["black", "blue", "green", "red"]
		labels = ["average", "median", "max", "min"]
		if len(self.__data) < 1:
			if len(additional_params) > 0:
				self.fig, (self.ax1, self.ax2) = plt.subplots(2, sharex=True)
			else:
				self.fig, self.ax1 = plt.subplot()
			plt.ion()
			self.__data = [[curr_step], [average_fitness], [median_fitness], [max_fitness], [min_fitness]]

			if len(additional_params) > 0:
				self.__add_data = [[curr_step]]

			for k, v in additional_params.items():
				if isinstance(v, (np.ndarray, list)):
					for i in v:
						self.__add_data.append([i])
				else:
					self.__add_data.append([v])

			for i in range(4):
				(line,) = self.ax1.plot(self.__data[0], self.__data[i + 1], color=colors[i], label=labels[i])
				self.__lines.append(line)

			for i, (k, v) in enumerate(additional_params.items()):
				(add_line,) = self.ax2.plot(self.__add_data[0], self.__add_data[i + 1], color=COLORS[i], label=k)
				self.__add_lines.append(add_line)

			self.ax1.set(xlabel="Algorithm Steps", ylabel="Fitness")
		else:
			self.__data[0].append(curr_step)
			self.__data[1].append(average_fitness)
			self.__data[2].append(median_fitness)
			self.__data[3].append(max_fitness)
			self.__data[4].append(min_fitness)

			for i, line in enumerate(self.__lines):
				line.set_xdata(np.array(self.__data[0]))
				line.set_ydata(np.array(self.__data[i + 1]))

			if len(additional_params) > 0:
				self.__add_data[0].append(curr_step)

				for i, (k, v) in enumerate(additional_params.items()):
					if isinstance(v, (np.ndarray, list)):
						for j in v:
							self.__add_data[1 + i].append(j)
					else:
						self.__add_data[1 + i].append(v)

				for i, add_line in enumerate(self.__add_lines):
					add_line.set_xdata(np.array(self.__add_data[0]))
					add_line.set_ydata(np.array(self.__add_data[i + 1]))

		ymin = min([min(d) for d in self.__data[1:]])
		ymax = max([max(d) for d in self.__data[1:]])
		yrange = ymax - ymin
		self.ax1.set_xlim((0, curr_step))
		self.ax1.set_ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))

		if len(additional_params) > 0:
			ymin = min([min(d) for d in self.__add_data[1:]])
			ymax = max([max(d) for d in self.__add_data[1:]])
			yrange = ymax - ymin
			self.ax2.set_xlim((0, curr_step))
			self.ax2.set_ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))
			self.ax2.legend()

		plt.draw()
		plt.pause(0.00001)
		self.ax1.legend()
		if block:
			plt.ioff()
			plt.show()


	def __parse_obj_fun(self, obj_func):
		if isinstance(obj_func, (str, int)):
			return loadFunction(obj_func, lib="cma")
		elif isinstance(obj_func, dict):
			if not "id" in obj_func:
				raise ValueError("Objective function must have an id")
			id = obj_func.pop('id')
			lib = obj_func.pop("lib", "cma")
			return loadFunction(id, lib=lib, options=obj_func)
		else:
			return obj_func

	def reset(self):
		plt.ioff()
		if hasattr(self, 'ax1'):
			self.ax1.cla()
			plt.close()
		self.__lines = []
		self.__add_lines = []
		self.__data = []
		self.__add_data = []

	def set_seed(self, seed=None):
		self.np_rng = np.random.default_rng(seed=seed)
		self.seed = seed
		


# region Kimeme
class DriverNotReady(Exception):
	def __init__(self, *args: object) -> None:
		message = (
			"Driver uninitialized. A KimemeDriver must be explicitly initialized before use with the "
			"initialize() method."
		)
		super().__init__(message)


class KimemeFileDriverError(Exception):
	def __init__(self, *args: object) -> None:
		message = "Invalid step method for KimemeFileDriver, use next_step instead (for offline training)"
		super().__init__(message)


class KimemeFileDriver(SolverDriver, metaclass=ABCMeta):
	"""
	This class is in charge of loading a csv dump (maybe also other formats ?) of an evaluation, splitting the
		step-operator iterations and passing them to the an RL environment via the step method.
	This is one of the ways to train a RL agent on previous runs, probably not the best one.
	"""

	def __init__(self, filename, var_regex, fitness_regex, extras_columns=None, sep=";"):
		if extras_columns is None:
			extras_columns = []
		self.filename = filename
		self.var_regex = var_regex
		self.fitness_regex = fitness_regex
		self.extras_columns = extras_columns
		self.sep = sep
		self.data = None
		self.iterations = None
		self.current_iteration_index = None
		self.data_loaded = False
		self.variables = None
		self.extras = None

	def step(self, command):
		raise KimemeFileDriverError()

	# TODO (WIP) this is how we get the action: it depends on the environment, as all this logic is valid both for
	#      meme and scheduler env. extension for meme/sched/whatever-else which parses vars and extras and infer action
	@abstractmethod
	def compute_action(self, variables, next_variables, extras, next_extras):
		pass

	def unpack_solutions(self, iteration):
		solutions = self.data.loc[self.data["Iteration"] == iteration]
		variables = solutions.filter(regex=self.var_regex)
		fitness = solutions.filter(regex=self.fitness_regex)
		extras = solutions[[*self.extras_columns]]
		return variables, fitness, extras

	def next_step(self):
		"""
		Note that this should be run as an Offline Dataset (check https://docs.ray.io/en/master/rllib-offline.html)
		used for preprocessing in TRAINING, load from CSV to "batch format" json (see training_utils)
		:return:
		"""
		if not self.initialized():
			raise DriverNotReady()
		iteration = self.iterations[self.current_iteration_index]
		next_variables, next_fitness, next_extras = self.unpack_solutions(iteration)
		action = self.compute_action(self.variables, next_variables, self.extras, next_extras)
		self.current_iteration_index += 1
		return action, next_variables.to_numpy(), next_fitness.to_numpy()

	def reset(self, new_source=None, auto_reinit=False):
		self.iterations.sort()
		self.current_iteration_index = 1
		if new_source is not None:
			self.filename = new_source
			self.data_loaded = False
			if auto_reinit:
				self.initialize()
		return self.unpack_solutions(0)[:2]  # return starting situation (after DOE), exclude "extra" entry

	def initialize(self):
		self.data = pd.read_csv(self.filename, sep=self.sep)
		self.data_loaded = True
		self.iterations = self.data["Iteration"].unique()
		self.reset()

	def initialized(self):
		return self.data_loaded


class KimemeSchedulerFileDriver(KimemeFileDriver):
	# TODO need to rebuild actions on parameter changes too!
	def compute_action(self, variables, next_variables, extras, next_extras):
		operators_involved = next_extras["OperatorCode"].unique()
		if len(operators_involved) > 1:
			warnings.warn(
				"Multiple operators involved in step action, taking the most common occurrence of "
				'"OperatorCode". Consider reviewing the dataset and the corresponding project '
				"configuration."
			)
			action = next_extras["OperatorCode"].value_counts().argmax()
		else:
			action = operators_involved[0]
		return action


class KimemeMemeFileDriver(KimemeFileDriver):
	# TODO
	def compute_action(self, variables, next_variables, extras, next_extras):
		pass


# endregion

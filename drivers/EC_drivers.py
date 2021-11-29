from inspyred.ec import variators
from numpy.core.numeric import Inf
from drivers import SolverDriver, registerDriver
from abc import ABC, abstractmethod, ABCMeta
from benchmarks.utils import loadFunction
from utils.array_utils import getScalar
import numpy as np
import random
import copy
import math
import cma
import copy
import pygmo
from drivers.DE import DifferentialEvolutionSolver


class RastriginGADriver(SolverDriver, metaclass=ABCMeta):
	def __init__(self, dim, pop_dim, max_gen=50):
		self.dim = dim
		self.pop_dim = pop_dim
		self.num_selected = pop_dim
		self.pop = np.empty(0)
		self.init = False
		self.lower_bound = -5.12
		self.upper_bound = 5.12
		self.mut_rate = 0.1
		self.max_steps = max_gen
		self.curr_step = 0
		self.num_elites = 1

	def step(self, command):
		parents, _ = self.truncation_selection(self.pop, self.fitness)
		if command == 0:
			parents = self.mutation(parents)
		else:
			parents = self.crossover(parents)
		fitnessParent = self.evaluate_rastrign(parents)
		self.pop, self.fitness = self.generational_replacement(self.pop, parents, self.fitness, fitnessParent)
		self.curr_step += 1
		return self.pop, self.fitness

	def reset(self):
		self.curr_step = 0
		self.pop = self.np_rng.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.pop_dim, self.dim))
		self.fitness = self.evaluate_rastrign(self.pop)
		return self.pop, self.fitness

	def initialized(self):
		return self.init

	def initialize(self):
		self.pop = self.np_rng.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.pop_dim, self.dim))
		self.init = True

	def is_done(self):
		return self.curr_step >= self.max_steps

	def evaluate_rastrign(self, population):
		fitness = np.zeros(shape=(0,))
		for c in population:
			rastrign = sum([x ** 2 - 10 * math.cos(2 * math.pi * x) + 10 for x in c])
			fitness = np.append(fitness, [rastrign], axis=0)
		return fitness

	def truncation_selection(self, population, fitness):
		indSort = np.argsort(fitness)
		population = population[indSort]
		fitness = fitness[indSort]
		return population[: self.num_selected], fitness[: self.num_selected]

	def tournament_selection(self, population):
		tournament_size = 2
		if tournament_size > len(population):
			tournament_size = len(population)
		selected = []
		for _ in range(self.num_selected):
			tourn = random.sample(population, tournament_size)
			selected.append(max(tourn))
		return selected

	def generational_replacement(self, population, offspring, fitnessPop, fitnessOff):
		num_elites = self.num_elites
		indSort = np.argsort(fitnessPop)
		population = population[indSort]
		fitnessPop = fitnessPop[indSort]
		offspring = np.concatenate((offspring, population[:num_elites]), axis=0)
		fitnessOff = np.concatenate((fitnessOff, fitnessPop[:num_elites]), axis=0)
		# offspring.extend(population[:num_elites])
		# offspring.sort(reverse=True)
		indSort = np.argsort(fitnessOff)
		offspring = offspring[indSort]
		fitnessOff = fitnessOff[indSort]
		survivors = offspring[: len(population)]
		fitness = fitnessOff[: len(population)]
		return survivors, fitness

	def mutation(self, population):
		pop = copy.copy(population)
		for i, m in enumerate(pop):
			if random.random() < self.mut_rate:
				pop[i] += random.gauss(0, 0.1)
		return pop

	def crossover(self, population):
		if len(population) % 2 == 1:
			population = population[:-1]
		moms = population[::2]
		dads = population[1::2]
		children = []
		for i, (mom, dad) in enumerate(zip(moms, dads)):
			offspring = self.uniform_crossover(random, mom, dad)
			for o in offspring:
				children.append(o)
		return children

	def uniform_crossover(self, random, mom, dad):
		ux_bias = 0.5
		crossover_rate = 0.2
		children = []
		if random.random() < crossover_rate:
			bro = copy.copy(dad)
			sis = copy.copy(mom)
			for i, (m, d) in enumerate(zip(mom, dad)):
				if random.random() < ux_bias:
					bro[i] = m
					sis[i] = d
			children.append(bro)
			children.append(sis)
		else:
			children.append(mom)
			children.append(dad)
		return children


class CMAdriver(SolverDriver):
	def __init__(self, dim, pop_size, object_function="sphere", init_sigma=0.5, seed=None) -> None:
		super().__init__()
		super().set_seed(seed)
		self.dim = dim
		self.pop_size = pop_size
		self.obj_fun = (
			object_function
			if not isinstance(object_function, (str, int))
			else loadFunction(object_function, lib="cma")
		)
		self.curr_step = 0
		self.lower_bound = None
		self.upper_bound = None
		self.init_sigma = init_sigma
		self.chi_N = dim ** 0.5 * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))
		self.options = {
			"popsize": self.pop_size,
			"bounds": [self.lower_bound, self.upper_bound],  # in the paper here they have [None, None]
			"AdaptSigma": True,
			"verb_disp": 0,
			# "seed": self.seed,
		}
		self.reset()

	def step(self, command):
		self.es.tell(self.solutions, self.fitness)

		self.es.sigma = command["step_size"].item()

		if np.isnan(self.es.sigma):
			print("NaN step size detected!!!")

		self.solutions, self.fitness = self.es.ask_and_eval(self.obj_fun)

		# one state in "Learning Step-Size Adaptation in CMA-ES" paper
		conjugate_evolution_path = np.sqrt(np.sum(np.square(self.es.adapt_sigma.ps))) / self.chi_N - 1

		self.curr_step += 1

		return (
			self.solutions,
			self.fitness,
			{
				"step_size": np.array(self.es.sigma),
				"ps": np.array(conjugate_evolution_path),
				"es": self.es,
				"curr_step": self.curr_step,
			},
		)

	def is_done(self):
		return False

	def reset(self, condition={}):
		super().reset()
		self.set_condition(condition)
		self.curr_step = 0
		# self.solutions = self.np_rng.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.dim,))
		# self.solutions = np.random.randn(self.dim)
		self.solutions = [0] * self.dim
		self.es = cma.CMAEvolutionStrategy(self.solutions, self.init_sigma, self.options)
		self.solutions, self.fitness = self.es.ask_and_eval(self.obj_fun)
		self.es.mean_old = self.es.mean
		return (
			self.solutions,
			self.fitness,
			{"step_size": np.array(self.init_sigma, dtype=np.float32), "ps": np.array(0, dtype=np.float32), "es": self.es, "curr_step": self.curr_step},
		)

	def set_condition(self, condition):
		self.dim = condition.get("dim", self.dim)
		self.init_sigma = condition.get("init_sigma", self.init_sigma)

	def render(self, block=False):
		super().render(self.curr_step, self.fitness, {"step_size": self.es.sigma}, block)

	def initialized(self):
		return True

	def initialize(self):
		self.reset()

	def __repr__(self) -> str:
		return "CMA solver: [dim: {0}, pop_size: {1}, obj_fun: {2}, max_steps: {3}, init_sigma: {4}]".format(
			self.dim, self.pop_size, self.obj_fun.__name__, self.max_steps, self.init_sigma
		)


class DEdriver(SolverDriver):
	"""
	Differential Evolution driver

	strategy:
		"best1bin"
		"best1exp"
		"rand1exp"
		"randtobest1exp"
		"currenttobest1exp"
		"best2exp"
		"rand2exp"
		"randtobest1bin"
		"currenttobest1bin"
		"best2bin"
		"rand2bin"
		"rand1bin"
	"""

	def __init__(
		self, dim, pop_size, object_function="sphere", strategy="best1bin", sample=None, F_init=0.8, CR_init=0.7
	) -> None:
		super().__init__()
		self.dim = dim
		self.pop_size = pop_size
		self.strategy = strategy
		self.F_init = F_init
		self.CR_init = CR_init
		assert sample in [None, "normal", "uniform"]
		self.sample = sample
		self.obj_fun = (
			object_function
			if not isinstance(object_function, (str, int))
			else loadFunction(object_function, lib="cma")
		)

	def step(self, command):
		# F and CR can be both scalar or array of shape (dim,)

		self.curr_step += 1
		if self.sample:
			F, CR = self.sample_distr(command)
			self.solver.scale = F
			self.solver.cross_over_probability = CR
		else:
			self.solver.scale = command["F"]
			self.solver.cross_over_probability = command["CR"]

		if np.any(np.isnan(list(command.values()))):
			print("NaN action detected!!!")

		next(self.solver)
		self.solutions = copy.copy(self.solver.population)
		self.fitness = copy.copy(self.solver.population_energies)

		return (
			self.solutions,
			self.fitness,
			{
				"F": np.array(self.solver.scale, dtype=np.float32),
				"CR": np.array(self.solver.cross_over_probability, dtype=np.float32),
				**command
			},
		)

	def is_done(self):
		pass

	def reset(self, cond={}):
		super().reset()
		self.curr_step = 0
		self.solver = DifferentialEvolutionSolver(
			self.obj_fun,
			[[-5.12, 5.12]] * self.dim,
			mutation=self.F_init,
			strategy=self.strategy,
			recombination=self.CR_init,
			popsize=max(self.pop_size // self.dim, 1),  # population = popsize * len(x) from the doc
		)
		self.solver.dither = None
		if self.sample:
			command = {
				"F_mean": np.array([1], dtype=np.float32),
				"F_stdev": np.array([1], dtype=np.float32),
				"CR_mean": np.array([0.5], dtype=np.float32),
				"CR_stdev": np.array([1], dtype=np.float32),
			} if self.sample == "normal" else {
				"F_min": np.array([0], dtype=np.float32),
				"F_max": np.array([2], dtype=np.float32),
				"CR_min": np.array([0], dtype=np.float32),
				"CR_max": np.array([1], dtype=np.float32),
			}
			F, CR = self.sample_distr(command)
			self.solver.scale = F
			self.solver.cross_over_probability = CR
		next(self.solver)
		self.solutions = self.solver.population
		self.fitness = self.solver.population_energies
		return (
			self.solutions,
			self.fitness,
			{
				"F": np.array([self.solver.scale]),
				"CR": np.array([self.solver.cross_over_probability]),
				**command
			},
		)

	def sample_distr(self, command):
		if self.sample == "normal":
			F = np.random.normal(loc=command["F_mean"], scale=command["F_stdev"], size=self.pop_size)
			CR = np.random.normal(loc=command["CR_mean"], scale=command["CR_stdev"], size=self.pop_size)
		elif self.sample == "uniform":
			F = np.random.uniform(low=command["F_min"], high=command["F_max"], size=self.pop_size)
			CR = np.random.uniform(low=command["CR_min"], high=command["CR_max"], size=self.pop_size)
		return F, CR

	def initialized(self):
		return True

	def initialize(self):
		pass

	def render(self, block=False):
		super().render(
			self.curr_step, self.fitness, {"F": self.solver.scale, "CR": self.solver.cross_over_probability}, block
		)


registerDriver("RastriginGADriver", RastriginGADriver)
registerDriver("CMAdriver", CMAdriver)
registerDriver("DEdriver", DEdriver)

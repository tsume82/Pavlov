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
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver


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
    def __init__(self, dim, pop_size, object_function="sphere", init_sigma=0.5, max_steps=None, seed=None) -> None:
        super().__init__()
        super().set_seed(seed)
        self.dim = dim
        self.pop_size = pop_size
        self.obj_fun = (
            object_function
            if not isinstance(object_function, (str, int))
            else loadFunction(object_function, lib="cma")
        )
        self.max_steps = max_steps
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
        return False if self.max_steps == None else self.curr_step >= self.max_steps

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
            {"step_size": np.array(self.init_sigma), "ps": np.array(0), "es": self.es, "curr_step": self.curr_step},
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
        self, dim, pop_size, object_function="sphere", strategy="best1bin", F_init=0.8, CR_init=0.9
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pop_size = pop_size
        self.strategy = strategy
        self.F_init = F_init
        self.CR_init = CR_init
        self.obj_fun = (
            object_function
            if not isinstance(object_function, (str, int))
            else loadFunction(object_function, lib="cma")
        )

    def step(self, command):
        self.curr_step += 1
        self.solver.scale = command["F"].item()
        self.solver.cross_over_probability = command["CR"].item()

        next(self.solver)
        self.solutions = self.solver.population
        self.fitness = self.solver.population_energies

        return self.solutions, self.fitness, {}

    def is_done(self):
        pass

    def reset(self, cond):
        super().reset()
        self.curr_step = 0
        self.solver = DifferentialEvolutionSolver(
            self.obj_fun,
            [[-5.12, 5.12]] * self.dim,
            mutation = 0.8,
            strategy=self.strategy,
            popsize=max(self.pop_size // self.dim, 1),  # population = popsize * len(x) from the doc
        )
        next(self.solver)
        self.solutions = self.solver.population
        self.fitness = self.solver.population_energies
        return self.solutions, self.fitness, {}

    def initialized(self):
        return True

    def initialize(self):
        pass

    def render(self, block=False):
        super().render(self.curr_step, self.fitness, {"F": self.solver.scale, "CR": self.solver.cross_over_probability}, block)


registerDriver("RastriginGADriver", RastriginGADriver)
registerDriver("CMAdriver", CMAdriver)
registerDriver("DEdriver", DEdriver)

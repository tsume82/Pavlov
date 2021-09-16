from inspyred.ec import variators
from drivers import KimemeDriver
from abc import ABC, abstractmethod, ABCMeta
import numpy as np
import random
import copy
import inspyred
import math

class RastrignGADriver(KimemeDriver, metaclass=ABCMeta):
    def __init__(self, dim, pop_dim):
        self.dim = dim
        self.pop_dim = pop_dim
        self.pop = np.empty(0)
        self.init = False
        self.lower_bound = -5.12
        self.upper_bound = 5.12
        self.mut_rate = 0.1
        
    def step(self, command):
        if command == 0:
            self.pop = self.mutation(self.pop)
        else:
            self.pop = self.crossover(self.pop)
        fitness = self.evaluate_rastrign(self.pop)
        return self.pop, fitness

    def reset(self):
        self.pop = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.pop_dim, self.dim))
        fitness = self.evaluate_rastrign(self.pop)
        return self.pop, fitness

    def initialized(self):
        return self.init

    def initialize(self):
        self.pop = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.pop_dim, self.dim))
        self.init = True

    def evaluate_rastrign(self, population):
        fitness = []
        for c in population:
            fitness.append(sum([x**2 - 10 * math.cos(2 * math.pi * x) + 10 for x in c]))
        return fitness

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
            self.uniform_crossover.index = i
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
from inspyred.ec import variators
from drivers import KimemeDriver
from abc import ABC, abstractmethod, ABCMeta
import numpy as np
import random
import copy
import inspyred
import math

class RastriginGADriver(KimemeDriver, metaclass=ABCMeta):
    def __init__(self, dim, pop_dim):
        self.dim = dim
        self.pop_dim = pop_dim
        self.parent_dim = pop_dim // 2
        self.pop = np.empty(0)
        self.init = False
        self.lower_bound = -5.12
        self.upper_bound = 5.12
        self.mut_rate = 0.1
        self.max_steps = 20
        self.env_steps = 3
        self.curr_step = 0
        self.num_elites = 1
        
    def step(self, command):
        # stepPop = [], stepFit = []
        # for i in range(self.env_steps):
        parents, _ = self.truncation_selection(self.pop, self.fitness)
        # parents = self.pop
        if command == 0:
            parents = self.mutation(parents)
        else:
            parents = self.crossover(parents)
        fitnessParent = self.evaluate_rastrign(parents)
        self.pop, self.fitness = self.generational_replacement(self.pop, parents, self.fitness, fitnessParent)
        self.curr_step += 1
        # stepPop.append(self.pop)
        # stepFit.append(self.fitness)
        return self.pop, self.fitness

    def reset(self):
        self.curr_step = 0
        self.pop = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.pop_dim, self.dim))
        self.fitness = self.evaluate_rastrign(self.pop)
        return self.pop, self.fitness

    def initialized(self):
        return self.init

    def initialize(self):
        self.pop = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.pop_dim, self.dim))
        self.init = True

    def is_done(self):
        return self.curr_step >= self.max_steps

    def evaluate_rastrign(self, population):
        fitness = np.zeros(shape=(0,))
        for c in population:
            rastrign = sum([x**2 - 10 * math.cos(2 * math.pi * x) + 10 for x in c])
            fitness = np.append(fitness, [rastrign], axis=0)
        return fitness

    def truncation_selection(self, population, fitness):
        indSort = np.argsort(fitness)
        population = population[indSort]
        fitness = fitness[indSort]
        return population[:self.pop_dim], fitness[:self.pop_dim]

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
        survivors = offspring[:len(population)]
        fitness = fitnessOff[:len(population)]
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
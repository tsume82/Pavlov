import warnings
from abc import ABC, abstractmethod, ABCMeta
import pandas as pd
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
COLORS = list(TABLEAU_COLORS.values())


class Driver(ABC):
    # TODO overloading of this method with online training/enforcing must also pass the parameter tuning config action
    @abstractmethod
    def step(self, command) -> Tuple[np.ndarray, np.ndarray]: # (evaluated solutions, fitness)
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, np.ndarray]: # (initialized solutions, fitness)
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
    def __init__(self) -> None:
        self.__lines = []
        self.__data = []

    def render(self, curr_step, fitness, additional_params={}):
        max_fitness = np.max(fitness)
        min_fitness = np.min(fitness)
        median_fitness = np.median(fitness)
        average_fitness = np.mean(fitness)
        colors = ['black', 'blue', 'green', 'red']
        labels = ['average', 'median', 'max', 'min']
        for i, (k, v) in enumerate(additional_params.items()):
            colors.append(COLORS[i])
            labels.append(k)
        if len(self.__data) < 1:
            plt.figure("plot data")
            plt.ion()
            self.__data = [[curr_step], [average_fitness], [median_fitness], [max_fitness], [min_fitness]]
            for k, v in additional_params.items():
                self.__data.append([v])
            for i in range(len(self.__data) - 1):
                line, = plt.plot(self.__data[0], self.__data[i+1], color=colors[i], label=labels[i])
                self.__lines.append(line)
            plt.xlabel('Evaluations')
            plt.ylabel('Fitness')
        else:
            self.__data[0].append(curr_step)
            self.__data[1].append(average_fitness)
            self.__data[2].append(median_fitness)
            self.__data[3].append(max_fitness)
            self.__data[4].append(min_fitness)

            for i, (k, v) in enumerate(additional_params.items()):
                self.__data[5+i].append(v)

            for i, line in enumerate(self.__lines):
                line.set_xdata(np.array(self.__data[0]))
                line.set_ydata(np.array(self.__data[i+1]))

        ymin = min([min(d) for d in self.__data[1:]])
        ymax = max([max(d) for d in self.__data[1:]])
        yrange = ymax - ymin
        plt.xlim((0, curr_step))
        plt.ylim((ymin - 0.1*yrange, ymax + 0.1*yrange))
        plt.draw()
        plt.pause(0.00001)
        plt.legend()
        plt.show()
    
    def reset(self):
        plt.ioff()
        plt.figure("plot data")
        plt.cla()
        self.__lines = []
        self.__data = []


# region Kimeme
class DriverNotReady(Exception):
    def __init__(self, *args: object) -> None:
        message = "Driver uninitialized. A KimemeDriver must be explicitly initialized before use with the " \
                  "initialize() method."
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
            warnings.warn("Multiple operators involved in step action, taking the most common occurrence of "
                          "\"OperatorCode\". Consider reviewing the dataset and the corresponding project "
                          "configuration.")
            action = next_extras["OperatorCode"].value_counts().argmax()
        else:
            action = operators_involved[0]
        return action


class KimemeMemeFileDriver(KimemeFileDriver):
    # TODO
    def compute_action(self, variables, next_variables, extras, next_extras):
        pass

# endregion
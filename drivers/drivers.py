from abc import ABC, abstractmethod
import pandas as pd


class KimemeDriver(ABC):
    """
        This class is in charge of communicating with Kimeme, being the file-based CLI or the interaction via sockets
        The general interface will be defined in conjunction with the kimeme-side modules, which are yet to be designed.
        This is a draft class at the moment, but it will be needed soon for the scheduler environment
    """

    """
    def __init__(self, mode, filename=None, CLI_path=None, port=None,):
        # "" "
        :param mode: 0 for file, 1 for cli launch, 2 for interactive
        :param filename: specify the project zip or the dataset file to load
        :param CLI_path: path to kimeme CLI executable
        :param port: communication port for RPC
        # "" "
        self.mode = mode
    """

    @abstractmethod
    def step(self, command):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def initialized(self):
        pass

    @abstractmethod
    def initialize(self):
        pass


class DriverNotReady(Exception):
    def __init__(self, *args: object) -> None:
        message = "Driver uninitialized. A KimemeDriver must be explicitly initialized before use with the " \
                  "initialize() method."
        super().__init__(message)


class KimemeFileDriver(KimemeDriver):
    """
    This class is in charge of loading a csv dump (maybe also other formats ?) of an evaluation, splitting the
        step-operator iterations and passing them to the an RL environment via the step method.
    This is one of the ways to train a RL agent on previous runs, the quickest and most straight-forward compatible
        procedure
    """
    def __init__(self, filename, var_regex, fitness_regex):
        self.filename = filename
        self.var_regex = var_regex
        self.fitness_regex = fitness_regex
        self.data = None
        self.iterations = None
        self.current_iteration_index = None
        self.data_loaded = False

    def step(self, command):
        """
        :param command: IGNORED: this class should be only used for training! TODO can this work?
        Note that this should be run as an Offline Dataset (check https://docs.ray.io/en/master/rllib-offline.html)
        TODO need preprocessing in TRAINING, initialization, from CSV to "batch format" (may be tricky)
        :return:
        """
        if not self.initialized():
            raise DriverNotReady
        iteration = self.iterations[self.current_iteration_index]
        solutions = self.data.loc[self.data["Iteration"] == iteration]
        variables = solutions.filter(regex=self.var_regex)
        fitness = solutions.filter(regex=self.fitness_regex)
        self.current_iteration_index += 1
        return variables.to_numpy(), fitness.to_numpy()

    def reset(self):
        self.iterations.sort()
        self.current_iteration_index = 0

    def initialize(self):
        self.data = pd.read_csv(self.filename)
        self.data_loaded = True
        self.iterations = self.data["Iteration"].unique()
        self.reset()

    def initialized(self):
        return self.data_loaded

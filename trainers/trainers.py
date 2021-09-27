from abc import ABC, abstractmethod, ABCMeta


# TODO most of the parameters will be passed by config dict, which should be parsed at training/enforcing module level
#   according to the general plan. (exactly one step upper in the call stack)
# This class takes care of the "problem configurator" duty, together with the parameter parsing and seeing of the tools
class Trainer(ABC):
    @abstractmethod
    def train(self, config, from_file=None):
        pass

    @abstractmethod
    def test(self, config): # TODO more params? what does this need?
        pass


class SchedulerTrainer(Trainer):
    def train(self, config, from_file=None):
        pass

    def test(self, config):
        pass

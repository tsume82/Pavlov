from abc import ABC, abstractmethod
from tf_agents.agents.reinforce import *
import tensorflow as tf
from tensorforce.agents import VanillaPolicyGradient as TForceReinforce
from tf_agents.networks import actor_distribution_network

TFA_AGENTS = ["TFA_REINFORCE"]
TFORCE_AGENTS = ["TForce_REINFORCE"]
RAY_AGENTS = ["Ray_REINFORCE"]


class AgentBuilder:
    tf_agent = None
    tforce_agent = None
    ray_agent = None

    @staticmethod
    def build(config, env, optimizer):
        assert "agent.algorithm" in config.keys()
        algorithm = config["agent.algorithm"]
        if algorithm in TFA_AGENTS:
            if algorithm == "TFA_REINFORCE":
                assert "agent.algorithm.REINFORCE.fc_layer" in config.keys
                actor_net = actor_distribution_network.ActorDistributionNetwork(
                    env.observation_spec(),
                    env.action_spec(),
                    fc_layer_params=config["agent.algorithm.REINFORCE.fc_layer"]
                )

                train_step_counter = tf.compat.v2.Variable(0)
                tf_agent = reinforce_agent.ReinforceAgent(
                    env.time_step_spec(),
                    env.action_spec(),
                    actor_network=actor_net,
                    optimizer=optimizer,
                    normalize_returns=True,
                    train_step_counter=train_step_counter
                )

            tf_agent.initialize()
            # TODO self.get_policy = lambda _: self.tf_agent.policy
            # TODO step etc...
            # buildTFA function that maps steps to a uniform interface of methods

        if algorithm in TFORCE_AGENTS:
            if algorithm == "TForce_REINFORCE":
                max_episode_steps = config["agent.algorithm.TForce_REINFORCE.max_episode_steps"]
                batch_size = config["agent.algorithm.TForce_REINFORCE.batch_size"]
                tforce_agent = TForceReinforce(
                    env.states(),
                    env.actions(),
                    max_episode_steps, batch_size
                )

            tforce_agent.initialize()
            #TODO build_from_TForce()

        if algorithm in RAY_AGENTS:
            if algorithm == "Ray_REINFORCE":
                max_episode_steps = config["agent.algorithm.Ray_REINFORCE.max_episode_steps"]
                batch_size = config["agent.algorithm.Ray_REINFORCE.batch_size"]
                ray_agent = # TODO

            # TODO ray_agent.initialize()
            # TODO self.build_from_Ray()

    # meta functions:
    # "act" -> enforcing the current policy on a state of the environment
    # "train" -> use collected data to train the agent
    # also other utils function to save/load different policies
class Agent(ABC):
    @abstractmethod
    @property
    def name(self) -> str:
        pass

    @abstractmethod
    def act(self, stop_condition):
        pass

    @abstractmethod
    def train(self, stop_condition):    # TODO episode? set of episodes? (probably the second)
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def load(self, from_file):
        pass

    @abstractmethod
    def save(self, to_file):
        pass


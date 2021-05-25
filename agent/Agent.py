from tf_agents.agents.reinforce import *
import tensorflow as tf
from tensorforce.agents import VanillaPolicyGradient as TForceReinforce
from tf_agents.networks import actor_distribution_network

TFA_AGENTS = ["TFA_REINFORCE"]
TFORCE_AGENTS = ["TForce_REINFORCE"]


class Agent:
    def __init__(self, config, env, optimizer):
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
                self.tf_agent = reinforce_agent.ReinforceAgent(
                    env.time_step_spec(),
                    env.action_spec(),
                    actor_network=actor_net,
                    optimizer=optimizer,
                    normalize_returns=True,
                    train_step_counter=train_step_counter
                )

            self.tf_agent.initialize()
            self.get_policy = lambda _: self.tf_agent.policy
            # TODO step etc...
            # buildTFA function that maps steps to a uniform interface of methods

        if algorithm in TFORCE_AGENTS:
            if algorithm == "TForce_REINFORCE":
                max_episode_steps = config["agent.algorithm.TForce_REINFORCE.max_episode_steps"]
                batch_size = config["agent.algorithm.TForce_REINFORCE.batch_size"]
                self.TForce_agent = TForceReinforce(
                    env.states(),
                    env.actions(),
                    max_episode_steps, batch_size
                )

            self.TForce_agent.initialize()
            self.build_from_TForce()

    # meta functions:
    # "act" -> enforcing the current policy on a state of the environment
    # "train" -> use collected data to train the agent
    # also other utils function to save/load different policies

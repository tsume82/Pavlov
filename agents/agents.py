from abc import ABC, abstractmethod, ABCMeta
from ray.rllib.utils.annotations import override
import ray.tune

# from ray.rllib.agents import Trainer as RayTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.tune.registry import register_env

TFA_AGENTS = ["TFA_REINFORCE"]
TFORCE_AGENTS = ["TForce_REINFORCE"]
# RAY_AGENTS = ["RayPolicyGradient", "RayPGWithTeacher"]
RAY_AGENTS = {}

def registerRayAgent(name, clazz):
    RAY_AGENTS[name] = clazz

class AgentBuilder:
    tf_agent = None
    tforce_agent = None
    ray_agent = None

    @classmethod
    def build(cls, config, env=None, optimizer=None):
        assert "agent.algorithm" in config.keys()
        algorithm = config["agent.algorithm"]
        if algorithm in TFA_AGENTS:
            # import this libs only if Ray isn't used, otherwise Ray doesn't work
            import tensorflow as tf
            from tf_agents.networks import actor_distribution_network
            from tf_agents.agents.reinforce import reinforce_agent

            if algorithm == "TFA_REINFORCE":
                assert "agent.algorithm.REINFORCE.fc_layer" in config.keys
                actor_net = actor_distribution_network.ActorDistributionNetwork(
                    env.observation_spec(),
                    env.action_spec(),
                    fc_layer_params=config["agent.algorithm.REINFORCE.fc_layer"],
                )

                train_step_counter = tf.compat.v2.Variable(0)
                cls.tf_agent = reinforce_agent.ReinforceAgent(
                    env.time_step_spec(),
                    env.action_spec(),
                    actor_network=actor_net,
                    optimizer=optimizer,
                    normalize_returns=True,
                    train_step_counter=train_step_counter,
                )

            if cls.tf_agent is not None:
                cls.tf_agent.initialize()
            # TODO self.get_policy = lambda _: self.tf_agent.policy
            # TODO step etc...
            # buildTFA function that maps steps to a uniform interface of methods

        if algorithm in TFORCE_AGENTS:
            # import this libs only if Ray isn't used, otherwise Ray doesn't work
            from tensorforce.agents import VanillaPolicyGradient as TForceReinforce

            if algorithm == "TForce_REINFORCE":
                max_episode_steps = config["agent.algorithm.TForce_REINFORCE.max_episode_steps"]
                batch_size = config["agent.algorithm.TForce_REINFORCE.batch_size"]
                cls.tforce_agent = TForceReinforce(env.states(), env.actions(), max_episode_steps, batch_size)

            cls.tforce_agent.initialize()
            # TODO build_from_TForce()

        if algorithm in RAY_AGENTS:
            # e.g. algorithm == "RayPolicyGradient":
            agent_id = "agent.algorithm." + algorithm
            env_id = "env"
            agent_config = {k[len(agent_id) + 1 :]: v for k, v in config.items() if k.startswith(agent_id)}
            env_config = {k[len(env_id) + 1 :]: v for k, v in config.items() if k.startswith(env_id)}
            cls.ray_agent = eval(RAY_AGENTS[algorithm])(agent_config, env_config)

            return cls.ray_agent


# meta functions:
# "act" -> enforcing the current policy on a state of the environment
# "train" -> use collected data to train the agent
# also other utils function to save/load different policies
class Agent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def act(self, stop_condition):
        pass

    @abstractmethod
    def train(self, stop_condition):  # TODO episode? set of episodes? (probably the second)
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


class RayAgent(Agent, metaclass=ABCMeta):
    @property
    @abstractmethod
    def agent_class(self):
        pass

    """
    agent_config (dict) contains the Ray parameters needed for the trainer agent
    env_config (dict) contains env_class, plus optional env_config_args and env_config_kwargs. These correspond to env 
        class, constructor args and kwargs respectively
    """

    def __init__(self, agent_config, env_config):
        self.env_config = env_config.copy()
        assert self.env_config.get("env_class") is not None
        self.env_class = self.env_config.get("env_class")

        agent_config["env_config"] = self.env_config.get("env_config", {})
        agent_config["env"] = self.env_class.__name__

        self.config = agent_config.copy()
        self.agent = None

        self.reset()

    # The environment is itself responsible for logging the states, the only returned value is the final state
    #   (this behavior may change in the future, if needed)
    # TODO generalize steps_max to arbitrary stop_condition (working on metrics? maybe use TFA ones)
    def act(self, steps_max=None):
        # https://docs.ray.io/en/master/rllib-training.html?highlight=computing%20actions#computing-actions
        episode_reward = 0
        done = False
        obs = self.env.reset()
        steps_done = 0
        while not done and (steps_max is None or steps_done < steps_max):  # run until episode ends
            action = self.agent.compute_single_action(obs)
            obs, reward, done, info = self.env.step(action)

            episode_reward += reward
            steps_done += 1

            if self.config["render_env"]:
                self.env.render()

        return obs, episode_reward, steps_done

    def train(self, stop_condition={}, autosave=False):
        return self.agent.train()
        # return ray.tune.run(
        #     self.agent_class,
        #     config=self.config,
        #     local_dir="./.logs",
        #     stop=stop_condition,
        #     checkpoint_at_end=autosave
        # )

    def reset(self):
        ray.shutdown()
        ray.init()
        register_env(self.env_class.__name__, lambda config: self.env_class(config))
        self.env = self.env_class(self.env_config.get("env_config", {}))
        self.agent = self.agent_class(env=self.env_class.__name__, config=self.config)

    def load(self, from_file):
        self.agent.load_checkpoint(from_file)

    def save(self, to_file):
        # TODO this may raise exceptions as per documentation, keeping an eye on it
        self.agent.save_checkpoint(to_file)


class RayPolicyGradient(RayAgent):
    name = "Policy Gradient"
    agent_class = PGTrainer
    registerRayAgent(__qualname__, __qualname__)


class RayPGWithTeacher(RayPolicyGradient):
    registerRayAgent(__qualname__, __qualname__)
    """
    Ray Policy Gradient With Teacher: use a teacher for an action instead of the agent when the teacher decides it
    additional arguments:
        agent.algorithm.RayPGWithTeacher.teacher: Teacher class
        agent.algorithm.RayPGWithTeacher.teacher_config: constructor params
    """

    def __init__(self, agent_config, env_config):
        self.teacher = agent_config.pop("teacher")(*agent_config.pop("teacher_config", []))
        super().__init__(agent_config, env_config)

    def reset(self):
        ray.shutdown()
        ray.init()

        class env_class_with_teacher(self.env_class):
            def step(_self, action): # override the enviroment step to add a step with a teacher

                if self.teacher.should_act(_self.state, _self.info):
                    action = self.teacher.act(_self.state, _self.info)

                _self.state, reward, done, _self.info = super().step(action)

                return _self.state, reward, done, _self.info

            def reset(_self):
                self.teacher.reset()
                _self.info = {}
                return super().reset()

        self.env_class = env_class_with_teacher
        self.env = self.env_class(self.env_config.get("env_config", {}))
        register_env(self.env_class.__name__, lambda config: self.env_class(config))
        self.agent = self.agent_class(env=self.env_class.__name__, config=self.config)


# TODO implement PPO and other Ray-based agents

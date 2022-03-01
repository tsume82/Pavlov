from abc import ABC, abstractmethod, ABCMeta
import ray.tune
import numpy as np

# from ray.rllib.agents import Trainer as RayTrainer
from ray.rllib.agents.pg import PGTrainer
from algorithms.GuidedPolicySearch import GuidedPolicySearch
from ray.tune.registry import register_env

# import logging
# logging.basicConfig(filename="./.logs/state.log",filemode='w+',format='%(message)s',datefmt='%H:%M:%S',level=logging.DEBUG)


from environments import ENVIRONMENTS

TFA_AGENTS = ["TFA_REINFORCE"]
TFORCE_AGENTS = ["TForce_REINFORCE"]
# RAY_AGENTS = ["RayPolicyGradient", "RayPGWithTeacher"]
RAY_AGENTS = {}


def registerRayAgent(name, clazz):
	RAY_AGENTS[name] = clazz


def buildRegister():
	for k, v in RAY_AGENTS.items():
		if type(v) == str:
			RAY_AGENTS[k] = eval(v)


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
			agent_id = "agent.algorithm"
			env_id = "env"
			agent_config = {k[len(agent_id) + 1 :]: v for k, v in config.items() if k.startswith(agent_id)}
			env_config = {k[len(env_id) + 1 :]: v for k, v in config.items() if k.startswith(env_id)}
			agent_config.pop("")
			cls.ray_agent = RAY_AGENTS[algorithm](agent_config, env_config)

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
		env_class = self.env_config.get("env_class")
		self.env_class = env_class if not isinstance(env_class, str) else ENVIRONMENTS[env_class]

		agent_config["env_config"] = self.env_config.get("env_config", {})
		agent_config["env"] = self.env_class.__name__

		self.config = agent_config.copy()
		self.agent = None

		self.reset()

	def act(self, steps_max=None):
		# https://docs.ray.io/en/master/rllib-training.html?highlight=computing%20actions#computing-actions
		episode_reward = 0
		done = False
		obs = self.env.reset()
		steps_done = 0
		# RNN_state = np.zeros(shape=(2,self.config.get("model",0).get("lstm_cell_size",0)))
		while not done and (steps_max is None or steps_done < steps_max):  # run until episode ends
			action = self.agent.compute_single_action(obs)
			obs, reward, done, info = self.env.step(action)

			episode_reward += reward
			steps_done += 1

			if self.config.get("render_env", False):
				self.env.render()

		return obs, episode_reward, steps_done

	def train(self, stop_condition={}, autosave=False):
		return self.agent.train()

	def reset(self):
		ray.shutdown()
		ray.init()
		self.env = self.env_class(self.env_config.get("env_config", {}))
		register_env(self.env_class.__name__, lambda config: self.env)
		self.agent = self.agent_class(env=self.env_class.__name__, config=self.config)

	def load(self, from_file):
		self.agent.load_checkpoint(from_file)

	def save(self, to_file):
		self.agent.save_checkpoint(to_file)


class RayPolicyGradient(RayAgent):
	name = "Policy Gradient"
	agent_class = PGTrainer
	registerRayAgent(__qualname__, __qualname__)


class RayGuidedPolicySearch(RayAgent):
	name = "Ray Guided Policy Search"
	agent_class = GuidedPolicySearch
	registerRayAgent(__qualname__, __qualname__)


from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer


class RayProximalPolicyOptimization(RayAgent):
	name = "Proximal Policy Optimization"
	agent_class = PPOTrainer
	registerRayAgent(__qualname__, __qualname__)


class RayAsyncProximalPolicyOptimization(RayAgent):
	name = "Async Proximal Policy Optimization"
	agent_class = APPOTrainer
	registerRayAgent(__qualname__, __qualname__)


class RayCSA(RayAgent):
	name = "CSA"

	class CSAagent:
		def __init__(self, env=None, config=None) -> None:
			pass

		def compute_single_action(self, obs):
			f_vals = obs[0][0]  # first fitness history
			es = obs[1]["es"]  # ES object of CMA's lib
			u = es.sigma
			hsig = es.adapt_sigma.hsig(es)
			es.hsig = hsig
			delta = es.adapt_sigma.update2(es, function_values=f_vals)
			u *= delta
			return {"step_size": u}

		def train(self):
			raise NotImplementedError()

	agent_class = CSAagent
	registerRayAgent(__qualname__, __qualname__)


class DEadapt(RayAgent):
	name = "DEadapt"

	class DEadaptAgent:
		adapt_strategies = {"ide": "iDE", "jde": "jDE"}

		de_strategies = [
			"best1bin",
			"best1exp",
			"rand1exp",
			"randtobest1exp",
			# "currenttobest1exp",
			"best2exp",
			"rand2exp",
			"randtobest1bin",
			# "currenttobest1bin",
			"best2bin",
			"rand2bin",
			"rand1bin",
		]

		def __init__(self, env=None, config=None) -> None:
			self.strategy = config.get("strategy", None)
			self.pop_size = config["pop_size"]
			self.maximize = config["maximize"]
			self.adapt_strategy = config["adapt_strategy"].lower()
			self.n_dist = lambda: np.random.normal(0, 1, size=self.pop_size)
			self.u_dist = lambda: np.random.uniform(0, 1, size=self.pop_size)

			assert self.adapt_strategy in self.adapt_strategies.keys()

			self.compute_single_action = getattr(self, self.adapt_strategies[self.adapt_strategy])

			if self.adapt_strategy == "ide":
				assert self.strategy in self.de_strategies
				self.mem_CR = self.n_dist() * 0.15 + 0.5
				self.mem_F = self.n_dist() * 0.15 + 0.5
			if self.adapt_strategy == "jde":
				self.mem_CR = self.u_dist()
				self.mem_F = self.u_dist() * 0.9 + 0.1

			self.F = self.mem_F
			self.CR = self.mem_CR

			self.bestF = self.mem_F[0]
			self.bestCR = self.mem_CR[0]
			self.bestFit = -np.inf if self.maximize else np.inf

		# this function will became iDE or jDE
		def compute_single_action(self, obs):
			return {"F": 0.8, "CR": 0.9}

		def iDE(self, obs):
			fitness = obs[0][0]
			oldFitness = obs[0][1] if len(obs[0])>1 else [-np.inf]*self.pop_size
			# maintain only improved F and CR in memory
			self.mem_F = np.where(fitness < oldFitness, self.F, self.mem_F)
			self.mem_CR = np.where(fitness < oldFitness, self.CR, self.mem_CR)

			bestFit, bestF, bestCR = self.__best(fitness)

			randF = np.random.choice(self.mem_F, size=7, replace=False)
			randCR = np.random.choice(self.mem_CR, size=7, replace=False)

			if self.strategy in ["best1bin", "best1exp"]:
				self.F = bestF + self.n_dist() * 0.5 * (randF[1] - randF[2])
				self.CR = bestCR + self.n_dist() * 0.5 * (randCR[1] - randCR[2])
			if self.strategy in ["rand1bin", "rand1exp"]:
				self.F = randF[0] + self.n_dist() * 0.5 * (randF[1] - randF[2])
				self.CR = randCR[0] + self.n_dist() * 0.5 * (randCR[1] - randCR[2])
			if self.strategy in ["randtobest1bin", "randtobest1exp"]:
				self.F = self.mem_F + self.n_dist() * 0.5 * (bestF - self.mem_F) + self.n_dist() * 0.5 * (randF[0] - randF[1])
				self.CR = self.mem_CR + self.n_dist() * 0.5 * (bestCR - self.mem_CR) + self.n_dist() * 0.5 * (randCR[0] - randCR[1])
			if self.strategy in ["best2bin", "best2exp"]:
				self.F = bestF + self.n_dist() * 0.5 * (randF[0] - randF[1]) + self.n_dist() * 0.5 * (randF[2] - randF[3])
				self.CR = bestCR + self.n_dist() * 0.5 * (randCR[0] - randCR[1]) + self.n_dist() * 0.5 * (randCR[2] - randCR[3])
			if self.strategy in ["rand2bin", "rand2exp"]:
				self.F = randF[4] + self.n_dist() * 0.5 * (randF[0] - randF[1]) + self.n_dist() * 0.5 * (randF[2] - randF[3])
				self.CR = randCR[4] + self.n_dist() * 0.5 * (randCR[0] - randCR[1]) + self.n_dist() * 0.5 * (randCR[2] - randCR[3])
			# if self.strategy in ["currenttobest1bin", "currenttobest1exp"]:

			return {"F": self.F, "CR": self.CR}
			
		def jDE(self, obs):
			fitness = obs[0][0]
			oldFitness = obs[0][1] if len(obs[0])>1 else [-np.inf]*self.pop_size

			self.mem_F = np.where(fitness < oldFitness, self.F, self.mem_F)
			self.mem_CR = np.where(fitness < oldFitness, self.CR, self.mem_CR)

			prob = self.u_dist() < 0.9
			newF = self.u_dist() * 0.9 + 0.1

			self.F = np.where(prob, self.mem_F, newF)
			self.CR = np.where(prob, self.mem_CR, self.u_dist())

			return {"F": self.F, "CR": self.CR}

		def __best(self, fitness):
			# TODO check if indexes do not change
			if self.maximize:
				_max = np.max(fitness)
				if _max > self.bestFit:
					self.bestFit = _max
					max_ind = np.argmax(fitness)
					self.bestF = self.F[max_ind]
					self.bestCR = self.CR[max_ind]
				return self.bestFit, self.bestF, self.bestCR
			else:
				_min = np.min(fitness)
				if _min < self.bestFit:
					self.bestFit = _min
					min_ind = np.argmin(fitness)
					self.bestF = self.F[min_ind]
					self.bestCR = self.CR[min_ind]
				return self.bestFit, self.bestF, self.bestCR


		def train(self):
			raise NotImplementedError()

	def reset(self):
		self.env = self.env_class(self.env_config.get("env_config", {}))
		register_env(self.env_class.__name__, lambda config: self.env)
		self.agent = self.agent_class(env=self.env_class.__name__, config=self.config)

	agent_class = DEadaptAgent
	registerRayAgent(__qualname__, __qualname__)


# TODO implement other Ray-based agents
buildRegister()

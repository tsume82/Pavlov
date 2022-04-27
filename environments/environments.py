import numpy as np
import gym
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
from gym import spaces
from gym.utils import seeding
from metrics import MetricProvider
from itertools import cycle
from drivers import DRIVERS

# region: Envrionment registration region
ENVIRONMENTS = {}

def registerEnvironment(env, clazz):
	ENVIRONMENTS[env] = clazz


def buildRegister():
	for k, v in ENVIRONMENTS.items():
		if type(v) == str:
			ENVIRONMENTS[k] = eval(v)
# endregion

class InvalidEnvironmentRequest(Exception):
	def __init__(self, *args: object) -> None:
		super().__init__(*args)


class SolverEnvironment(gym.Env):
	def __init__(self) -> None:
		self.render_mode = "human"
		self.block_render_when_done = False
		self.save_trajectory = False
		self.trajectory = {}

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


class MemePolicyEnvironment(SolverEnvironment):
	"""
	Environment implementing a meme policy: given an observation of the current evolution, the population get perturbed from the agent.
	"""

	registerEnvironment(__qualname__, __qualname__)

	def __init__(
		self,
		solver_driver,
		steps,
		state_metrics_names,
		state_metrics_config,
		reward_metric,
		reward_metric_config=[],
		action_space_config={},
		obj_function=None,
		maximize=False,
		solver_driver_args=[],
	):
		super().__init__()
		self.state_metrics = MetricProvider.combine(state_metrics_names)(state_metrics_config)
		self.observation_space = self.state_metrics.get_space()

		# action space
		param_max_bounds = action_space_config.get("max", np.Inf)
		param_min_bounds = action_space_config.get("min", -np.Inf)

		self.action_space = spaces.Tuple(
			[
				spaces.Box(
					low=param_min_bounds if isinstance(param_min_bounds, (int, float)) else param_min_bounds[i],
					high=param_max_bounds if isinstance(param_max_bounds, (int, float)) else param_max_bounds[i],
					dtype=np.float32,
					shape=[action_space_config.get("dim", 2)],
				)
				for i in range(action_space_config.get("popsize", 10))
			]
		)

		# reward space, note that the reward must be one-dimensional, so an appropriate metric must be used
		self.reward_metric = MetricProvider.get_metric(reward_metric)(*reward_metric_config)

		if isinstance(solver_driver, str):
			if solver_driver in DRIVERS:
				self.solver_driver = solver_driver(*solver_driver_args)
			else:
				raise KeyError("{} driver not registered".format(solver_driver))
		else:
			self.solver_driver = solver_driver

		self.state = None
		self.driver = solver_driver
		self.maximize = maximize
		self.obj_function = obj_function
		self.steps = steps
		self.curr_step = 0

		self.reset()

	def _build_state(self, evaluated_solutions, fitness):
		return self.state_metrics.compute(evaluated_solutions, fitness)

	def reset(self):
		self.curr_step = 0
		self.reward_metric.reset()
		self.state_metrics.reset()

		if self.solver_driver is not None:
			if not self.solver_driver.initialized():
				self.solver_driver.initialize()
			self.solutions, start_fitness = self.solver_driver.reset()
		elif self.obj_function is not None:
			self.solutions = np.array(self.action_space.sample())
			start_fitness = np.array([self.obj_function(x) for x in self.solutions])
		else:
			raise ValueError("obj_function or the driver must be initialized")

		self.state = self._build_state(self.solutions, start_fitness)
		return self.state

	def step(self, action):
		self.solutions = self.solutions + action

		if self.solver_driver is not None:
			evaluated_solutions, fitness = self.solver_driver.step(self.solutions)
		elif self.obj_function is not None:
			fitness = np.array([self.obj_function(x) for x in self.solutions])
			evaluated_solutions = self.solutions
		else:
			raise ValueError("obj_function or the driver must be initialized")

		self.state = self._build_state(evaluated_solutions, fitness)
		reward = self.reward_metric.compute(evaluated_solutions, fitness)

		if not self.maximize:
			reward *= -1

		done = self.solver_driver.is_done() if self.solver_driver is not None else False
		done = done or self.curr_step >= self.steps

		self.curr_step += 1
		return self.state, reward, done, {}

	def render(self, mode="human"):
		print("Step {}: state is {}".format(self.curr_step, self.state))


class SchedulerPolicyEnvironment(SolverEnvironment):
	# TODO steps -> generic stop conditions based on metrics
	registerEnvironment(__qualname__, __qualname__)

	def __init__(
		self,
		solver_driver,
		steps,
		memes_no,
		state_metrics_names,
		state_metrics_config,
		reward_metric,
		reward_metric_config=[],
		action_space_config=None,
		solver_driver_args=[],
		maximize=False,
		conditions=[],
		**args
	):
		"""
		action space is divided in 2 parts:
				- meme to activate, Discrete space of dimension meme_no
				- parameters, Bounded Continuous/Discrete parameter which can be applied to a subset of memes.
						the idea is to limit the search space, as having many many combinations of parameters of memes which will
						not be activated would make the problem harder
				it's up the the step and the kimeme interface to apply the parameters to the correct subset of memes
		observation space: based on metrics, which are build on a list of solutions -> TODO from network table somehow?
		"""
		super().__init__()
		# observation space (state), build with a set of metrics
		self.state_metrics = MetricProvider.combine(state_metrics_names)(state_metrics_config)
		self.observation_space = self.state_metrics.get_space()

		# action space, can also include parameter tuning
		self.memes_no = memes_no
		if action_space_config is not None:
			parameter_space = {
				key: spaces.Box(low=np.array(value["min"]), high=np.array(value["max"]), dtype=np.float32)
				for (key, value) in action_space_config.items()
			}
			if memes_no > 1:
				self.action_space = spaces.Dict({"meme": spaces.Discrete(memes_no), **parameter_space})
			else:
				self.action_space = spaces.Dict({**parameter_space})
		else:
			self.action_space = spaces.Discrete(memes_no)

		# reward space, note that the reward must be one-dimensional, so an appropriate metric must be used
		self.reward_metric = MetricProvider.get_metric(reward_metric)(*reward_metric_config)

		if isinstance(solver_driver, str):
			if solver_driver in DRIVERS:
				self.solver_driver = DRIVERS[solver_driver](*solver_driver_args)
			else:
				raise KeyError("{} driver not registered".format(solver_driver))
		else:
			self.solver_driver = solver_driver

		self.state = None  # fetch from kimeme-driver in self.reset()
		self.maximize = maximize
		self.steps = steps
		self.condition_iterator = cycle(enumerate(conditions))
		self.done = False
		self.last_action = None
		self.last_reward = None
		self.curr_step = 0
		self.cumulative_reward = 0
		self.block_render_when_done = args.get("block_render_when_done", False)
		self.save_trajectory = args.get("save_trajectory", False)
		self.reset()

	def _build_state(self, evaluated_solutions, fitness, **solver_params):
		return self.state_metrics.compute(evaluated_solutions, fitness, **solver_params)

	def step(self, action):
		# this will actually launch an eventual cli or interface with kimeme via RPC, it will take time
		evaluated_solutions, self.fitness, solver_params = self.solver_driver.step(action)

		if self.maximize: # metrics minimize only
			self.fitness *= -1

		self.state = self._build_state(evaluated_solutions, self.fitness, **solver_params)
		reward = self.reward_metric.compute(evaluated_solutions, self.fitness)

		if not self.maximize:
			reward *= -1
		self.cumulative_reward += reward

		self.done = self.solver_driver.is_done() or self.curr_step >= self.steps

		self.curr_step += 1
		self.last_action = action
		self.last_reward = reward

		if self.save_trajectory:
			self.trajectory["solutions"] = np.vstack([self.trajectory["solutions"], evaluated_solutions])
			self.trajectory["fitness"] = np.vstack([self.trajectory["fitness"], self.fitness])
			self.trajectory["actions"].append(action)

		return (
			self.state,
			reward,
			self.done,
			{
				"solutions": evaluated_solutions,
				"fitness": self.fitness,
				**{**solver_params, "condition": self.cond_index},
			},
		)

	def reset(self):
		self.curr_step = 1
		self.cumulative_reward = 0
		if not self.solver_driver.initialized():
			self.solver_driver.initialize()
		self.reward_metric.reset()
		self.state_metrics.reset()

		self.cond_index, cond = next(self.condition_iterator, (0, {}))
		start_solutions, start_fitness, solver_params = self.solver_driver.reset(cond)
		self.state = self._build_state(
			start_solutions, start_fitness, **{**solver_params, "condition": self.cond_index}
		)

		if self.save_trajectory:
			self.trajectory = {
				"solutions": np.array(start_solutions),
				"fitness": np.array(start_fitness),
				"actions": [{}],
			}

		return self.state

	def render(self, mode="human"):
		if mode is None:
			return
		if mode == "terminal":
			print("                     ┌──────────┐")
			print("─────────────────────┤Step: {0}\t├─────────────────────".format(self.curr_step))
			print("                     └──────────┘")
			print("Action:\t", self.last_action)
			print("Reward:\t", self.last_reward)
			print("New State:\t", self.state)
		elif mode == "human":
			if self.done:
				best = np.max(self.fitness) if self.maximize else np.min(self.fitness)
				print("best fitness of the last population: ", best)
			self.solver_driver.render(block=self.done if self.block_render_when_done else False)


class MemePolicyRayEnvironment(MemePolicyEnvironment):
	# according to the ray doc, the env must have only one param: the env configuration (https://docs.ray.io/en/latest/rllib-env.html)
	registerEnvironment(__qualname__, __qualname__)

	def __init__(self, env_config):
		super().__init__(
			driver=env_config.get("driver", None),
			steps=env_config.get("steps"),
			state_metrics_names=env_config.get("state_metrics_names"),
			state_metrics_config=env_config.get("state_metrics_config"),
			reward_metric=env_config.get("reward_metric"),
			reward_metric_config=env_config.get("reward_metric_config", []),
			action_space_config=env_config.get("action_space_config", {}),
			obj_function=env_config.get("obj_function", None),
			maximize=env_config.get("maximize", False),
			solver_driver_args=env_config.get("solver_driver_args", []),
		)


class SchedulerPolicyRayEnvironment(SchedulerPolicyEnvironment):
	# according to the ray doc, the env must have only one param: the env configuration (https://docs.ray.io/en/latest/rllib-env.html)
	registerEnvironment(__qualname__, __qualname__)

	def __init__(self, env_config):
		super().__init__(
			solver_driver=env_config.get("solver_driver"),
			steps=env_config.get("steps"),
			memes_no=env_config.get("memes_no", 1),
			state_metrics_names=env_config.get("state_metrics_names"),
			state_metrics_config=env_config.get("state_metrics_config"),
			reward_metric=env_config.get("reward_metric"),
			reward_metric_config=env_config.get("reward_metric_config", []),
			action_space_config=env_config.get("action_space_config", None),
			maximize=env_config.get("maximize", False),
			solver_driver_args=env_config.get("solver_driver_args", []),
			conditions=env_config.get("conditions", []),
			**env_config.get("args", {})
		)


class SchedulerPolicyMultiRayEnvironment(VectorEnv):
	"""
	Vectorized Version of the SchedulerPolicyEnvironment: It allows multiple solvers acting as multiple environments for one agent.

	The number of sub environments is defined by the number of solver_driver_args. It must be a list of arguments for each solver.

	For now the solver must be the same, with the same observation, action and reward spaces. The only difference between the environments
	are the arguments of the solvers (with the purose of training a policy working with different functions/dimensions)
	"""
	registerEnvironment(__qualname__, __qualname__)

	def __init__(self, env_config):
		assert type(env_config["solver_driver_args"][0]) == list # different args for every solver
		num_envs = len(env_config["solver_driver_args"])

		self.envs = [] # sub-environments list
		for i in range(num_envs):
			curr_env_conf = env_config.copy()
			curr_env_conf["solver_driver_args"] = curr_env_conf["solver_driver_args"][i]
			self.envs.append(SchedulerPolicyRayEnvironment(curr_env_conf))
		
		# observation and action spaces of sub environments must be the same
		observation_space = self.envs[0].observation_space
		action_space = self.envs[0].action_space

		super().__init__(observation_space, action_space, num_envs)

	@override(VectorEnv)
	def vector_reset(self):
		return [e.reset() for e in self.envs]

	@override(VectorEnv)
	def reset_at(self, index):
		return self.envs[index].reset()

	@override(VectorEnv)
	def vector_step(self, actions):
		obs_batch, rew_batch, done_batch, info_batch = [], [], [], []
		for i in range(len(self.envs)):
			obs, rew, done, info = self.envs[i].step(actions[i])
			obs_batch.append(obs)
			rew_batch.append(rew)
			done_batch.append(done)
			info_batch.append(info)
		return obs_batch, rew_batch, done_batch, info_batch

	@override(VectorEnv)
	def get_unwrapped(self):
		return self.envs


buildRegister()
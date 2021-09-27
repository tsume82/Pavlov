# from tensorforce.environments import Environment as TForceEnv
from math import inf
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from metrics import *


class InvalidEnvironmentRequest(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


# TODO the obj_function must be built, interfacing with the CLI and with Kimeme runtime
class MemePolicyEnvironment_prec(gym.Env):
    def render(self, mode="human"):
        print("Step {}: state is {}".format(self.curr_step, self.state))

    def __init__(self, obj_no, H, steps, obj_function, var_boundaries, step_boundaries, start_x=None, dim=None):
        assert (type(start_x) is np.ndarray and len(start_x.shape) == 1) or (type(dim) == int and dim > 0)

        self.state = start_x
        self.dim = start_x.shape[0] if start_x is not None else dim
        self.obj_no = obj_no  # fitness length
        self.H = H  # History Length
        self.steps = steps
        self.curr_step = 0
        self.obj_function = obj_function
        self.archive = np.zeros(shape=(self.H, self.dim))
        self.archive_fitness = np.zeros(shape=(self.H, self.obj_no))

        # deltaX to next solution
        self.action_space = spaces.Box(
            low=step_boundaries[0], high=step_boundaries[1], shape=((self.dim,)), dtype=np.float32
        )

        # TODO temporary state, LTO-like, current position, current gradient + recent gradients
        self.observation_space = spaces.Box(
            low=var_boundaries[0],
            high=var_boundaries[1],
            shape=((self.dim + self.obj_no + self.obj_no * self.H,)),
            dtype=np.float32,
        )

        self.seed()
        self.reset(start_x=start_x)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _unpack_state(self):
        state_location, state_gradient, state_recent_gradient = np.split(
            self.state, (self.dim, self.dim + self.obj_no)
        )  # last is equal to self.dim+self.obj_no+self.obj_no*self.H
        state_recent_gradient.reshape(self.obj_no, self.H)
        return state_location, state_gradient, state_recent_gradient

    def _pack_state(self, state_location, state_gradient, state_recent_gradient, set_state=True):
        new_state = np.concatenate((state_location, state_gradient.flatten(), state_recent_gradient.flatten()))
        if set_state:
            self.state = new_state
        return new_state

    def _evaluate(self, ind):
        fit = self.obj_function(ind)
        self.archive[self.curr_step % self.H] = ind
        self.archive_fitness[self.curr_step % self.H] = fit
        return fit

    def reset(self, start_x=None):
        if start_x is None:
            state_location = np.random.random(size=(self.dim,))
        else:
            state_location = start_x
        self.state = np.concatenate((state_location, np.zeros(self.obj_no + self.obj_no * self.H)))
        self.curr_step = 0
        self.archive = np.zeros(shape=(self.H, self.dim))
        self.archive_fitness = np.zeros(shape=(self.H, self.obj_no))
        return self.state

    def step(self, action):
        state_location, state_gradient, state_recent_gradient = self._unpack_state()
        next_location = state_location + action
        fit = self._evaluate(next_location)
        next_state = self.build_state(next_location)
        done = self.curr_step >= self.steps
        reward = -fit  # TODO should depend on metrics
        self.curr_step += 1
        return next_state, reward, done, {}

    def compute_curr_gradient(self):
        # TODO test thoroughly
        index = self.curr_step % self.H
        oldest_index = (index + 1) % self.H
        # subtract previous fitness, along each axis
        grads = self.archive_fitness - np.roll(self.archive_fitness, 1, axis=0)
        # delete oldest entry, whose gradient is not meaningful having no predecessor in the archive
        np.delete(grads, oldest_index, 0)
        curr_grad = grads[index]
        return curr_grad, grads

    def build_state(self, next_location):
        # print(self.state)
        next_gradient, next_recent_gradient = self.compute_curr_gradient()  # TODO will depend on metrics
        # print(self.state)
        return self._pack_state(next_location, next_gradient, next_recent_gradient)


class MemePolicyEnvironment(gym.Env):
    """
    Environment implementing a meme policy: given an observation of the current evolution, the population get perturbed from the agent.
    """

    def __init__(
        self,
        driver,
        steps,
        state_metrics_names,
        state_metrics_config,
        reward_metric,
        reward_metric_config,
        action_space_config={},
        obj_function=None,
        maximize=True,
    ):

        self.state_metrics = MetricProvider.combine(state_metrics_names)(state_metrics_config)
        self.observation_space = self.state_metrics.get_space()

        # action space
        param_max_bounds = action_space_config.get("max", np.Inf)
        param_min_bounds = action_space_config.get("min", -np.Inf)

        self.action_space = spaces.Tuple([
            spaces.Box(
                low=param_min_bounds if isinstance(param_min_bounds, (int, float)) else param_min_bounds[i],
                high=param_max_bounds if isinstance(param_max_bounds, (int, float)) else param_max_bounds[i],
                dtype=np.float32,
                shape=[action_space_config.get("dim", 2)]
            )
            for i in range(action_space_config.get("popsize", 10))
        ])

        # reward space, note that the reward must be one-dimensional, so an appropriate metric must be used
        self.reward_metric = MetricProvider.get_metric(reward_metric)(*reward_metric_config)

        self.state = None
        self.driver = driver
        self.maximize = maximize
        self.obj_function = obj_function
        self.steps = steps
        self.curr_step = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _build_state(self, evaluated_solutions, fitness):
        return self.state_metrics.compute(evaluated_solutions, fitness)

    def reset(self):
        self.curr_step = 0
        self.reward_metric.reset()
        self.state_metrics.reset()

        if self.driver is not None:
            if not self.driver.initialized():
                self.driver.initialize()
            self.solutions, start_fitness = self.driver.reset()
        elif self.obj_function is not None:
            self.solutions = np.array(self.action_space.sample())
            start_fitness = np.array([self.obj_function(x) for x in self.solutions])
        else:
            raise ValueError("obj_function or the driver must be initialized")

        self.state = self._build_state(self.solutions, start_fitness)
        return self.state

    def step(self, action):
        self.solutions = self.solutions + action
        
        if self.driver is not None:
            evaluated_solutions, fitness = self.driver.step(self.solutions)
        elif self.obj_function is not None:
            fitness = np.array([self.obj_function(x) for x in self.solutions])
            evaluated_solutions = self.solutions
        else:
            raise ValueError("obj_function or the driver must be initialized")

        self.state = self._build_state(evaluated_solutions, fitness)
        reward = self.reward_metric.compute(evaluated_solutions, fitness)

        if not self.maximize:
            reward *= -1

        done = self.driver.is_done() if self.driver is not None else False
        done = done or self.curr_step >= self.steps

        self.curr_step += 1
        return self.state, reward, done, {}

    def render(self, mode="human"):
        print("Step {}: state is {}".format(self.curr_step, self.state))


class MemePolicyRayEnvironment(MemePolicyEnvironment):
    # according to the ray doc, the env must have only one param: the env configuration (https://docs.ray.io/en/latest/rllib-env.html)
    def __init__(self, env_config):
        super().__init__(
            env_config.get("driver", None),
            env_config.get("steps"),
            env_config.get("state_metrics_names"),
            env_config.get("state_metrics_config"),
            env_config.get("reward_metric"),
            env_config.get("reward_metric_config"),
            env_config.get("action_space_config", {}),
            env_config.get("obj_function", None),
            env_config.get("maximize", True),
        )


class SchedulerPolicyEnvironment(gym.Env):
    # TODO steps -> generic stop conditions based on metrics
    def __init__(
        self,
        solver_driver,
        steps,
        memes_no,
        state_metrics_names,
        state_metrics_config,
        reward_metric,
        reward_metric_config,
        parameter_tune_config=None,
        maximize=True,
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
        # observation space (state), build with a set of metrics
        self.state_metrics = MetricProvider.combine(state_metrics_names)(state_metrics_config)
        self.observation_space = self.state_metrics.get_space()

        # action space, can also include parameter tuning
        self.memes_no = memes_no
        if parameter_tune_config is not None:
            param_max_bounds = np.array([parameter_tune_config[k]["max"] for k, v in parameter_tune_config.items()])
            param_min_bounds = np.array([parameter_tune_config[k]["min"] for k, v in parameter_tune_config.items()])
            parameter_space = spaces.Box(low=param_min_bounds, high=param_max_bounds, dtype=np.float32)  # TODO dict
            self.action_space = spaces.Tuple((spaces.Discrete(memes_no), parameter_space))
        else:
            self.action_space = spaces.Discrete(memes_no)

        # reward space, note that the reward must be one-dimensional, so an appropriate metric must be used
        self.reward_metric = MetricProvider.get_metric(reward_metric)(*reward_metric_config)

        self.state = None  # fetch from kimeme-driver in self.reset()
        self.solver_driver = solver_driver
        self.maximize = maximize
        self.steps = steps
        self.last_action = None
        self.last_reward = None
        self.curr_step = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _build_state(self, evaluated_solutions, fitness):
        return self.state_metrics.compute(evaluated_solutions, fitness)

    def step(self, action):
        # this will actually launch an eventual cli or interface with kimeme via RPC, it will take time
        evaluated_solutions, fitness = self.solver_driver.step(action)
        self.state = self._build_state(evaluated_solutions, fitness)
        reward = self.reward_metric.compute(evaluated_solutions, fitness)

        if not self.maximize:
            reward *= -1

        done = self.solver_driver.is_done()
        done = done or self.curr_step >= self.steps

        self.curr_step += 1
        self.last_action = action
        self.last_reward = reward
        return self.state, reward, done, {}

    def reset(self):
        self.curr_step = 0
        if not self.solver_driver.initialized():
            self.solver_driver.initialize()
        self.reward_metric.reset()
        self.state_metrics.reset()
        start_solutions, start_fitness = self.solver_driver.reset()
        self.state = self._build_state(start_solutions, start_fitness)
        return self.state

    def render(self, mode="human"):
        print("                     ┌──────────┐")
        print("─────────────────────┤Step: {0}\t├─────────────────────".format(self.curr_step))
        print("                     └──────────┘")
        print("Action:\t", self.last_action)
        print("Reward:\t", self.last_reward)
        print("New State:\t", self.state)


class SchedulerPolicyRayEnvironment(SchedulerPolicyEnvironment):
    # according to the ray doc, the env must have only one param: the env configuration (https://docs.ray.io/en/latest/rllib-env.html)
    def __init__(self, env_config):
        super().__init__(
            solver_driver=env_config.get("solver_driver"),
            steps=env_config.get("steps"),
            memes_no=env_config.get("memes_no"),
            state_metrics_names=env_config.get("state_metrics_names"),
            state_metrics_config=env_config.get("state_metrics_config"),
            reward_metric=env_config.get("reward_metric"),
            reward_metric_config=env_config.get("reward_metric_config"),
            parameter_tune_config=env_config.get("parameter_tune_config", None),
            maximize=env_config.get("maximize", True),
        )

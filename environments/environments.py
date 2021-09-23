# from tensorforce.environments import Environment as TForceEnv
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from metrics import *


class InvalidEnvironmentRequest(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

# TODO the obj_function must be built, interfacing with the CLI and with Kimeme runtime
class MemePolicyEnvironment(gym.Env):
    def render(self, mode='human'):
        print("Step {}: state is {}".format(self.curr_step, self.state))

    def __init__(self, obj_no, H, steps, obj_function, var_boundaries, step_boundaries, start_x=None, dim=None):
        assert (type(start_x) is np.ndarray and len(start_x.shape) == 1) or (type(dim) == int and dim > 0)

        self.state = start_x
        self.dim = start_x.shape[0] if start_x is not None else dim
        self.obj_no = obj_no # fitness length
        self.H = H # History Length
        self.steps = steps
        self.curr_step = 0
        self.obj_function = obj_function
        self.archive = np.zeros(shape=(self.H, self.dim))
        self.archive_fitness = np.zeros(shape=(self.H, self.obj_no))

        # deltaX to next solution
        self.action_space = spaces.Box(
            low=step_boundaries[0],
            high=step_boundaries[1],
            shape=((self.dim,)),
            dtype=np.float32
        )

        # TODO temporary state, LTO-like, current position, current gradient + recent gradients
        self.observation_space = spaces.Box(
            low=var_boundaries[0],
            high=var_boundaries[1],
            shape=((self.dim + self.obj_no + self.obj_no*self.H,)),
            dtype=np.float32
        )

        self.seed()
        self.reset(start_x=start_x)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _unpack_state(self):
        state_location, state_gradient, state_recent_gradient = np.split(
            self.state, (self.dim, self.dim+self.obj_no))   # last is equal to self.dim+self.obj_no+self.obj_no*self.H
        state_recent_gradient.reshape(self.obj_no,self.H)
        return state_location, state_gradient, state_recent_gradient

    def _pack_state(self, state_location, state_gradient, state_recent_gradient, set_state=True):
        new_state = np.concatenate((state_location, state_gradient.flatten(), state_recent_gradient.flatten()))
        if set_state:
            self.state = new_state
        return new_state

    def _evaluate(self, ind):
        fit = self.obj_function(ind)
        self.archive[self.curr_step%self.H] = ind
        self.archive_fitness[self.curr_step%self.H] = fit
        return fit

    def reset(self, start_x=None):
        if start_x is None:
            state_location = np.random.random(size=(self.dim,))
        else:
            state_location = start_x
        self.state = np.concatenate((state_location, np.zeros(self.obj_no + self.obj_no*self.H)))
        self.curr_step = 0
        self.archive = np.zeros(shape=(self.H, self.dim))
        self.archive_fitness = np.zeros(shape=(self.H, self.obj_no))
        return self.state

    def step(self, action):
        state_location, state_gradient, state_recent_gradient = self._unpack_state()
        next_location = state_location + action
        fit = self._evaluate(next_location)
        next_state = self.build_state(next_location)
        done = (self.curr_step >= self.steps)
        reward = -fit  # TODO should depend on metrics
        self.curr_step += 1
        return next_state, reward, done, {}

    def compute_curr_gradient(self):
        # TODO test thoroughly
        index = self.curr_step % self.H
        oldest_index = (index+1) % self.H
        # subtract previous fitness, along each axis
        grads = self.archive_fitness-np.roll(self.archive_fitness, 1, axis=0)
        # delete oldest entry, whose gradient is not meaningful having no predecessor in the archive
        np.delete(grads, oldest_index, 0)
        curr_grad = grads[index]
        return curr_grad, grads

    def build_state(self, next_location):
        # print(self.state)
        next_gradient, next_recent_gradient = self.compute_curr_gradient()  # TODO will depend on metrics
        # print(self.state)
        return self._pack_state(next_location, next_gradient, next_recent_gradient)

class MemePolicyRayEnvironment(MemePolicyEnvironment):
    # according to the ray doc, the env must have only one param: the env configuration (https://docs.ray.io/en/latest/rllib-env.html)
    def __init__(self, env_config):
        obj_no = env_config.get("obj_no")
        H = env_config.get("H")
        steps = env_config.get("steps")
        obj_function = env_config.get("obj_function")
        var_boundaries = env_config.get("var_boundaries")
        step_boundaries = env_config.get("step_boundaries")
        dim = env_config.get("dim", None)
        start_x = env_config.get("start_x", None)
        super().__init__(obj_no, H, steps, obj_function, var_boundaries, step_boundaries, start_x, dim)

class SchedulerPolicyEnvironment(gym.Env):
    # TODO steps -> generic stop conditions based on metrics
    def __init__(self, kimeme_driver, steps, memes_no, state_metrics_names, space_metrics_config, reward_metric,
                 reward_metric_config, parameter_tune_config=None):
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
        self.state_metrics = MetricProvider.combine(state_metrics_names)(space_metrics_config)
        self.observation_space = self.state_metrics.get_space()

        # action space, can also include parameter tuning
        self.memes_no = memes_no
        if parameter_tune_config is not None:
            param_max_bounds = np.array([parameter_tune_config[p]["max"] for _, p in parameter_tune_config.items()])
            param_min_bounds = np.array([parameter_tune_config[p]["mix"] for _, p in parameter_tune_config.items()])
            parameter_space = spaces.Box(low=param_min_bounds, high=param_max_bounds, dtype=np.float32)
            self.action_space = spaces.Tuple((spaces.Discrete(memes_no), parameter_space))
        else:
            self.action_space = spaces.Discrete(memes_no)

        # reward space, note that the reward must be one-dimensional, so an appropriate metric must be used
        self.reward_metric = MetricProvider.get_metric(reward_metric)(*reward_metric_config)

        self.state = None  # fetch from kimeme-driver in self.reset()
        self.kimeme_driver = kimeme_driver
        self.steps = steps
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
        evaluated_solutions, fitness = self.kimeme_driver.step(action)
        self.state = self._build_state(evaluated_solutions, fitness)
        reward = self.reward_metric.compute(evaluated_solutions, fitness)

        done = self.kimeme_driver.is_done()
        done = done or self.curr_step >= self.steps

        self.curr_step += 1
        return self.state, reward, done, {}

    def reset(self):
        self.curr_step = 0
        if not self.kimeme_driver.initialized():
            self.kimeme_driver.initialize()
        self.reward_metric.reset()
        self.state_metrics.reset()
        start_solutions, start_fitness = self.kimeme_driver.reset()
        self.state = self._build_state(start_solutions, start_fitness)
        return self.state

    def render(self, mode='human'):
        pass

class SchedulerPolicyRayEnvironment(SchedulerPolicyEnvironment):
    # according to the ray doc, the env must have only one param: the env configuration (https://docs.ray.io/en/latest/rllib-env.html)
    def __init__(self, env_config):
        kimeme_driver = env_config.get("kimeme_driver")
        steps = env_config.get("steps")
        memes_no = env_config.get("memes_no")
        state_metrics_names = env_config.get("state_metrics_names")
        space_metrics_config = env_config.get("space_metrics_config")
        reward_metric = env_config.get("reward_metric")
        reward_metric_config = env_config.get("reward_metric_config")
        parameter_tune_config = env_config.get("parameter_tune_config",None)
        super().__init__(kimeme_driver, steps, memes_no, state_metrics_names, space_metrics_config, reward_metric,
                 reward_metric_config, parameter_tune_config)



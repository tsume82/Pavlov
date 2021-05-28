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

    # TODO accept variable config in input? probably not needed
    def __init__(self, obj_no, H, steps, obj_function, var_boundaries, step_boundaries, start_x=None, dim=None):
        assert (type(start_x) is np.ndarray and len(start_x.shape) == 1) or (type(dim) == int and dim > 0)

        self.state = start_x
        self.dim = start_x.shape[0] if start_x is not None else dim
        self.obj_no = obj_no
        self.H = H
        self.steps = steps
        self.curr_step = 0
        self.obj_function = obj_function
        self.archive = np.zeros(shape=(H, self.dim))
        self.archive_fitness = np.zeros(shape=(H, obj_no))

        print(step_boundaries.shape)
        print(step_boundaries[:1, :])
        print(step_boundaries[1:, :])

        # deltaX to next solution
        self.action_space = spaces.Box(
            low=step_boundaries[:1, :],
            high=step_boundaries[1:, :],
            shape=(1, self.dim),
            dtype=np.float32
        )

        print(var_boundaries.shape)
        print(var_boundaries[:1, :])
        print(var_boundaries[1:, :])

        # TODO temporary state, LTO-like, current position, current gradient + recent gradients
        self.observation_space = spaces.Box(
            low=var_boundaries[:1, :],
            high=var_boundaries[1:, :],
            shape=(1, self.dim + self.obj_no + self.obj_no*self.H),
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
        grads = self.archive_fitness-np.roll(self.archive_fitness, 1, 0)
        # delete oldest entry, whose gradient is not meaningful having no predecessor in the archive
        np.delete(grads, oldest_index, 0)
        curr_grad = grads[index]
        return curr_grad, grads

    def build_state(self, next_location):
        print(self.state)
        next_gradient, next_recent_gradient = self.compute_curr_gradient()  # TODO will depend on metrics
        print(self.state)
        return self._pack_state(next_location, next_gradient, next_recent_gradient)


class SchedulerPolicyEnvironment(gym.Env):
    # TODO steps -> generic stop conditions based on metrics
    # TODO WIP
    def __init__(self, kimeme_driver,  steps, memes_no, state_metrics_names, space_metrics_config, parameter_tune_config):
        """
        action space is divided in 2 parts:
            - meme to activate, Discrete space of dimension meme_no
            - parameters, Bounded Continuous/Discrete parameter which can be applied to a subset of memes.
                the idea is to limit the search space, as having many many combinations of parameters of memes which will
                not be activated would make the problem harder
            it's up the the step and the kimeme interface to apply the parameters to the correct subset of memes
        observation space: based on metrics, which are build on a list of solutions -> TODO from network table somehow?
        """
        space_metrics = [MetricProvider.get_metric(state_metrics_names[sm])(*space_metrics_config[sm]) for sm in range(len(state_metrics_names))]
        self.observation_space = spaces.Tuple(space_metrics)
        # TODO do something similar for reward
        self.memes_no = memes_no

        self.state = None  # TODO fetch from kimeme-driver?
        self.kimeme_driver = kimeme_driver
        self.steps = steps
        self.curr_step = 0
        self.archive = np.zeros(shape=(H, self.dim))
        self.archive_fitness = np.zeros(shape=(H, obj_no))


        self.seed()
        self.reset()

    # TODO command kimeme-interface to activate a meme
    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


"""
def environment_loader(env_name,config) -> gym.Env:
    assert type(env_name) is str
    env = None
    if env_name.startswith("gym"):
        env = gym.make(env_name)
    elif env_name.startswith("TForceMeme"):
        env = MemePolicyEnvironment(
            config[""]
        )
    # elif ():
    #    ...
    else:
        raise InvalidEnvironmentRequest()

    env.reset()
    return env
"""
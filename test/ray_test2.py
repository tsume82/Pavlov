from math import inf
from gym import spaces
import gym
import numpy as np

import ray
from ray.rllib.agents.pg import PGTrainer
from ray.tune.registry import register_env
from ray.rllib.utils.spaces.repeated import Repeated


DICT_SPACE = spaces.Dict({
    "sensors": spaces.Dict({
        "position": spaces.Box(low=-100, high=100, shape=(3, )),
        "velocity": spaces.Box(low=-1, high=1, shape=(3, )),
        "front_cam": spaces.Tuple(
            (spaces.Box(low=0, high=1, shape=(10, 10, 3)),
             spaces.Box(low=0, high=1, shape=(10, 10, 3)))),
        "rear_cam": spaces.Box(low=0, high=1, shape=(10, 10, 3)),
    }),
    "inner_state": spaces.Dict({
        "charge": spaces.Discrete(100),
        "job_status": spaces.Dict({
            "task": spaces.Discrete(5),
            "progress": spaces.Box(low=0, high=100, shape=()),
        })
    })
})

DICT_SAMPLES = [DICT_SPACE.sample() for _ in range(10)]

TUPLE_SPACE = spaces.Tuple([
    spaces.Box(low=-np.inf, high=np.inf, shape=(1, 50))
    # spaces.Tuple((spaces.Box(low=0, high=1, shape=(10, 10, 3)),
    #               spaces.Box(low=0, high=1, shape=(10, 10, 3)))),
    # spaces.Discrete(5),
])
TUPLE_SAMPLES = [TUPLE_SPACE.sample() for _ in range(10)]

# Constraints on the Repeated space.
MAX_PLAYERS = 4
MAX_ITEMS = 7
MAX_EFFECTS = 2
ITEM_SPACE = spaces.Box(-5, 5, shape=(1, ))
EFFECT_SPACE = spaces.Box(9000, 9999, shape=(4, ))
PLAYER_SPACE = spaces.Dict({
    "location": spaces.Box(-100, 100, shape=(2, )),
    "items": Repeated(ITEM_SPACE, max_len=MAX_ITEMS),
    "effects": Repeated(EFFECT_SPACE, max_len=MAX_EFFECTS),
    "status": spaces.Box(-1, 1, shape=(10, )),
})
REPEATED_SPACE = Repeated(PLAYER_SPACE, max_len=MAX_PLAYERS)
REPEATED_SAMPLES = [REPEATED_SPACE.sample() for _ in range(10)]

class NestedTupleEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = TUPLE_SPACE
        # self._spec = EnvSpec("NestedTupleEnv-v0")
        self.steps = 0

    def reset(self):
        self.steps = 0
        return TUPLE_SAMPLES[0]

    def step(self, action):
        self.steps += 1
        return TUPLE_SAMPLES[self.steps], 1, self.steps >= 5, {}

if __name__ == "__main__":
    register_env("nested", lambda _: NestedTupleEnv())
    pg = PGTrainer(
        env="nested",
        config={
            "num_workers": 0,
            # "rollout_fragment_length": 5,
            # "train_batch_size": 5,
            "model": {
                "use_lstm": False,
            },
            "framework": "tf",
        })
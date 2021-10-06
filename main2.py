from gin import config
from environments import SchedulerPolicyRayEnvironment, MemePolicyRayEnvironment
from agents import AgentBuilder
from drivers import KimemeSchedulerFileDriver, RastriginGADriver, CMAdriver, CSATeacher
import math
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from utils.plot_utils import plot_episodes
from benchmarks import *
import warnings
warnings.filterwarnings("ignore")


rl_configuration_1 = {
    "agent.algorithm": "RayPolicyGradient",
    "env.env_class": MemePolicyRayEnvironment,
    "env.env_config": {
        "steps": 10,
        "state_metrics_names": ["RecentGradients"],
        "state_metrics_config": [(10, 6, 1, None, 2)],
        "reward_metric": "Best",
        "reward_metric_config": [],
        "action_space_config": {"max": 5.12, "min": -5.12, "dim": 2, "popsize": 10},
        "obj_function": rastrigin,
        "maximize": False,
    },
}

rl_configuration_2 = {
    "agent.algorithm": "RayPolicyGradient",
    "agent.algorithm.RayPolicyGradient.framework": "tf",
    "agent.algorithm.RayPolicyGradient.model": {
        "use_lstm": True,
    },
    "env.env_class": SchedulerPolicyRayEnvironment,
    "env.env_config": {
        "solver_driver": RastriginGADriver(2, 10),
        "steps": 10,
        "memes_no": 2,
        "state_metrics_names": ["RecentGradients"],
        "state_metrics_config": [(10, 6, 1, None, 2)],
        # "space_metrics_config" : ((10, 6, 1, None, 10),),
        "reward_metric": "Best",
        "reward_metric_config": [],
        "parameter_tune_config": None,
    },
}

paper_cma_es_configuration = {
    "agent.algorithm": "RayPolicyGradient",
    "agent.algorithm.RayPolicyGradient.render_env": False,
    "agent.algorithm.RayPolicyGradient.batch_mode": "complete_episodes",
    "agent.algorithm.RayPolicyGradient.lr": 0.001,
    "agent.algorithm.RayPolicyGradient.train_batch_size": 200,
    "agent.algorithm.RayPolicyGradient.optimizer": "Adam",
    "agent.algorithm.RayPolicyGradient.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [50, 50],
    },
    "env.env_class": SchedulerPolicyRayEnvironment,
    "env.env_config": {
        "solver_driver": CMAdriver(10, 6, object_function=sphere),
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory", "SolverState"],
        "state_metrics_config": [
            (40, True),
            ({"step_size": {"max": 1, "min": 1e-10}}, 40),
            ({"ps": {"max": 10, "min": -10}},)
        ],
        "reward_metric": "Best",
        "reward_metric_config": [False],
        "memes_no": 1,
        "parameter_tune_config": {"step_size": {"max": 1, "min": 1e-10}},
    },
}

paper_cma_es_configuration_with_conditions = {
    "agent.algorithm": "RayPolicyGradient",
    "agent.algorithm.RayPolicyGradient.render_env": False,
    "agent.algorithm.RayPolicyGradient.batch_mode": "complete_episodes",
    "agent.algorithm.RayPolicyGradient.lr": 0.001,
    "agent.algorithm.RayPolicyGradient.train_batch_size": 1000,
    "agent.algorithm.RayPolicyGradient.optimizer": "Adam",
    "agent.algorithm.RayPolicyGradient.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [50, 50],
    },
    "env.env_class": SchedulerPolicyRayEnvironment,
    "env.env_config": {
        "solver_driver": CMAdriver(10, 6, object_function=rastrigin),
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory", "SolverState"],
        "state_metrics_config": [
            (40, True),
            ({"step_size": {"max": 1, "min": 1e-10}}, 40),
            ({"ps": {"max": 10, "min": -10}},)
        ],
        "reward_metric": "Best",
        "reward_metric_config": [False],
        "memes_no": 1,
        "parameter_tune_config": {"step_size": {"max": 1, "min": 1e-10}},
        "conditions":[
            {'dim': 5, 'init_sigma': 0.5},
            {'dim': 10, 'init_sigma': 0.5},
            {'dim': 15, 'init_sigma': 0.5},
            {'dim': 20, 'init_sigma': 0.5},
            {'dim': 25, 'init_sigma': 0.5},
            {'dim': 30, 'init_sigma': 0.5},
            {'dim': 5, 'init_sigma': 1.0},
            {'dim': 10, 'init_sigma': 1.0},
            {'dim': 15, 'init_sigma': 1.0},
            {'dim': 20, 'init_sigma': 1.0},
            {'dim': 25, 'init_sigma': 1.0},
            {'dim': 30, 'init_sigma': 1.0},
        ]
    },
}

paper_cma_es_config_with_cond_teacher = {
    "agent.algorithm": "RayPGWithTeacher",
    "agent.algorithm.RayPGWithTeacher.teacher": CSATeacher,
    "agent.algorithm.RayPGWithTeacher.teacher_config": {2, 1e-10}, # max, min of the action
    "agent.algorithm.RayPGWithTeacher.render_env": False,
    "agent.algorithm.RayPGWithTeacher.batch_mode": "complete_episodes",
    "agent.algorithm.RayPGWithTeacher.lr": 0.001,
    "agent.algorithm.RayPGWithTeacher.train_batch_size": 1000,
    "agent.algorithm.RayPGWithTeacher.optimizer": "Adam",
    "agent.algorithm.RayPGWithTeacher.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [50, 50],
    },
    "env.env_class": SchedulerPolicyRayEnvironment,
    "env.env_config": {
        "solver_driver": CMAdriver(10, 6, object_function=rastrigin),
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory", "SolverState"],
        "state_metrics_config": [
            (40, True),
            ({"step_size": {"max": 2, "min": 1e-10}}, 40),
            ({"ps": {"max": 10, "min": -10}},)
        ],
        "reward_metric": "Best",
        "reward_metric_config": [False],
        "memes_no": 1,
        "parameter_tune_config": {"step_size": {"max": 2, "min": 1e-10}},
        # "conditions":[
        #     {'dim': 5, 'init_sigma': 0.5},
        #     {'dim': 10, 'init_sigma': 0.5},
        #     {'dim': 15, 'init_sigma': 0.5},
        #     {'dim': 20, 'init_sigma': 0.5},
        #     {'dim': 25, 'init_sigma': 0.5},
        #     {'dim': 30, 'init_sigma': 0.5},
        #     {'dim': 5, 'init_sigma': 1.0},
        #     {'dim': 10, 'init_sigma': 1.0},
        #     {'dim': 15, 'init_sigma': 1.0},
        #     {'dim': 20, 'init_sigma': 1.0},
        #     {'dim': 25, 'init_sigma': 1.0},
        #     {'dim': 30, 'init_sigma': 1.0},
        # ]
    },
}

def main(agent_config, train=True, folder="./.checkpoints"):
    max_episodes = 12000
    # max_episodes = 1
    episodes = 0
    if train:
        agent = AgentBuilder.build(agent_config)
        # agent.load(folder+"/checkpoint-2000")
        p = plot_episodes()
        while episodes < max_episodes:
            res = agent.train()
            episodes = res["episodes_total"]
            p.plot(res["hist_stats"]["episode_reward"][: res["episodes_this_iter"]])
            # pprint(res)
            print()
            print("═════════════════════╣Ep.: {0}\t╠═════════════════════".format(episodes))
            print()
            print("Min:\t", res["episode_reward_min"])
            print("Max:\t", res["episode_reward_max"])
            print("Mean:\t", res["episode_reward_mean"])

            if episodes % 3000 == 0:
                agent.save(folder)

        p.save(folder+"/train.svg", agent_config)
        p.show()

    else:
        agent_config["env.env_config"]["args"] = {"block_render_when_done": True}
        agent_config["agent.algorithm.RayPGWithTeacher.render_env"] = True
        agent_config["env.env_config"]["conditions"] = []
        agent = AgentBuilder.build(agent_config)
        agent.load(folder+"/checkpoint-600")
        agent.act()


if __name__ == "__main__":
    main(paper_cma_es_config_with_cond_teacher, train=False, folder="./.checkpoints/CMA with teacher/")

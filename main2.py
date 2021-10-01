from gin import config
from environments import SchedulerPolicyRayEnvironment, MemePolicyRayEnvironment
from agents import AgentBuilder
from drivers import KimemeSchedulerFileDriver, RastriginGADriver, CMAdriver
import math
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from utils.plot_utils import plot_episodes


def rastrign(ind):
    return sum([x ** 2 - 10 * math.cos(2 * math.pi * x) + 10 for x in ind])


rl_configuration_1 = {
    "agent.algorithm": "Ray_PolicyGradient",
    "env.env_class": MemePolicyRayEnvironment,
    "env.env_config": {
        "steps": 10,
        "state_metrics_names": ["RecentGradients"],
        "state_metrics_config": [(10, 6, 1, None, 2)],
        "reward_metric": "Best",
        "reward_metric_config": [],
        "action_space_config": {"max": 5.12, "min": -5.12, "dim": 2, "popsize": 10},
        "obj_function": rastrign,
        "maximize": False,
    },
}

rl_configuration_2 = {
    "agent.algorithm": "Ray_PolicyGradient",
    "agent.algorithm.Ray_PolicyGradient.framework": "tf",
    "agent.algorithm.Ray_PolicyGradient.model": {
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

cma_es_configuration = {
    "agent.algorithm": "Ray_PolicyGradient",
    "agent.algorithm.Ray_PolicyGradient.render_env": False,
    "agent.algorithm.Ray_PolicyGradient.batch_mode": "complete_episodes",
    "agent.algorithm.Ray_PolicyGradient.lr": 0.002,
    "agent.algorithm.Ray_PolicyGradient.train_batch_size": 200,
    "agent.algorithm.Ray_PolicyGradient.optimizer": "Adam",
    "agent.algorithm.Ray_PolicyGradient.model": {
        "use_lstm": True,
        "lstm_cell_size": 1,
        "max_seq_len": 20,
        # "fcnet_activation": "tanh", # Supported values: "tanh", "relu", "swish" (or "silu"), "linear"
        # "fcnet_hiddens": [8],
    },
    "env.env_class": SchedulerPolicyRayEnvironment,
    "env.env_config": {
        "solver_driver": CMAdriver(10, 6),
        "maximize": False,
        "steps": 20,
        "state_metrics_names": ["DifferenceOfBest"],
        "state_metrics_config": [[]],
        "reward_metric": "Best",
        "reward_metric_config": [False],
        "memes_no": 1,
        "parameter_tune_config": {"step_size": {"max": 1, "min": 1e-10}},
    },
}

# configuration of RL CMA ES matching the paper: Learning Step-Size Adaptation in CMA-ES TODO teacher policy
# TODO "We obtain the final policy after training for 15 iterations of GPS"
paper_cma_es_configuration = {
    "agent.algorithm": "Ray_PolicyGradient",
    "agent.algorithm.Ray_PolicyGradient.render_env": True,
    "agent.algorithm.Ray_PolicyGradient.batch_mode": "complete_episodes",
    "agent.algorithm.Ray_PolicyGradient.lr": 0.001,
    "agent.algorithm.Ray_PolicyGradient.train_batch_size": 200,
    "agent.algorithm.Ray_PolicyGradient.optimizer": "Adam",
    "agent.algorithm.Ray_PolicyGradient.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [50, 50],
    },
    "env.env_class": SchedulerPolicyRayEnvironment,
    "env.env_config": {
        "solver_driver": CMAdriver(10, 6),
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest"],
        "state_metrics_config": [[40]],
        "reward_metric": "Best",
        "reward_metric_config": [False],
        "memes_no": 1,
        "parameter_tune_config": {"step_size": {"max": 1, "min": 1e-10}},
    },
}

paper_cma_es_configuration_2 = {
    "agent.algorithm": "Ray_PolicyGradient",
    "agent.algorithm.Ray_PolicyGradient.render_env": False,
    "agent.algorithm.Ray_PolicyGradient.batch_mode": "complete_episodes",
    "agent.algorithm.Ray_PolicyGradient.lr": 0.001,
    "agent.algorithm.Ray_PolicyGradient.train_batch_size": 200,
    "agent.algorithm.Ray_PolicyGradient.optimizer": "Adam",
    "agent.algorithm.Ray_PolicyGradient.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [50, 50],
    },
    "env.env_class": SchedulerPolicyRayEnvironment,
    "env.env_config": {
        "solver_driver": CMAdriver(10, 6),
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverState"],
        "state_metrics_config": [[40, False], [{"step_size": {"max": 1, "min": 1e-10}}, 40]],
        "reward_metric": "Best",
        "reward_metric_config": [False],
        "memes_no": 1,
        "parameter_tune_config": {"step_size": {"max": 1, "min": 1e-10}},
    },
}


def main(agent_config, train=True):
    if train:
        agent = AgentBuilder.build(agent_config)
        # agent.load("./.checkpoints/checkpoint-3000-CMA-paper")
        p = plot_episodes()
        for i in range(3000):
            res = agent.train()
            p.plot(res["hist_stats"]["episode_reward"][: res["episodes_this_iter"]])
            # pprint(res)
            print("                     ╔══════════╗")
            print("═════════════════════╣It.: {0}\t╠═════════════════════".format(i))
            print("                     ╚══════════╝")
            print("Number episodes:\t", res["episodes_total"])
            print("Min:\t", res["episode_reward_min"])
            print("Max:\t", res["episode_reward_max"])
            print("Mean:\t", res["episode_reward_mean"])

            if i != 0 and (i + 1) % 500 == 0:
                agent.save("./.checkpoints")

        p.save("./.plots/train_CMA.svg", agent_config)
        p.show()

    else:
        agent_config["env.env_config"]["args"] = {"block_render_when_done": True}
        agent = AgentBuilder.build(agent_config)
        agent.load("./.checkpoints/CMA 2/checkpoint-3000")
        agent.act()


if __name__ == "__main__":
    main(paper_cma_es_configuration_2, train=True)

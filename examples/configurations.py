from numpy.core.numeric import Inf
from environments import SchedulerPolicyRayEnvironment, MemePolicyRayEnvironment
from drivers import KimemeSchedulerFileDriver, RastriginGADriver, CMAdriver
from benchmarks import functions
from copy import deepcopy


def update_and_return(config, key):
    copied_conf = deepcopy(config)

    def update(d, u):  # update a single value in a nested dict
        for k, v in u.items():
            d[k] = update(d.get(k, {}), v) if isinstance(v, dict) else v
        return d

    return update(copied_conf, key)


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
        "obj_function": functions.rastrigin,
        "maximize": False,
    },
}
rl_configuration_2 = {
    "agent.algorithm": "RayPolicyGradient",
    "agent.algorithm.framework": "tf",
    "agent.algorithm.model": {
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
        "action_space_config": None,
    },
}
pg_cma_es_configuration = {
    "agent.algorithm": "RayPolicyGradient",
    "agent.algorithm.render_env": False,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 0.001,
    "agent.algorithm.train_batch_size": 200,
    "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [50, 50],
    },
    "env.env_class": SchedulerPolicyRayEnvironment,
    "env.env_config": {
        "solver_driver": CMAdriver(10, 6, object_function=functions.rastrigin),
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory", "SolverState"],
        "state_metrics_config": [
            (40, True),
            ({"step_size": {"max": 1, "min": 1e-10}}, 40),
            ({"ps": {"max": 10, "min": -10}},),
        ],
        "reward_metric": "Best",
        "reward_metric_config": [False],
        "memes_no": 1,
        "action_space_config": {"step_size": {"max": 1, "min": 1e-10}},
    },
}
ppo_cma_es_configuration = {
    "agent.algorithm": "RayProximalPolicyOptimization",
    "agent.algorithm.render_env": False,
    "agent.algorithm.num_workers": 0,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 1e-05,
    "agent.algorithm.train_batch_size": 200,
    # "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.vf_clip_param": 1000,
	# "entropy_coeff": 0.01,
	# "grad_clip": 1e5,
    "agent.algorithm.model": {"fcnet_activation": "relu", "fcnet_hiddens": [50, 50]},
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "CMAdriver",
        "solver_driver_args": [10, 10, 11, 0.38],
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory", "SolverState"],
        "state_metrics_config": [
            [40, False, 1, True, False],
            [{"step_size": {"max": 3, "min": 0}}, 40],
            [{"ps": {"max": 10, "min": -10}}],
        ],
        "reward_metric": "Best",
        "reward_metric_config": [False, True],
        "memes_no": 1,
        "action_space_config": {"step_size": {"max": 3, "min": 0.05}},
    },
}
ppo_de_configuration = {
    "agent.algorithm": "RayProximalPolicyOptimization",
    "agent.algorithm.render_env": False,
    "agent.algorithm.num_workers": 0,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 1e-05,
    "agent.algorithm.train_batch_size": 200,
    "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.vf_clip_param": 1e7,
    "agent.algorithm.model": {"fcnet_activation": "relu", "fcnet_hiddens": [50, 50]},
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "DEdriver",
        "solver_driver_args": [10, 10, 12, "best1bin"],
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory", "SolverStateHistory"],
        "state_metrics_config": [
            [40, False, 1, True, False],
            [{"F": {"max": 2, "min": 0}}, 40],
            [{"CR": {"max": 1, "min": 0}}, 40]
        ],
        "reward_metric": "Best",
        "reward_metric_config": [False, True],
        "memes_no": 1,
        "action_space_config": {"F": {"max": 2, "min": 0}, "CR": {"max": 1, "min": 0}},
    },
}
pg_configuration = {
    "agent.algorithm": "RayPolicyGradient",
    "agent.algorithm.render_env": False,
    "agent.algorithm.num_workers": 0,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 1e-4,
    "agent.algorithm.train_batch_size": 200,
    "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.model": {
        "fcnet_activation": "tanh",
        "fcnet_hiddens": [30, 30],
    },
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "CMAdriver",
        "solver_driver_args": [10, 10, 1, 1.63, None, 42],
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory"],
        "state_metrics_config": [(40, False, 1, True, False), ({"step_size": {"max": 3, "min": 0}}, 40)],
        "reward_metric": "Best",
        "reward_metric_config": [False, False],  # (maximize=True, use_best_of_run=False, fit_dim=1, fit_index=0)
        "memes_no": 1,
        "action_space_config": {"step_size": {"max": 3, "min": 1e-10}},
    },
}
appo_configuration = {
    "agent.algorithm": "RayAsyncProximalPolicyOptimization",
    "agent.algorithm.render_env": False,
    # "agent.algorithm.num_workers": 0,
    # "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 1e-5,
    "agent.algorithm.train_batch_size": 200,
    "agent.algorithm.optimizer": "Adam",
    # "agent.algorithm.vf_clip_param": 2e5,
    "agent.algorithm.model": {
        "fcnet_activation": "tanh",
        "fcnet_hiddens": [30, 30],
    },
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "CMAdriver",
        "solver_driver_args": [10, 10, 1, 1.54],
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory"],
        "state_metrics_config": [(40, False, 1, True, False), ({"step_size": {"max": 3, "min": 0}}, 40)],
        "reward_metric": "Best",
        "reward_metric_config": [False, False],  # (maximize=True, use_best_of_run=False, fit_dim=1, fit_index=0)
        "memes_no": 1,
        "action_space_config": {"step_size": {"max": 3, "min": 1e-10}},
    },
}
CSA_configuration = {
    "agent.algorithm": "RayCSA",
    "agent.algorithm.render_env": False,
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "CMAdriver",
        # functions -> [12, 11, 2, 23, 15, 8, 17, 20, 1, 16]
        # init sigmas -> [1.28, 0.38, 1.54, 1.18, 0.1, 1.66, 0.33, 0.1, 1.63, 0.1]
        "solver_driver_args": [10, 10, 23, 1.18],
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["FitnessHistory", "SolverState"],
        "state_metrics_config": [(10, 1), ({"es": {"max": Inf, "min": -Inf}},)],
        "reward_metric": "Best",
        "reward_metric_config": [False, False],  # (maximize=True, use_best_of_run=False, fit_dim=1, fit_index=0)
        "memes_no": 1,
        "action_space_config": {"step_size": {"max": 3, "min": 1e-10}},
    },
}
de_adapt_configuration = {
    "agent.algorithm": "DEadapt",
    "agent.algorithm.strategy": "best1bin",
    "agent.algorithm.adapt_strategy": "iDE",
    "agent.algorithm.pop_size": 10,
    "agent.algorithm.maximize": False,
    "agent.algorithm.render_env": False,
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "DEdriver",
        "solver_driver_args": [10, 10, 12, "best1bin"],
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["FitnessHistory"],
        "state_metrics_config": [(10, 2)],
        "reward_metric": "Best",
        "reward_metric_config": [False, True],  # (maximize=True, use_best_of_run=False, fit_dim=1, fit_index=0)
        "memes_no": 1,
        "action_space_config": {"F": {"max": [2]*10, "min": [0]*10}, "CR": {"max": [1]*10, "min": [0]*10}},
    },
}


# A way to get a list of equal configurations with some difference on some parameter
# ["BentCigar", "Discus", "Ellipsoid", "Katsuura", "Rastrigin", "Rosenbrock", "Schaffers", "Schwefel", "Sphere", "Weierstrass"]
all_ppo_configurations = [
    update_and_return(
        ppo_cma_es_configuration,
        {
            "env.env_config": {"solver_driver_args": [10, 10, fun, sigma_init]},
            "agent.algorithm.vf_clip_param": clip,
        },
    )
    for clip, fun, sigma_init in zip(
        [1e7, 10000, 2e5, 100, 100, 1e4, 10, 5000, 50, 100],
        [12, 11, 2, 23, 15, 8, 17, 20, 1, 16],
        [1.28, 0.38, 1.54, 1.18, 0.1, 1.66, 0.33, 0.1, 1.63, 0.1],
    )
]

# ["AttractiveSector", "BuecheRastrigin", "CompositeGR", "DifferentPowers", "LinearSlope", "SharpRidge", "StepEllipsoidal", "RosenbrockRotated", "SchaffersIllConditioned","LunacekBiR", "GG101me", "GG21hi"]
# [1e3, 1e4, 1e5, 10, 100, 1000, 10, 10, 100, 10, 100, 1000, 10, 50, 100, 100, 200, 500, 1e3, 5e3, 1e4, 10, 20, 50, 100, 1000, 5000, 100, 200, 1000, 100, 500, 1000, 100, 500, 1000],
# [6, 6, 6, 4, 4, 4, 19, 19, 19, 14, 14, 14, 5, 5, 5, 13, 13, 13, 7, 7, 7, 9, 9, 9, 18, 18, 18, 24, 24, 24, 21, 21, 21, 22, 22, 22],
# [5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20],
extended_ppo_configurations = [
    update_and_return(
        ppo_cma_es_configuration,
        {
            "env.env_config": {"solver_driver_args": [dim, 10, fun, 0.5]},
            "agent.algorithm.vf_clip_param": clip,
        },
    )
    for clip, fun, dim in zip(
        [1000, 100, 500, 1000],
        [21, 22, 22, 22],
        [20, 5, 10, 20],
    )
]


# dict of all configurations in this file
ALL_CONFIGURATIONS = {k: v for k, v in locals().items() if not "__" in k and isinstance(v, (dict, list))}

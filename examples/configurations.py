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
pg_cma_configuration = {
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
ppo_cma_configuration = {
    "agent.algorithm": "RayProximalPolicyOptimization",
    "agent.algorithm.render_env": False,
    "agent.algorithm.num_workers": 0,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 1e-05,
    "agent.algorithm.train_batch_size": 200,
    # "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.vf_clip_param": 10,
    # "entropy_coeff": 0.01,
    # "grad_clip": 1e5,
    "agent.algorithm.model": {"fcnet_activation": "relu", "fcnet_hiddens": [50, 50]},
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "CMAdriver",
        "solver_driver_args": [10, 10, 11, 0.5],
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory"],
        "state_metrics_config": [[40, False, 1, True, True], [{"step_size": {"max": 3, "min": 0}}, 40]],
        "reward_metric": "DeltaBest",
        "reward_metric_config": [False, True, True],
        "memes_no": 1,
        "action_space_config": {"step_size": {"max": 3, "min": 1e-5}},
    },
}
ppo_de_configuration = {
    "agent.algorithm": "RayProximalPolicyOptimization",
    "agent.algorithm.render_env": False,
    "agent.algorithm.num_workers": 0,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 5e-05,
    "agent.algorithm.train_batch_size": 200,
    "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.vf_clip_param": 10,
    "agent.algorithm.model": {"fcnet_activation": "relu", "fcnet_hiddens": [50, 50]},
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "DEdriver",
        "solver_driver_args": [10, 10, 12, "best1bin", "uniform"],
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory"],
        "state_metrics_config": [
            [40, False, 1, True, False],
            [
                {
                    "F_min": {"max": [2], "min": [0]},
                    "F_max": {"max": [2], "min": [0]},
                    "CR_min": {"max": [1], "min": [0]},
                    "CR_max": {"max": [1], "min": [0]},
                },
                40,
            ],
        ],
        "reward_metric": "DeltaBest",
        "reward_metric_config": [False, True, True],
        "memes_no": 1,
        "action_space_config": {
            "F_min": {"max": 2, "min": 0},
            "F_max": {"max": 2, "min": 0},
            "CR_min": {"max": 1, "min": 0},
            "CR_max": {"max": 1, "min": 0},
        },
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
        # (maximize=True, use_best_of_run=False, fit_dim=1, fit_index=0)
        "reward_metric_config": [False, False],
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
        # (maximize=True, use_best_of_run=False, fit_dim=1, fit_index=0)
        "reward_metric_config": [False, False],
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
        # (dim, pop_size, object_function, strategy)
        "solver_driver_args": [10, 10, 12, "best1bin"],
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["FitnessHistory"],
        "state_metrics_config": [(10, 2)],
        "reward_metric": "Best",
        # (maximize=True, use_best_of_run=False, fit_dim=1, fit_index=0)
        "reward_metric_config": [False, True],
        "memes_no": 1,
        "action_space_config": {"F": {"max": [2] * 10, "min": [0] * 10}, "CR": {"max": [1] * 10, "min": [0] * 10}},
    },
}


# region: multienv ##########################################################################


# multienv_ppo_de_configuration = {
#     "agent.algorithm": "RayProximalPolicyOptimization",
#     "agent.algorithm.render_env": False,
#     "agent.algorithm.num_workers": 0,
#     "agent.algorithm.batch_mode": "complete_episodes",
#     "agent.algorithm.lr": 5e-05,
#     "agent.algorithm.train_batch_size": 200,
#     "agent.algorithm.optimizer": "Adam",
#     "agent.algorithm.vf_clip_param": 10,
#     "agent.algorithm.model": {"fcnet_activation": "relu", "fcnet_hiddens": [50, 50]},
#     "env.env_class": "SchedulerPolicyMultiRayEnvironment",
#     "env.env_config": {
#         "solver_driver": "DEdriver",
#         "solver_driver_args": [
#             [10, 10, 12, "best1bin", "uniform"],
#             [10, 10, 11, "best1bin", "uniform"],
#             [10, 10, 2, "best1bin", "uniform"],
#             [10, 10, 23, "best1bin", "uniform"],
#             [10, 10, 15, "best1bin", "uniform"],
#             [10, 10, 8, "best1bin", "uniform"],
#             [10, 10, 17, "best1bin", "uniform"],
#             [10, 10, 20, "best1bin", "uniform"],
#             [10, 10, 1, "best1bin", "uniform"],
#             [10, 10, 16, "best1bin", "uniform"],
#         ],
#         "maximize": False,
#         "steps": 50,
#         "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory"],
#         "state_metrics_config": [
#             [40, False, 1, True, False],
#             [
#                 {
#                     "F_min": {"max": [2], "min": [0]},
#                     "F_max": {"max": [2], "min": [0]},
#                     "CR_min": {"max": [1], "min": [0]},
#                     "CR_max": {"max": [1], "min": [0]},
#                 },
#                 40,
#             ],
#         ],
#         "reward_metric": "DeltaBest",
#         "reward_metric_config": [False, True, True],
#         "memes_no": 1,
#         "action_space_config": {
#             "F_min": {"max": 2, "min": 0},
#             "F_max": {"max": 2, "min": 0},
#             "CR_min": {"max": 1, "min": 0},
#             "CR_max": {"max": 1, "min": 0},
#         },
#     },
# }

multienv_ppo_de_uniform_configuration = update_and_return(
    ppo_de_configuration, {
        "env.env_class": "SchedulerPolicyMultiRayEnvironment",
        "env.env_config": {"solver_driver_args": [[dim, 10, fun, "best1bin", "uniform"] for fun, dim in zip([
            12, 11, 2, 23, 15, 8, 17, 20, 1, 16,
            6,
            6,
            6,
            4,
            4,
            4,
            19,
            19,
            19,
            14,
            14,
            14,
            5,
            5,
            5,
            13,
            13,
            13,
            7,
            7,
            7,
            9,
            9,
            9,
            18,
            18,
            18,
            24,
            24,
            24,
            21,
            21,
            21,
            22,
            22,
            22,
        ],
            [
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
        ])]}
    }
)

multienv_ppo_de_gaussian_configuration = update_and_return(
    ppo_de_configuration, {
        "env.env_class": "SchedulerPolicyMultiRayEnvironment",
        "env.env_config": {"solver_driver_args": [[dim, 10, fun, "best1bin", "normal"] for fun, dim in zip([
            12, 11, 2, 23, 15, 8, 17, 20, 1, 16,
            6,
            6,
            6,
            4,
            4,
            4,
            19,
            19,
            19,
            14,
            14,
            14,
            5,
            5,
            5,
            13,
            13,
            13,
            7,
            7,
            7,
            9,
            9,
            9,
            18,
            18,
            18,
            24,
            24,
            24,
            21,
            21,
            21,
            22,
            22,
            22,
        ],
            [
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
        ])]}
    }
)


# endregion #################################################################################


# A way to get a list of equal configurations with some difference on some parameter
# ["BentCigar", "Discus", "Ellipsoid", "Katsuura", "Rastrigin", "Rosenbrock", "Schaffers", "Schwefel", "Sphere", "Weierstrass"]
ppo_cma_configuration_10_funcs = [
    update_and_return(
        ppo_cma_configuration,
        {
            "env.env_config": {"solver_driver_args": [10, 10, fun, 0.5]}
        },
    )
    for clip, fun, sigma_init in zip(
        [1e7, 10000, 2e5, 100, 100, 1e4, 10, 5000, 50, 100],
        [12, 11, 2, 23, 15, 8, 17, 20, 1, 16],
        [1.28, 0.38, 1.54, 1.18, 0.1, 1.66, 0.33, 0.1, 1.63, 0.1],
    )
]

de_uniform_ppo_configuration_46_funcs = [
    update_and_return(
        ppo_de_configuration,
        {
            "env.env_config": {"solver_driver_args": [dim, 10, fun, "best1bin", "uniform"]},
        },
    )
    for fun, dim in zip(
        [
            6,
            6,
            6,
            4,
            4,
            4,
            19,
            19,
            19,
            14,
            14,
            14,
            5,
            5,
            5,
            13,
            13,
            13,
            7,
            7,
            7,
            9,
            9,
            9,
            18,
            18,
            18,
            24,
            24,
            24,
            21,
            21,
            21,
            22,
            22,
            22,
        ],
        [
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
        ],
    )
]

de_gaussian_ppo_configuration_46_funcs = [
    update_and_return(
        ppo_de_configuration,
        {
            "env.env_config": {"solver_driver_args": [dim, 10, fun, "best1bin", "normal"]},
        },
    )
    for fun, dim in zip(
        [
            6,
            6,
            6,
            4,
            4,
            4,
            19,
            19,
            19,
            14,
            14,
            14,
            5,
            5,
            5,
            13,
            13,
            13,
            7,
            7,
            7,
            9,
            9,
            9,
            18,
            18,
            18,
            24,
            24,
            24,
            21,
            21,
            21,
            22,
            22,
            22,
        ],
        [
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
        ],
    )
]

# ["AttractiveSector", "BuecheRastrigin", "CompositeGR", "DifferentPowers", "LinearSlope", "SharpRidge", "StepEllipsoidal", "RosenbrockRotated", "SchaffersIllConditioned","LunacekBiR", "GG101me", "GG21hi"]
# [1e3, 1e4, 1e5, 10, 100, 1000, 10, 10, 100, 10, 100, 1000, 10, 50, 100, 100, 200, 500, 1e3, 5e3, 1e4, 10, 20, 50, 100, 1000, 5000, 100, 200, 1000, 100, 500, 1000, 100, 500, 1000],
# [6, 6, 6, 4, 4, 4, 19, 19, 19, 14, 14, 14, 5, 5, 5, 13, 13, 13, 7, 7, 7, 9, 9, 9, 18, 18, 18, 24, 24, 24, 21, 21, 21, 22, 22, 22],
# [5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20],
cma_ppo_configuration_46_funcs = [
    update_and_return(
        ppo_cma_configuration,
        {
            "env.env_config": {"solver_driver_args": [dim, 10, fun, 0.5]},
        },
    )
    for fun, dim in zip(
        [
            12, 11, 2, 23, 15, 8, 17, 20, 1, 16,
            6,
            6,
            6,
            4,
            4,
            4,
            19,
            19,
            19,
            14,
            14,
            14,
            5,
            5,
            5,
            13,
            13,
            13,
            7,
            7,
            7,
            9,
            9,
            9,
            18,
            18,
            18,
            24,
            24,
            24,
            21,
            21,
            21,
            22,
            22,
            22,
        ],
        [
            10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
            5,
            10,
            20,
        ],
    )
]

cma_ppo_configuration_46_funcs0 = cma_ppo_configuration_46_funcs[0]
cma_ppo_configuration_46_funcs1 = cma_ppo_configuration_46_funcs[1]
cma_ppo_configuration_46_funcs2 = cma_ppo_configuration_46_funcs[2]
cma_ppo_configuration_46_funcs3 = cma_ppo_configuration_46_funcs[3]
cma_ppo_configuration_46_funcs4 = cma_ppo_configuration_46_funcs[4]
cma_ppo_configuration_46_funcs5 = cma_ppo_configuration_46_funcs[5]
cma_ppo_configuration_46_funcs6 = cma_ppo_configuration_46_funcs[6]
cma_ppo_configuration_46_funcs7 = cma_ppo_configuration_46_funcs[7]
cma_ppo_configuration_46_funcs8 = cma_ppo_configuration_46_funcs[8]
cma_ppo_configuration_46_funcs9 = cma_ppo_configuration_46_funcs[9]
cma_ppo_configuration_46_funcs10 = cma_ppo_configuration_46_funcs[10]
cma_ppo_configuration_46_funcs11 = cma_ppo_configuration_46_funcs[11]
cma_ppo_configuration_46_funcs12 = cma_ppo_configuration_46_funcs[12]
cma_ppo_configuration_46_funcs13 = cma_ppo_configuration_46_funcs[13]
cma_ppo_configuration_46_funcs14 = cma_ppo_configuration_46_funcs[14]
cma_ppo_configuration_46_funcs15 = cma_ppo_configuration_46_funcs[15]
cma_ppo_configuration_46_funcs16 = cma_ppo_configuration_46_funcs[16]
cma_ppo_configuration_46_funcs17 = cma_ppo_configuration_46_funcs[17]
cma_ppo_configuration_46_funcs18 = cma_ppo_configuration_46_funcs[18]
cma_ppo_configuration_46_funcs19 = cma_ppo_configuration_46_funcs[19]
cma_ppo_configuration_46_funcs20 = cma_ppo_configuration_46_funcs[20]
cma_ppo_configuration_46_funcs21 = cma_ppo_configuration_46_funcs[21]
cma_ppo_configuration_46_funcs22 = cma_ppo_configuration_46_funcs[22]
cma_ppo_configuration_46_funcs23 = cma_ppo_configuration_46_funcs[23]
cma_ppo_configuration_46_funcs24 = cma_ppo_configuration_46_funcs[24]
cma_ppo_configuration_46_funcs25 = cma_ppo_configuration_46_funcs[25]
cma_ppo_configuration_46_funcs26 = cma_ppo_configuration_46_funcs[26]
cma_ppo_configuration_46_funcs27 = cma_ppo_configuration_46_funcs[27]
cma_ppo_configuration_46_funcs28 = cma_ppo_configuration_46_funcs[28]
cma_ppo_configuration_46_funcs29 = cma_ppo_configuration_46_funcs[29]
cma_ppo_configuration_46_funcs30 = cma_ppo_configuration_46_funcs[30]
cma_ppo_configuration_46_funcs31 = cma_ppo_configuration_46_funcs[31]
cma_ppo_configuration_46_funcs32 = cma_ppo_configuration_46_funcs[32]
cma_ppo_configuration_46_funcs33 = cma_ppo_configuration_46_funcs[33]
cma_ppo_configuration_46_funcs34 = cma_ppo_configuration_46_funcs[34]
cma_ppo_configuration_46_funcs35 = cma_ppo_configuration_46_funcs[35]
cma_ppo_configuration_46_funcs36 = cma_ppo_configuration_46_funcs[36]
cma_ppo_configuration_46_funcs37 = cma_ppo_configuration_46_funcs[37]
cma_ppo_configuration_46_funcs38 = cma_ppo_configuration_46_funcs[38]
cma_ppo_configuration_46_funcs39 = cma_ppo_configuration_46_funcs[39]
cma_ppo_configuration_46_funcs40 = cma_ppo_configuration_46_funcs[40]
cma_ppo_configuration_46_funcs41 = cma_ppo_configuration_46_funcs[41]
cma_ppo_configuration_46_funcs42 = cma_ppo_configuration_46_funcs[42]
cma_ppo_configuration_46_funcs43 = cma_ppo_configuration_46_funcs[43]
cma_ppo_configuration_46_funcs44 = cma_ppo_configuration_46_funcs[44]
cma_ppo_configuration_46_funcs45 = cma_ppo_configuration_46_funcs[45]

# dict of all configurations in this file
ALL_CONFIGURATIONS = {k: v for k, v in locals().items(
) if not "__" in k and isinstance(v, (dict, list))}

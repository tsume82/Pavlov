from environments import SchedulerPolicyRayEnvironment, MemePolicyRayEnvironment
from drivers import KimemeSchedulerFileDriver, RastriginGADriver, CMAdriver, CSATeacher
from benchmarks import CEC2017, functions
# COCO example usage: CMAdriver(10, 6, object_function=lambda x: COCO.bbob[0](x))
# COCO objects aren't serializable

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
        "parameter_tune_config": None,
    },
}
# Bad for rastrigin
paper_cma_es_configuration = {
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
            ({"ps": {"max": 10, "min": -10}},)
        ],
        "reward_metric": "Best",
        "reward_metric_config": [False],
        "memes_no": 1,
        "parameter_tune_config": {"step_size": {"max": 1, "min": 1e-10}},
    },
}
paper_cma_es_configuration_2 = {
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
        "state_metrics_names": ["DifferenceOfBest"],
        "state_metrics_config": [
            (40, True)
        ],
        "reward_metric": "Best",
        "reward_metric_config": [False],
        "memes_no": 1,
        "parameter_tune_config": {"step_size": {"max": 1, "min": 1e-10}},
    },
}
paper_cma_es_configuration_with_conditions = {
    "agent.algorithm": "RayPolicyGradient",
    "agent.algorithm.render_env": False,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 0.001,
    "agent.algorithm.train_batch_size": 1000,
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

# No paper based
ppo_configuration = {
	"agent.algorithm": "RayProximalPolicyOptimization",
    "agent.algorithm.render_env": False,
    "agent.algorithm.batch_mode": "complete_episodes",
    # "agent.algorithm.lr": 1e-7,
    "agent.algorithm.train_batch_size": 200,
    "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.vf_clip_param": 500,
    "agent.algorithm.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [20, 20],
    },
    "env.env_class": SchedulerPolicyRayEnvironment,
    "env.env_config": {
        "solver_driver": CMAdriver(10, 10, object_function=functions.rastrigin),
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest",'BestsHistory'],
        "state_metrics_config": [
            (40, False),
            (40, False, {"max": 300,"min": 0})
        ],
        "reward_metric": "Best",
        "reward_metric_config": [False, False], # (maximize=True, use_best_of_run=False, fit_dim=1, fit_index=0)
        "memes_no": 1,
        "parameter_tune_config": {"step_size": {"max": 3, "min": 1e-10}},
    },
}
pg_configuration = {
	"agent.algorithm": "RayPolicyGradient",
    "agent.algorithm.render_env": False,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 0.001,
    "agent.algorithm.train_batch_size": 1000,
    "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.vf_clip_param": 1000,
    "agent.algorithm.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [50, 50],
    },
    "env.env_class": SchedulerPolicyRayEnvironment,
    "env.env_config": {
        "solver_driver": CMAdriver(10, 6, object_function=functions.rastrigin),
        "maximize": False,
        "steps": 50,
        "state_metrics_names": ["DifferenceOfBest", "SolverStateHistory"],
        "state_metrics_config": [
            (40, False),
        ],
        "reward_metric": "Best",
        "reward_metric_config": [False, False], # (maximize=True, use_best_of_run=False, fit_dim=1, fit_index=0)
        "memes_no": 1,
        "parameter_tune_config": {"step_size": {"max": 3, "min": 1e-10}},
    },
}
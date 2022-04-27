import sys
sys.path.insert(1, "./") # can be run with terminal
from os.path import isdir, isfile
from main import main

def test_train_single_function(num_dir = 1):
	test_cma_single_function =  {
		"agent.algorithm": "RayProximalPolicyOptimization",
		"agent.algorithm.render_env": False,
		"agent.algorithm.num_workers": 0,
		"agent.algorithm.batch_mode": "complete_episodes",
		"agent.algorithm.lr": 1e-05,
		"agent.algorithm.train_batch_size": 200,
		"agent.algorithm.optimizer": "Adam",
		"agent.algorithm.model": {"fcnet_activation": "relu", "fcnet_hiddens": [50, 50]},
		"env.env_class": "SchedulerPolicyRayEnvironment",
		"env.env_config": {
			"solver_driver": "CMAdriver",
			"solver_driver_args": [10, 10, 11, 0.5],
			"maximize": False,
			"steps": 50,
			"state_metrics_names": ["MetricHistory"],
			#, "MetricHistory", "MetricHistory", "MetricHistory", "SolverStateHistory"],
			"state_metrics_config": [
				# ["DeltaBest", [True, True], 40], 
				# ["IntraDeltaF", [], 40], 
				["IntraDeltaX", [], 40], 
				# ["InterDeltaX", [], 40], 
				# [{"step_size": {"max": 3, "min": 0}}, 40]
				],
			"reward_metric": "DeltaBest",
			"reward_metric_config": [True, True],
			"memes_no": 1,
			"action_space_config": {"step_size": {"max": 3, "min": 1e-5}},
		},
	}

	kwards = {
		"create_dir" : f"generated_dir{num_dir}",
		"max_episodes" : 100,
		"episodes_to_checkpoint" : 100
	}

	main(test_cma_single_function, True, "./test", **kwards)

	assert isfile(f"./test/generated_dir{num_dir}/config.json")
	assert isfile(f"./test/generated_dir{num_dir}/model_weights-25")
	assert isfile(f"./test/generated_dir{num_dir}/train.svg")


if __name__ == "__main__":
	test_train_single_function(1)
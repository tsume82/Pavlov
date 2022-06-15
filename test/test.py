import sys, os
sys.path.insert(1, "./") # can be run with terminal
from os.path import isdir, isfile, join
from main import main
import traceback

TEST_DIR = "test"
# GENERATED_DIR = join(TEST_DIR, "generated_dir")

def train_single_fun_cma(num_test = 1):
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
			"solver_driver_args": [10, 10, 11, 0.5, [-5.12, 5.12]],
			"maximize": False,
			"steps": 50,
			"state_metrics_names": ["MetricHistory", "MetricHistory", "MetricHistory", "MetricHistory", "SolverStateHistory"],
			"state_metrics_config": [
				["IntraDeltaF", [], 40], 
				["InterDeltaF", [], 40], 
				["IntraDeltaX", [], 40], 
				["InterDeltaX", [], 40], 
				[{"step_size": {"max": 3, "min": 0}}, 40]
				],
			"reward_metric": "DeltaBest",
			"reward_metric_config": [True, True],
			"memes_no": 1,
			"action_space_config": {"step_size": {"max": 3, "min": 1e-5}},
		},
	}

	kwards = {
		"create_dir" : f"generated_dir{num_test}",
		"max_episodes" : 100,
		"episodes_to_checkpoint" : 100,
		# "verbose": 0
	}

	gen_dir = join(TEST_DIR, kwards["create_dir"])
	
	main(test_cma_single_function, True, "./test", **kwards)
	
	assert isfile(f"{gen_dir}/config.json"), "configuration file not found"
	assert isfile(f"{gen_dir}/model_weights-25"), "model wheights not found"
	assert isfile(f"{gen_dir}/train.svg"), "train.svg not found"

def train_single_fun_de(num_test = 2):
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
			"solver_driver": "DEdriver",
			"solver_driver_args": [10, 10, {"id": 7, "instance": 3}, "best1bin", "uniform"],
			"maximize": False,
			"steps": 50,
			"state_metrics_names": ["MetricHistory", "MetricHistory", "MetricHistory", "MetricHistory", "SolverStateHistory"],
			"state_metrics_config": [
				["IntraDeltaF", [], 40], 
				["InterDeltaF", [], 40], 
				["IntraDeltaX", [], 40], 
				["InterDeltaX", [], 40], 
				[
					{"F_min": {"max": [2], "min": [0]},
					"F_max": {"max": [2], "min": [0]},
					"CR_min": {"max": [1], "min": [0]},
					"CR_max": {"max": [1], "min": [0]},},
					40
				]
			],
			"reward_metric": "DeltaBest",
			"reward_metric_config": [True, True],
			"memes_no": 1,
			"action_space_config": {
				"F_min": {"max": 2, "min": 0},
				"F_max": {"max": 2, "min": 0},
				"CR_min": {"max": 1, "min": 0},
				"CR_max": {"max": 1, "min": 0},
			},
		},
	}

	kwards = {
		"create_dir" : f"generated_dir{num_test}",
		"max_episodes" : 100,
		"episodes_to_checkpoint" : 100,
		# "verbose": 0
	}

	gen_dir = join(TEST_DIR, kwards["create_dir"])
	
	main(test_cma_single_function, True, "./test", **kwards)
	
	assert isfile(f"{gen_dir}/config.json"), "configuration file not found"
	assert isfile(f"{gen_dir}/model_weights-25"), "model wheights not found"
	assert isfile(f"{gen_dir}/train.svg"), "train.svg not found"


if __name__ == "__main__":
	# delete generated directories in ./test/generated_dir... before execute

	tests = [
		train_single_fun_cma, 
		train_single_fun_de
	]
	for i, test in enumerate(tests):
		num = i + 1
		try:
			# print(f"\033[92m{num}...", end="\r")
			test(num)
			print(f"\033[92m{num}) Passed!", end="\033[0m\n")
		except Exception:
			print("\033[91m" + str(traceback.format_exc()))
			print(f"{num}) Failed", end="\033[0m\n")

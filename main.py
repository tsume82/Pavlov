import argparse

from agents import AgentBuilder
from utils.plot_utils import plot_episodes, plot_experiment, save_experiment
from utils.config_utils import loadConfiguration, saveConfiguration
import warnings
import traceback
warnings.filterwarnings("ignore")

from os import listdir, makedirs, environ
from os.path import isfile, join, basename, isdir, normpath, exists
# environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"


def getLastCheckpoint(folder):
	if not isdir(folder):
		return None
	
	def cp_filter(file):
		base = basename(file)
		return "checkpoint" in base or "model_weights" in base

	checkpoints = [f for f in listdir(folder) if isfile(join(folder, f)) and cp_filter(f)]

	toint = lambda file: int(basename(file).split("-")[1])
	cp_iterations = [toint(file) for file in checkpoints]

	if len(checkpoints) > 0:
		last = cp_iterations.index(max(cp_iterations))
		return checkpoints[last]

	return None


def loadCheckpoint(agent, checkpoint_arg, folder: str):
	"""
		load checkpoint based on the value of the checkpoint argument:
		- False: Do not load a checkpoint
		- True: Load the last checkpoint in the folder
		- str: Load the given checkpoint (can be both a path or a checkpoint name in the current folder)
	"""
	if checkpoint_arg is not False: # enter if it's str or True
		if checkpoint_arg is True:
			checkpoint_arg = getLastCheckpoint(folder)
			if not checkpoint_arg:
				print(f"There are no checkpoints in {folder}. No checkpoint is loaded")
				return

		checkpoint = checkpoint_arg if isfile(checkpoint_arg) else join(folder, checkpoint_arg)
		
		if isfile(checkpoint):
			agent.load(checkpoint)
			print(f"loaded {checkpoint}")
		else:
			print(f"{checkpoint} not loaded")

def train_agent(agent_config, folder="./.checkpoints", **kwargs):
	max_episodes = kwargs.get("max_episodes", 5000)
	episodes_to_checkpoint = kwargs.get("episodes_to_checkpoint", 1000)

	episodes = 0
	agent = AgentBuilder.build(agent_config)
	loadCheckpoint(agent, kwargs.get("checkpoint", False), folder)
	plotter = plot_episodes()
	
	while episodes < max_episodes:
		res = agent.train()
		if episodes == 0: print(f"{'Episode':^17}│{'Min':^19}│{'Max':^19}│{'Mean':^18}\n{'━'*17}┿{'━'*19}┿{'━'*19}┿{'━'*19}")
		episodes = res["episodes_total"]
		# loss = res["info"]["learner"]["default_policy"]["learner_stats"]["total_loss"]
		# plotter.plot(loss)
		plotter.plot(res["hist_stats"]["episode_reward"][0:res["episodes_this_iter"]])
		print(f"{str(episodes)+' / '+str(max_episodes):^17}│{res['episode_reward_min']:^19.9f}│{res['episode_reward_max']:^19.9f}│{res['episode_reward_mean']:^19.9f}")

		if episodes % episodes_to_checkpoint < res["episodes_this_iter"]:
			agent.save(folder)
			plotter.save(join(folder, "train.svg"), agent_config)
			saveConfiguration(agent_config, folder)

	plotter.close()


def test_agent(agent_config, folder="./.checkpoints", **kwargs):
	agent_config["env.env_config"]["args"] = {"block_render_when_done": True}
	agent_config["agent.algorithm.render_env"] = kwargs.get("plot", True)
	agent_config["env.env_config"]["conditions"] = []
	agent = AgentBuilder.build(agent_config)

	loadCheckpoint(agent, kwargs["checkpoint"], folder)

	agent.act()

def test_multiple_times(agent_config, folder="./.checkpoints", **kwargs):
	agent_config["env.env_config"]["args"] = {"save_trajectory": True}
	agent_config["agent.algorithm.render_env"] = False
	agent = AgentBuilder.build(agent_config)

	loadCheckpoint(agent, kwargs["checkpoint"], folder)

	num_runs = kwargs["num_runs"]
	experiment = []
	for i in range(num_runs):
		print(f"\rrun: {i+1}/{num_runs}\t", end="\n" if i+1 == num_runs else "")
		agent.act()
		experiment.append(agent.env.trajectory)
		agent.env.reset()
		# agent.reset() # Really slow and sometimes not needed. TODO is possible to implement an agent reset without ray.shutdown()?

	title = agent_config["env.env_config"].get("solver_driver_args", "")[2]

	if kwargs.get("save_exp", False):
		save_experiment(experiment, folder, name=kwargs["save_exp"])
	if kwargs.get("plot", True):
		plot_experiment(experiment, title=title if isinstance(title, str) else "fitness")

def create_folder_and_train(agent_config, folder, **kwargs):
	if kwargs.get("checkpoint", None):
		train_agent(agent_config, folder, **kwargs)
	else:
		dir = kwargs.get("create_dir", None)
		if dir:
			makedirs(join(folder, dir))
			next = dir
		else:
			dirs = [f for f in listdir(folder) if isdir(join(folder, f))]
			next = str(max([int(d) for d in dirs if d.isnumeric()] + [0])+1)
			makedirs(join(folder, next))
		train_agent(agent_config, join(folder, next), **kwargs)

def main(agent_config, train=True, folder="./.checkpoints", **kwargs):
	def run_config(config: dict):
		if train:
			# train_agent(config, folder, **kwargs)
			create_folder_and_train(config, folder, **kwargs)
		elif kwargs["num_runs"]:
			test_multiple_times(config, folder, **kwargs)
		else:
			test_agent(config, folder, **kwargs)

	if isinstance(agent_config, list):
		for _config in agent_config:
			try:
				run_config(_config)
			except Exception as e:
				traceback.print_exc()
	else:
		run_config(agent_config)


if __name__ == "__main__":
	from colorama import Fore
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=
	"""\033[33mPavlov: a reinforcement learning python library for training adaptive metaheuristics.\033[39m

\033[32mTest: \033[33mpython\033[39m test/test.py

\033[32mUsage: \033[39mWrite a configuration file, example \033[33mconfig.json\033[39m:
\033[36m
{
	"agent.algorithm": "RayProximalPolicyOptimization",
	"agent.algorithm.render_env": false,
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
		"maximize": false,
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
		"reward_metric_config": [true, true],
		"memes_no": 1,
		"action_space_config": {"step_size": {"max": 3, "min": 1e-5}}
	}
}\033[39m

\033[32mRun the training:
\033[33mpython\033[39m main.py train -c config.json""")

	parser.add_argument("train", nargs='?', type=str, default=True, help="train: '1', 'true' or 'train' for training mode, otherwise test (inference) mode is selected")
	parser.add_argument("--dir", "-d", type=str, default="", help="directory: the directory of the experiment (this directory is used to save and load files)")
	parser.add_argument("--multi", "-m", dest="num_runs", type=int, default=None, help="multiexperiment: run multiple runs of test")
	parser.add_argument("--max_ep", dest="max_episodes", type=int, help="max_episodes: maximum number of episodes in training (default: 5000)", default=5000)
	parser.add_argument("--checkpoint", "-cp", dest="checkpoint", nargs="?", default=False, const=True, help="checkpoint: the name of the checkpoint file to test or training starting from that checkpoint. If no checkpoint is passed, automatically is choosen the last one in the directory")
	parser.add_argument("--ep_to_cp", dest="episodes_to_checkpoint", type=int, help="episodes_to_checkpoint: the number of episodes before saving a checkpoint (default: 1000)", default=1000)
	parser.add_argument("--config","-c", nargs="?", default=False, const=True, help="config: the configuraton file of the experiment, if the flag has no arguments the config.json file in the experiment directory is used")
	parser.add_argument("--multi-config","-mc", dest="multi_config", default=None, help="multi config: test or train from multiple configurations")
	parser.add_argument("--create-dir","-cd", dest="create_dir", default=None, help="create dir with this name and save here the training files (weights, configurations...)")
	parser.add_argument("--save", "-s", dest="save_exp", nargs="?", default=False, const=True, help="save the experiment, it's a flag by default, optionally you can choose the name of the save file")
	parser.add_argument("--noplot", "-np", dest="plot", action="store_false", default=True, help="do no plot during testing")
	# parser.add_argument("--verbose", "-v", dest="verbose", type=int, default=1, help="verbose level: 0: no output, 1: errors, 2: warnings, 3: info, 4: debug")
	args = parser.parse_args()

	if isinstance(args.train, str):
		args.train = args.train.lower() in ["true", "1", "train"]

	kwargs = vars(args)

	train = kwargs.pop("train")
	folder = normpath(kwargs.pop("dir"))

	config = kwargs.pop("config")
	multi_config = kwargs.get("multi_config")

	# load configuration(s) from json or variable in configurations.py
	if multi_config:
		configuration = loadConfiguration(multi_config)
		assert isinstance(configuration, list)
	else:
		configuration = loadConfiguration(config if isinstance(config, str) else folder)
		assert isinstance(configuration, dict)

	# in test mode the checkpoint is mandatory
	if not train and kwargs["checkpoint"] is False:
		kwargs["checkpoint"] = True

	main(configuration, train=train, folder=folder, **kwargs)

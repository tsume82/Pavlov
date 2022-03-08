import argparse

from numpy import argmax
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
	loadCheckpoint(agent, kwargs["checkpoint"], folder)
	plotter = plot_episodes()
	while episodes < max_episodes:
		res = agent.train()
		episodes = res["episodes_total"]
		plotter.plot(res["hist_stats"]["episode_reward"][: res["episodes_this_iter"]])
		# pprint(res)
		print()
		print(f"═════════════════════╣Ep.: {episodes}\t╠═════════════════════")
		print()
		print("Min:\t", res["episode_reward_min"])
		print("Max:\t", res["episode_reward_max"])
		print("Mean:\t", res["episode_reward_mean"])

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
			if len(dirs) == 0:
				next = "1"
			else:
				next = str(max([int(d) for d in dirs])+1)
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
	parser = argparse.ArgumentParser(description="launch the training/testing of an agent")
	parser.add_argument("train", nargs='?', type=str, default=True, help="train: '1', 'true' or 'train' for training mode, otherwise test mode is selected")
	parser.add_argument("--dir", "-d", type=str, default="", help="directory: the directory of the experiment")
	parser.add_argument("--multi", "-m", dest="num_runs", type=int, default=None, help="multiexperiment: run multiple runs of test")
	parser.add_argument("--max_ep", dest="max_episodes", type=int, help="max_episodes: maximum number of episodes in training", default=5000)
	parser.add_argument("--checkpoint", "-cp", dest="checkpoint", nargs="?", default=False, const=True, help="checkpoint: the name of the checkpoint file to test or training starting from that checkpoint. If no checkpoint is passed, automatically is choosen the last one in the directory")
	parser.add_argument("--ep_to_cp", dest="episodes_to_checkpoint", type=int, help="episodes_to_checkpoint: the number of episodes before saving a checkpoint", default=1000)
	parser.add_argument("--config","-c", nargs="?", default=False, const=True, help="config: the configuraton file of the experiment, if the flag has no arguments the config.json file in the experiment directory is used")
	parser.add_argument("--multi-config","-mc", dest="multi_config", default=None, help="multi config: test or train from multiple configurations")
	parser.add_argument("--create-dir","-cd", dest="create_dir", default=None, help="create dir with this name and save here the training")
	parser.add_argument("--save", "-s", dest="save_exp", nargs="?", default=False, const=True, help="save the experiment")
	parser.add_argument("--noplot", "-np", dest="plot", action="store_false", default=True, help="do no plot during testing")
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

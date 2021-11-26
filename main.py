import argparse
from agents import AgentBuilder
from utils.plot_utils import plot_episodes, plot_experiment, save_experiment
from utils.config_utils import loadConfiguration, saveConfiguration
import warnings
import traceback
warnings.filterwarnings("ignore")

from os import listdir, makedirs, environ
from os.path import isfile, join, basename, isdir
# environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"

def getLastCheckpoint(folder):
	if not isdir(folder):
		return ""
	checkpoints = [f for f in listdir(folder) if isfile(join(folder, f)) and "checkpoint" in basename(f)]
	if len(checkpoints) == 0:
		print("There are no checkpoints in {}. No checkpoint is loaded".format(folder))
		return ""
	last = max(checkpoints, key=lambda f: int(basename(f).split("-")[1]))
	print("loaded {}".format(last))
	return last

def train_agent(agent_config, folder="./.checkpoints", **kwargs):
	max_episodes = kwargs.get("max_episodes", 5000)
	checkpoint = kwargs.get("checkpoint", None)
	episodes_to_checkpoint = kwargs.get("episodes_to_checkpoint", 1000)

	episodes = 0
	agent = AgentBuilder.build(agent_config)
	if checkpoint:
		agent.load(folder +"/"+ checkpoint)
		print("loaded {}".format(folder +"/"+ checkpoint))
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

		if episodes % episodes_to_checkpoint < res["episodes_this_iter"]:
			agent.save(folder)
			p.save(folder + "/train.svg", agent_config)
			saveConfiguration(agent_config, folder)

	if not kwargs.get("multi_config", False):
		p.show()
	else:
		p.close()


def test_agent(agent_config, folder="./.checkpoints", **kwargs):
	checkpoint = kwargs.get("checkpoint", None)
	agent_config["env.env_config"]["args"] = {"block_render_when_done": True}
	agent_config["agent.algorithm.render_env"] = kwargs.get("plot", True)
	agent_config["env.env_config"]["conditions"] = []
	agent = AgentBuilder.build(agent_config)
	if checkpoint:
		checkpoint = folder + "/" + checkpoint
	else:
		checkpoint = folder + "/" + getLastCheckpoint(folder)
	if isfile(checkpoint):
		agent.load(checkpoint)
	agent.act()

def test_multiple_times(agent_config, folder="./.checkpoints", **kwargs):
	checkpoint = kwargs.get("checkpoint", None)
	agent_config["env.env_config"]["args"] = {"save_trajectory": True}
	agent_config["agent.algorithm.render_env"] = False
	agent = AgentBuilder.build(agent_config)
	if checkpoint:
		checkpoint = folder + "/" + checkpoint
	else:
		checkpoint = folder + "/" + getLastCheckpoint(folder)
	if isfile(checkpoint):
		agent.load(checkpoint)
	experiment = []
	for i in range(kwargs["num_runs"]):
		print("\rrun: {}/{}\t".format(i+1,kwargs["num_runs"]), end="\n" if i+1 == kwargs["num_runs"] else "")
		agent.act()
		experiment.append(agent.env.trajectory)
		agent.env.reset()
		# agent.reset() # Really slow and sometimes not needed
	title = agent_config["env.env_config"].get("solver_driver_args", "")[2]
	if kwargs.get("save_exp", False):
		save_experiment(experiment, folder, name=kwargs["save_exp"])
	if kwargs.get("plot", True):
		plot_experiment(experiment, title=title if isinstance(title, str) else "fitness")

def create_folder_and_train(agent_config, folder, **kwargs):
	if kwargs.get("checkpoint", None):
		train_agent(agent_config, folder, **kwargs)
	else:
		dirs = [f for f in listdir(folder) if isdir(join(folder, f))]
		if len(dirs) == 0:
			next = "/1"
		else:
			next = "/"+str(max([int(d) for d in dirs])+1)
		makedirs(folder+next)
		train_agent(agent_config, folder+next, **kwargs)

def main(agent_config, train=True, folder="./.checkpoints", **kwargs):
	def parse_config_and_run(config):
		if isinstance(config, str):
			config = loadConfiguration(config if config else folder)
		assert isinstance(config, dict)
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
				parse_config_and_run(_config)
			except Exception as e:
				traceback.print_exc()
	else:
		parse_config_and_run(agent_config)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="launch the training/testing of an agent")
	parser.add_argument("train", nargs='?', type=str, default=True, help="train: '1', 'true' or 'train' for training mode, otherwise test mode is selected")
	parser.add_argument("--dir", "-d", type=str, default="", help="directory: the directory of the experiment")
	parser.add_argument("--multi", "-m", dest="num_runs", type=int, default=None, help="multiexperiment: run multiple runs of test")
	parser.add_argument("--max_ep", dest="max_episodes", type=int, help="max_episodes: maximum number of episodes in training", default=5000)
	parser.add_argument("--checkpoint", "-cp", dest="checkpoint", type=str, help="checkpoint: the name of the checkpoint file to test or training starting from that checkpoint. If no checkpoint is passed, automatically is choosen the last one during test", default=None)
	parser.add_argument("--ep_to_cp", dest="episodes_to_checkpoint", type=int, help="episodes_to_checkpoint: the number of episodes before saving a checkpoint", default=1000)
	parser.add_argument("--config","-c", nargs="?", default=False, const=True, help="config: the configuraton file of the experiment, if the flag has no arguments the config.json file in the experiment directory is used")
	parser.add_argument("--multi-config","-mc", dest="multi_config", default=None, help="multi config: test or train from multiple configurations")
	parser.add_argument("--save", "-s", dest="save_exp", nargs="?", default=False, const=True, help="save the experiment")
	parser.add_argument("--noplot", "-np", dest="plot", action="store_false", default=True, help="do no plot during testing")
	args = parser.parse_args()

	if isinstance(args.train, str):
		args.train = args.train.lower() in ["true", "1", "train"]

	kwargs = vars(args)

	train = kwargs.pop("train")
	folder = kwargs.pop("dir")

	config = kwargs.pop("config")
	multi_config = kwargs.get("multi_config")

	if multi_config:
		configuration = loadConfiguration(multi_config)
		assert isinstance(configuration, list)
	else:
		configuration = config if isinstance(config, str) else "" if config else "ppo_configuration"

	main(configuration, train=train, folder=folder, **kwargs)

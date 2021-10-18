from posixpath import splitext
import sys
import argparse
import types
from agents import AgentBuilder
from pprint import pprint, pformat
from utils.plot_utils import plot_episodes
from examples.configurations import paper_cma_es_configuration, paper_cma_es_configuration_2, ppo_configuration

import warnings

warnings.filterwarnings("ignore")
from os import listdir, makedirs
from os.path import isfile, join, basename, isdir


def getLastCheckpoint(folder):
    checkpoints = [f for f in listdir(folder) if isfile(join(folder, f)) and "checkpoint" in basename(f)]
    if len(checkpoints) == 0:
        raise Exception("There are no checkpoints in "+folder)
    last = max(checkpoints, key=lambda f: int(basename(f).split("-")[1]))
    return last


def train_agent(agent_config, folder="./.checkpoints", **kwargs):
    max_episodes = kwargs.get("max_episodes", 12000)
    checkpoint = kwargs.get("checkpoint", None)
    episodes_to_checkpoint = kwargs.get("episodes_to_checkpoint", 3000)

    episodes = 0
    agent = AgentBuilder.build(agent_config)
    if checkpoint:
        agent.load(folder + checkpoint)
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
            p.save(folder + "/train.svg", agent_config, "config.txt")

    p.show()


def test_agent(agent_config, folder="./.checkpoints", **kwargs):
    checkpoint_number = kwargs.get("checkpoint_number", None)
    agent_config["env.env_config"]["args"] = {"block_render_when_done": True}
    agent_config["agent.algorithm.render_env"] = True
    agent_config["env.env_config"]["conditions"] = []
    agent = AgentBuilder.build(agent_config)
    checkpoint = (
        folder + "/checkpoint-" + str(checkpoint_number) if checkpoint_number else folder+"/"+getLastCheckpoint(folder)
    )
    agent.load(checkpoint)
    agent.act()


def main(agent_config, train=True, folder="./.checkpoints", **kwargs):
    if train:
        # train_agent(agent_config, folder, **kwargs)
        multi_experiment_train(agent_config, folder, **kwargs)
    else:
        test_agent(agent_config, folder, **kwargs)

def multi_experiment_train(agent_config, folder, **kwargs):
    if kwargs.get("checkpoint", None):
        train_agent(agent_config, folder, **kwargs)
    else:
        dirs = [f for f in listdir(folder) if isdir(join(folder, f))]
        if len(dirs) == 0:
            next = "/1"
        else:
            next = "/"+str(int(max(dirs))+1)
        makedirs(folder+next)
        train_agent(agent_config, folder+next, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="launch the training/testing of an agent")
    parser.add_argument("train", nargs='?', type=str, default=True)
    parser.add_argument("--dir", "-d", type=str, default=None)
    parser.add_argument("--cp_num", dest="checkpoint_number", type=int, help="checkpoint_number", default=None)
    parser.add_argument("--max_ep", dest="max_episodes", type=int, help="max_episodes", default=3000)
    parser.add_argument("--checkpoint", "-c", dest="checkpoint", type=str, help="checkpoint", default=None)
    parser.add_argument("--ep_to_cp", dest="episodes_to_checkpoint", type=int, help="episodes_to_checkpoint", default=3000)
    # TODO argument for agent configuration
    args = parser.parse_args()

    if isinstance(args.train, str):
        args.train = args.train.lower() in ["true", "1", "train"]

    kwargs = vars(args)

    train = kwargs.pop("train")
    folder = kwargs.pop("dir")
    folder = folder if folder else "./.checkpoints/CMA ppo"

    main(ppo_configuration, train=train, folder=folder, **kwargs)

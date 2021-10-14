from agents import AgentBuilder
from pprint import pprint
from utils.plot_utils import plot_episodes
from examples.configurations import paper_cma_es_configuration

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"

def main(agent_config, train=True, folder="./.checkpoints"):
    # max_episodes = 12000
    max_episodes = 1000
    episodes = 0
    if train:
        agent = AgentBuilder.build(agent_config)
        # agent.load(folder+"/checkpoint-2000")
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

            if episodes % 3000 == 0:
                agent.save(folder)

        # p.save(folder+"/train.svg", agent_config)
        p.show()

    else:
        agent_config["env.env_config"]["args"] = {"block_render_when_done": True}
        agent_config["agent.algorithm.render_env"] = True
        agent_config["env.env_config"]["conditions"] = []
        agent = AgentBuilder.build(agent_config)
        # agent.load(folder+"/checkpoint-3000")
        agent.act()


if __name__ == "__main__":
    main(paper_cma_es_configuration, train=True, folder="./.checkpoints/CMA paper sphere/")

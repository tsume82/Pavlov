from environments import environment_loader
from agents import Agent

rl_configuration = {
    "model.train": True,
    "model.environments": "gym.CartPole-v0",
    "model.state": "metrics",
    "model.state.metrics": [
        "recentGradients"
    ],
    "model.state.metrics.recentGradients": {
        "method": "avg",
        "on": 0.5,  # half of recent episode
    },
    "agent.algorithm": "REINFORCE",
    "agent.algorithm.REINFORCE.fc_layer": (100, 1),
    "agent.algorithm.net": "ActorNetFC",
    "agent.optimizer": "Adam",
    "agent.optimize.Adam.LR": 0.01
}


def main():
    env = environment_loader(rl_configuration["model.environments"], rl_configuration)
    agent_builder = Agent(rl_configuration, env)

if __name__ == '__main__':
    main()

from environments import SchedulerPolicyEnvironment
from agents import AgentBuilder
from drivers import KimemeSchedulerFileDriver

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
    # env = environment_loader(rl_configuration["model.environments"], rl_configuration)
    # agent_builder = Agent(rl_configuration, env)
    driver = KimemeSchedulerFileDriver("/home/kimeme/RL/PavlovTMP/sample_run.csv.txt", "x\(.*\)", "f",
                                       "Error,OperatorID,OperatorCode,CMAES_Individual_Type,CMAES_Individual_ID,"
                                       "Iteration".split(","))

    steps = 22      # test data (techinically 24, TODO check for corner cases)
    memes_no = 5    # test data (+ DOE of course)
    env = SchedulerPolicyEnvironment(
        driver, steps, memes_no,
        ("RecentGradients",), ((2, 6, 1, None, 1),),
        "Best", (), parameter_tune_config=None
    )

    print(env.kimeme_driver.next_step())


if __name__ == '__main__':
    main()

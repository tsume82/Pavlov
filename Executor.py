class Executor:
    # TODO is this needed? using drivers module
    # basically doing what a tf_agents driver does
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

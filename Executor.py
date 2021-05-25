class Executor
    # basically going what a tf_agents driver does
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent
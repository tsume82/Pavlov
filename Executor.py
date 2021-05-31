class Executor:
    # TODO this IS needed, drivers module communicate low-level with kimeme as per an environment sets them up
    #   this is an interface for the exposed train and enforce library API: agent execution in the classic while loop
    # basically doing what a tf_agents driver does
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent
        # TODO maybe check structure of TF_Agents drivers

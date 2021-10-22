import json
from os.path import isfile
from examples.configurations import ALL_CONFIGURATIONS
# TODO save configuration with pickle? (CONS: not readable, PROS: can save anything)

def saveConfiguration(agent_config, folder):
	with open(folder+"/config.json", "w") as f:
		json.dump(agent_config, f)

"""
	config: name of a configuration in the file configurations.py or path of a configuration json or a folder with a config.json
"""
def loadConfiguration(config: str):
	if isfile(config):
		with open(config, "r") as f:
			agent_config = json.load(f)
	elif isfile(config + "/config.json"):
		with open(config + "/config.json", "r") as f:
			agent_config = json.load(f)
	else:
		agent_config = ALL_CONFIGURATIONS[config]
	return agent_config
	


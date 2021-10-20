import json
from os.path import isfile
# TODO save configuration with pickle? (CONS: not readable, PROS: can save anything)

def saveConfiguration(agent_config, folder):
	with open(folder+"/config.json", "w") as f:
		json.dump(agent_config, f)

def loadConfiguration(folder):
	config_file = folder + "/config.json"
	if isfile(config_file):
		with open(config_file, "r") as f:
			return json.load(f)
	return None


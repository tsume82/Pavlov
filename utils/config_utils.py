import json
from os.path import isdir
# TODO save configuration with pickle? (CONS: not readable, PROS: can save anything)

def saveConfiguration(agent_config, folder):
	with open(folder+"/config.json", "w") as f:
		json.dump(agent_config, f)

def loadConfiguration(folder):
	if isdir(folder):
		folder = folder + "/config.json"
	with open(folder, "r") as f:
		return json.load(f)


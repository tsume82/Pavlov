from environments import SchedulerPolicyRayEnvironment, MemePolicyRayEnvironment
from agents import AgentBuilder
from drivers import KimemeSchedulerFileDriver, RastriginGADriver, CMAdriver
import math
import numpy as np
from pprint import pprint

def rastrign(ind):
	return sum([x**2 - 10 * math.cos(2 * math.pi * x) + 10 for x in ind])

rl_configuration_1 = {
    "agent.algorithm": "Ray_PolicyGradient",
	"env.env_class": MemePolicyRayEnvironment,
	"env.env_config": {
		"steps":10,
		"state_metrics_names" : ["RecentGradients"],
		"state_metrics_config" : [(10, 6, 1, None, 2)],
		"reward_metric" : "Best",
		"reward_metric_config" : [],
		"action_space_config" : {
			"max": 5.12,
			"min": -5.12,
			"dim": 2,
			"popsize": 10
		},
		"obj_function": rastrign,
		"maximize": False
	},
}

rl_configuration_2 = {
    "agent.algorithm": "Ray_PolicyGradient",
    "agent.algorithm.Ray_PolicyGradient.framework" : "tf",
    "agent.algorithm.Ray_PolicyGradient.model" : {
		"use_lstm": True,
	},
	"env.env_class": SchedulerPolicyRayEnvironment,
	"env.env_config": {
		"solver_driver" : RastriginGADriver(2, 10),
		"steps" : 10,
		"memes_no" : 2,
		"state_metrics_names" : ["RecentGradients"],
		"state_metrics_config" : [(10, 6, 1, None, 2)],
		# "space_metrics_config" : ((10, 6, 1, None, 10),),
		"reward_metric" : "Best",
		"reward_metric_config" : [],
		"parameter_tune_config" : None,
	},
}

cma_es_configuration = {
    "agent.algorithm": "Ray_PolicyGradient",
    "agent.algorithm.Ray_PolicyGradient.framework" : "tf",
    "agent.algorithm.Ray_PolicyGradient.model" : {
		"use_lstm": True,
		'lstm_cell_size': 16,
		'fcnet_activation': 'tanh',
		'fcnet_hiddens': [16]
	},
	"env.env_class": SchedulerPolicyRayEnvironment,
	"env.env_config": {
		"solver_driver" : CMAdriver(2, 6),
		"maximize": False,
		"steps" : 100,
		"memes_no" : 1,
		"state_metrics_names" : ["RecentGradients"],
		"state_metrics_config" : [(6, 6, 1, None, 2)],
		"reward_metric" : "Best",
		"reward_metric_config" : [],
		"parameter_tune_config" : {
			"step_size": {"max": 1,"min": 1e-10}
		},
	},
}

def main():
	agent = AgentBuilder.build(rl_configuration_1)
	for i in range(5):
		obs, episode_reward, steps_done = agent.act()
		print(obs)
		print(episode_reward)
		print(steps_done)

def main2():
	agent = AgentBuilder.build(cma_es_configuration)
	# obs, episode_reward, steps_done = agent.act()
	for i in range(10):
		res = agent.train()
		# pprint(res)
		print("-------------------",i,"-------------------")
		print("Number episodes:\t",res['episodes_total'])
		print("Min:\t",res["episode_reward_min"])
		print("Max:\t",res["episode_reward_max"])
		print("Mean:\t", res["episode_reward_mean"])



if __name__ == '__main__':
    main2()

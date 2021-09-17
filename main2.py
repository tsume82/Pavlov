import gym
import ray
from environments import SchedulerPolicyRayEnvironment, MemePolicyRayEnvironment
from agents import AgentBuilder
from drivers import KimemeSchedulerFileDriver, RastrignGADriver
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models import ModelCatalog
import math
import numpy as np

def rastrign(ind):
	return sum([x**2 - 10 * math.cos(2 * math.pi * x) + 10 for x in ind])

rl_configuration_1 = {
    "agent.algorithm": "Ray_PolicyGradient",
	"env.env_class": MemePolicyRayEnvironment,
	"env.env_config_args": {
		"obj_no":1,
		"H":10,
		"steps":10,
		"obj_function":rastrign,
		"step_boundaries":np.array([[-5.12,-5.12],[5.12,5.12]]),
		"var_boundaries":np.array([[-5.12,-5.12,-5.12,-5.12,-5.12,-5.12,-5.12,-5.12,-5.12,-5.12,-5.12,-5.12,-5.12],[5.12,5.12,5.12,5.12,5.12,5.12,5.12,5.12,5.12,5.12,5.12,5.12,5.12]]),
		"dim":2
	},
}

rl_configuration_2 = {
    "agent.algorithm": "Ray_PolicyGradient",
	"env.env_class": SchedulerPolicyRayEnvironment,
	"env.env_config_args": {
		"kimeme_driver" : RastrignGADriver(2, 50),
		"steps" : 10,
		"memes_no" : 2,
		"state_metrics_names" : ("RecentGradients",),
		"space_metrics_config" : ((50, 6, 50, None, 1),),
		"reward_metric" : "Best",
		"reward_metric_config" : (),
		"parameter_tune_config" : None,
	},
}

def main():
	# env = MemePolicyEnvironment(1, 10, 10, rastrign, np.array([[-5.12,-5.12],[5.12,5.12]]), np.array([[-5.12,-5.12],[5.12,5.12]]), None, 2)
	agent = AgentBuilder.build(rl_configuration_1)
	for i in range(5):
		obs, episode_reward, steps_done = agent.act()
		print(obs)
		print(episode_reward)
		print(steps_done)

def main2():
	agent = AgentBuilder.build(rl_configuration_2)
	for i in range(5):
		obs, episode_reward, steps_done = agent.act()
		print(obs)
		print(episode_reward)
		print(steps_done)
	


if __name__ == '__main__':
    main2()

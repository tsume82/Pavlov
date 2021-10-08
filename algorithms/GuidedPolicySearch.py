from algorithms.GPS_helper.algorithm import Algorithm as BADMM_GPS
from algorithms import BADMM_hyperparameters
from ray.rllib.policy import Policy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.trainer import with_common_config
import numpy as np
import gym
from itertools import cycle
from ray.util.debug import disable_log_once_globally
disable_log_once_globally()


class GPSPolicy(Policy):
	def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config):
		super().__init__(observation_space, action_space, config)
		self.w = 1.0
		self._hyperparams = config.update(BADMM_hyperparameters.config)
		self._conditions = config['common']['conditions']
		if 'train_conditions' in config:
			self._train_idx = config['train_conditions']
			self._test_idx = config['test_conditions']
		else:
			self._train_idx = range(self._conditions)
			config['train_conditions'] = config['common']['conditions']
			self._hyperparams=config
			self._test_idx = self._train_idx
		self._train_iterator = cycle(self._train_idx)
		self._test_fncs = config['test_functions']

		# self._data_files_dir = config['common']['data_files_dir']
		# self.policy_path = config['policy_path']
		self.network_config = config['algorithm']['policy_opt']['network_params']

		self.algorithm = config['algorithm']['type'](config['algorithm'])

	def compute_single_action(
		self,
		obs,
		state=None,
		prev_action=None,
		prev_reward=None,
		info=None,
		episode=None,
		clip_actions=None,
		explore=None,
		timestep=None,
		unsquash_actions=None,
		**kwargs
	):
		return self.action_space.sample(), [], {}

	def compute_actions(
		self,
		obs_batch,
		state_batches,
		prev_action_batch=None,
		prev_reward_batch=None,
		info_batch=None,
		episodes=None,
		**kwargs
	):
		# TODO: 0.7 prob to use actual policy

		# return action batch, RNN states, extra values to include in batch
		return [self.action_space.sample() for _ in obs_batch], [], {}

	def learn_on_batch(self, samples):
		pass
		# print(samples)
		# implement your learning code here

	def get_weights(self):
		return {"w": self.w}

	def set_weights(self, weights):
		self.w = weights["w"]

	def load_batch_into_buffer(self, batch, buffer_index: int = 0) -> int:
		self.batch = batch
		return len(batch)

	# def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
	# 	pass

	def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
		# for m, cond in enumerate(self._train_idx):
		# 	for i in range(self._hyperparams['num_samples']):
		# 		self.__take_sample(cond, m, i)

		# traj_sample_lists = [self.agent.get_samples(cond, -self._hyperparams['num_samples']) for cond in self._train_idx]

		# new_sample = 1
		# traj_sample_lists[cond].append(new_sample)
		# # Clear agent samples.
		# self.agent.clear_samples()
		# self.algorithm.iteration(traj_sample_lists)

		# #pol_sample_lists = self._take_policy_samples(self._train_idx)

		# #self._prev_traj_costs, self._prev_pol_costs = self.disp.update(itr,                                                                                                                                                             self.algorithm, self.agent, traj_sample_lists, pol_sample_lists)
		# self.algorithm.policy_opt.policy.pickle_policy(self.algorithm.policy_opt._dO, self.algorithm.policy_opt._dU, self._data_files_dir + ('policy_itr_%02d' % itr))
		# self._test_peformance(t_length=50, iteration=itr)
		return {}

	# def __take_sample(self, cond, m, i, t_length=50):
	# 	if self.algorithm.iteration_count == 0:
	# 		pol = self.algorithm.cur[m].traj_distr
	# 	else:
	# 		if self.algorithm._hyperparams['sample_on_policy']:
	# 			pol = self.algorithm.policy_opt.policy
	# 		else:
	# 			if np.random.rand() < 0.7:
	# 				pol = self.algorithm.cur[m].traj_distr
	# 			else:
	# 				pol = CSAPolicy(T=self.agent.T)

	# 	self.agent.sample(pol, cond, t_length=t_length)

GuidedPolicySearch = build_trainer(
	name="GuidedPolicySearch",
	default_config=with_common_config(
		{
			"num_workers": 0,
		}
	).update(BADMM_hyperparameters.config),
	default_policy=GPSPolicy,
)
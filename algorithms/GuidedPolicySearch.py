from ray.rllib.utils.annotations import override
from algorithms.GPS_helper.algorithm import Algorithm as BADMM_GPS
from algorithms import BADMM_hyperparameters
from ray.rllib.policy import Policy
from algorithms.GPS_helper.policy import CSAPolicy
from algorithms.GPS_helper.lin_gauss_policy import LinearGaussianPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.rllib.agents.trainer import Trainer, with_common_config
import numpy as np
import gym
from itertools import cycle
from ray.util.debug import disable_log_once_globally

disable_log_once_globally()


class GPSPolicy(Policy):
	def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config):
		super().__init__(observation_space, action_space, config)
		self.w = 1.0
		config.update(BADMM_hyperparameters.config)
		self._hyperparams = config
		self.evolution_length = 50
		self.samples_length = 25
		self.curr_step = 0
		self._conditions = config["common"]["conditions"]
		if "train_conditions" in config:
			self._train_idx = config["train_conditions"]
			self._test_idx = config["test_conditions"]
		else:
			self._train_idx = range(self._conditions)
			config["train_conditions"] = config["common"]["conditions"]
			self._hyperparams = config
			self._test_idx = self._train_idx
		self._train_iterator = cycle(self._train_idx)
		self._test_fncs = config["test_functions"]

		# self._data_files_dir = config['common']['data_files_dir']
		# self.policy_path = config['policy_path']
		self.network_config = config["algorithm"]["policy_opt"]["network_params"]

		self.algorithm = config["algorithm"]["type"](config["algorithm"])
		self.phase = 0  # init trajectory phase
		self.CSApolicy = CSAPolicy(T=self.evolution_length)
		self.condition_archive = [[] for _ in self._train_idx]
		self.action_mean = []

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

		# if noisy:
		#     noise = np.random.randn(self.evolution_length, self.dU)
		# else:
		noise = np.zeros((self.evolution_length, self.algorithm.dU))

		if self.phase == 0:
			es = self.config["env_config"]["solver_driver"].es
			fitness = self.config["env_config"]["solver_driver"].fitness
			action = self.CSApolicy.act(
				obs_batch[0], None, self.curr_step, noise, es, fitness
			)  # TODO why action is always the same

		# elif self.phase == 1:

		# TODO: 0.7 prob to use actual policy
		# if self.algorithm.iteration_count == 0:
		# 	policy = self.algorithm.cur[cond].traj_distr
		# else:
		# 	if self.algorithm._hyperparams['sample_on_policy']:
		# 		policy = self.algorithm.policy_opt.policy
		# 	else:
		# 		if np.random.rand() < 0.7:
		# 			policy = self.algorithm.cur[cond].traj_distr
		# 		else:
		# 			policy = CSAPolicy(T=self.agent.T)

		# policy.act(obs_batch[0], None, self.curr_step, noise[self.curr_step, :], None, None)

		self.curr_step += 1
		# if self.phase == 0 and self.curr_step >= self._train_idx:
		# 	self.phase = 1 # stop creating trajectories from CSA

		return [{"step_size": action}], [], {}

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
		l = self.evolution_length
		for evolution in range(len(self.batch["actions"]) // l - 1):
			sliced_batch = self.batch[evolution * l : (evolution + 1) * l]
			actions = sliced_batch["actions"]
			cond = sliced_batch["infos"][0]["condition"]
			self.condition_archive[cond].append(actions)
			if len(self.condition_archive[cond]) >= self.samples_length:
				mean_actions = np.mean(self.condition_archive[cond], axis=0)
				var_actions = np.std(self.condition_archive[cond], axis=0)
				kt = mean_actions.reshape((self.evolution_length, 1))
				Kt = np.zeros((self.algorithm.dU, self.algorithm.dX))
				K = np.tile(Kt[None, :, :], (self.evolution_length, 1, 1))
				PSig = var_actions.reshape((self.evolution_length, 1, 1))
				cholPSig = np.sqrt(var_actions).reshape((self.evolution_length, 1, 1))
				invPSig = 1.0 / var_actions.reshape((self.evolution_length, 1, 1))
				self.algorithm.cur[cond].traj_distr = LinearGaussianPolicy(K, kt, PSig, cholPSig, invPSig)

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

	def __take_sample(self, cond, m, i, t_length=50):
		if self.algorithm.iteration_count == 0:
			pol = self.algorithm.cur[m].traj_distr
		else:
			if self.algorithm._hyperparams["sample_on_policy"]:
				pol = self.algorithm.policy_opt.policy
			else:
				if np.random.rand() < 0.7:
					pol = self.algorithm.cur[m].traj_distr
				else:
					pol = CSAPolicy(T=self.agent.T)

		self.agent.sample(pol, cond, t_length=t_length)


class GuidedPolicySearch(Trainer):
	"""general implementation of Guided Policy Search"""

	_name = "GuidedPolicySearch"

	@override(Trainer)
	def __init__(self, env, config):
		env_creator = _global_registry.get(ENV_CREATOR, env)
		self.env = env_creator(config["env_config"])
		config.update(BADMM_hyperparameters.config)
		config.update(with_common_config({"num_workers": 0}))
		self.config = config
		self._hyperparams = config
		self.evolution_length = 50
		self.samples_length = 25
		self.curr_step = 0
		self._conditions = config["common"]["conditions"]
		if "train_conditions" in config:
			self._train_idx = config["train_conditions"]
			self._test_idx = config["test_conditions"]
		else:
			self._train_idx = range(self._conditions)
			config["train_conditions"] = config["common"]["conditions"]
			self._hyperparams = config
			self._test_idx = self._train_idx
		self._train_iterator = cycle(self._train_idx)
		self._test_fncs = config["test_functions"]

		# self._data_files_dir = config['common']['data_files_dir']
		# self.policy_path = config['policy_path']
		self.network_config = config["algorithm"]["policy_opt"]["network_params"]

		self.algorithm = config["algorithm"]["type"](config["algorithm"])
		self.phase = 0  # init trajectory phase
		self.CSApolicy = CSAPolicy(T=self.evolution_length)
		self.condition_archive = [[] for _ in self._train_idx]
		self.action_mean = []

		#################################################################

		self._episodes_total = 0
		self.episodes = []

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
	
		# if noisy:
		#     noise = np.random.randn(self.evolution_length, self.dU)
		# else:
		noise = np.zeros((self.evolution_length, self.algorithm.dU))

		if self.phase == 0:
			es = self.env.solver_driver.es
			fitness = self.env.solver_driver.fitness
			action = self.CSApolicy.act(
				obs, None, self.curr_step, noise, es, fitness
			)

		# elif self.phase == 1:

		# TODO: 0.7 prob to use actual policy
		# if self.algorithm.iteration_count == 0:
		# 	policy = self.algorithm.cur[cond].traj_distr
		# else:
		# 	if self.algorithm._hyperparams['sample_on_policy']:
		# 		policy = self.algorithm.policy_opt.policy
		# 	else:
		# 		if np.random.rand() < 0.7:
		# 			policy = self.algorithm.cur[cond].traj_distr
		# 		else:
		# 			policy = CSAPolicy(T=self.agent.T)

		# policy.act(obs_batch[0], None, self.curr_step, noise[self.curr_step, :], None, None)

		self.curr_step += 1
		# if self.phase == 0 and self.curr_step >= self._train_idx:
		# 	self.phase = 1 # stop creating trajectories from CSA

		return {"step_size": action}, [], {}

	@override(Trainer)
	def train(self):
		episode_reward = 0
		episodes_this_iteration = 0
		done = False
		obs = self.env.reset()
		
		for _ in range(self.config["train_batch_size"]):
			action, rnn_state, info = self.compute_single_action(obs)
			obs, reward, done, info = self.env.step(action)
			episode_reward += reward
			if done:
				obs = self.env.reset()
				self._episodes_total += 1
				episodes_this_iteration += 1
				self.episodes.append(episode_reward)

		

		return {
			"episodes_total": self._episodes_total,
			"episodes_this_iter": episodes_this_iteration,
			"hist_stats":{
				"episode_reward": self.episodes
			}
		}


# GuidedPolicySearch_ = build_trainer(
# 	name="GuidedPolicySearch",
# 	default_config=with_common_config(
# 		{
# 			"num_workers": 0,
# 		}
# 	).update(BADMM_hyperparameters.config),
# 	default_policy=GPSPolicy,
# )

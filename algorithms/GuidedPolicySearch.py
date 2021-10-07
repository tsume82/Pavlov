from algorithms.GPS.python.gps.algorithm.algorithm_badmm import AlgorithmBADMM
from ray.rllib.policy import Policy
from ray.rllib.agents.trainer_template import build_trainer
import gym

class GPSPolicy(Policy):
	def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config):
		super().__init__(observation_space, action_space, config)

	def compute_single_action(self, obs: TensorType, state: Optional[List[TensorType]] = None, prev_action: Optional[TensorType] = None, prev_reward: Optional[TensorType] = None, info: dict = None, episode: Optional["MultiAgentEpisode"] = None, clip_actions: bool = None, explore: Optional[bool] = None, timestep: Optional[int] = None, unsquash_actions: bool = None, **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
		return super().compute_single_action(obs, state=state, prev_action=prev_action, prev_reward=prev_reward, info=info, episode=episode, clip_actions=clip_actions, explore=explore, timestep=timestep, unsquash_actions=unsquash_actions, **kwargs)

	

GradientPolicySearch = build_trainer(
	name="GradientPolicySearch",
	default_policy=GPSPolicy
)
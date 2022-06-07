import numpy as np

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter


def build_experience_from_csv(env, out_filename, in_filenames=None):
    """
    :param env: environment to load data from, using its kimeme_driver
    :param out_filename: experience json output file
    :param in_filenames: input files to load from
    :return:
    """
    if in_filenames is None:
        in_filenames = []
    batch_builder = SampleBatchBuilder()  # TODO or MultiAgentSampleBatchBuilder, investigate
    writer = JsonWriter(out_filename)

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    # TODO we do have complex envs, keep an eye on this
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    eps_id = 0
    filename_i = 0
    # first time use the env.kimeme_driver own source, if present
    if env.initialized():
        filename_i = -1
    while filename_i < len(in_filenames):
        if filename_i >= 0:
            # TODO
            env.kimeme_driver.reset(new_source=in_filenames[filename_i], autoreinit=True)
        obs = env.reset()
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        done = False
        t = 0
        while not done:
            action, new_obs, rew, done, info = env.next_step()
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0,  # put the true action probability here TODO needs investigation, 1.0 should be correct
                action_logp=0.0,
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos=info,
                new_obs=prep.transform(new_obs))
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
        writer.write(batch_builder.build_and_reset())
        eps_id += 1
        filename_i += 1


def configure_problem():
    pass

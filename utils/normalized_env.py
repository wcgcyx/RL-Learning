import gym
import numpy as np
import jitterbug_dmc
from dm_control import suite

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


def get_normalized_env(task):
    if task == 'jitterbug':
        env = jitterbug_dmc.JitterbugGymEnv(
            suite.load(
            domain_name="jitterbug",
            task_name="move_from_origin",
            visualize_reward=True
            )
        )
    else:
        env = gym.make(task)
    return NormalizedActions(env)

import numpy as np
import random
from collections import deque
from .actor import Actor
from .critic import Critic


class Agent:

    def __init__(
            self,
            action_spec,
            observation_spec,
            actor_learning_rate=0.0001,
            critic_learning_rate=0.001,
            discount_factor=0.99,
            tau=0.001,
            min_std=-20,
            max_std=2,
            epsilon=0.000001,
            memory_size=100000,
            batch_size=64):
        self.action_dim = np.product(action_spec.shape)
        self.state_dim = 0
        for _, item in observation_spec.items():
            self.state_dim += np.product(item.shape)

        self.discount_factor = discount_factor
        self.actor = Actor(action_dim=self.action_dim, state_dim=self.state_dim,
                           learning_rate=actor_learning_rate, min_std=min_std,
                           max_std=max_std, epsilon=epsilon)
        self.critic = Critic(action_dim=self.action_dim, state_dim=self.state_dim,
                             learning_rate=critic_learning_rate, tau=tau)

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def choose_action(self, state):
        state = convert_state(state)
        return self.actor.predict(state)

    def store_transition(self, state_1, action, reward, end, state_2):
        memory = convert_state(state_1)
        memory = np.append(memory, action)
        memory = np.append(memory, reward)
        memory = np.append(memory, end)
        memory = np.append(memory, convert_state(state_2))
        self.memory.append(memory)

    def learn(self):
        if len(self.memory) > self.batch_size:
            batch_memory = np.array(random.sample(list(self.memory), self.batch_size))
        else:
            batch_memory = np.array(self.memory)
        state_1 = batch_memory[:, :self.state_dim]
        actions = batch_memory[:, self.state_dim:self.state_dim + self.action_dim]
        reward = batch_memory[:, self.state_dim + self.action_dim]
        end = batch_memory[:, self.state_dim + self.action_dim + 1]
        state_2 = batch_memory[:, -self.state_dim:]

        # Training q networks
        target_value = self.critic.predict_v_target(state_2)
        target_q_value = reward + (1 - end) * self.discount_factor * target_value
        self.critic.learn_q(state_1, actions, target_q_value)

        # Training v network
        new_action, log_prob = self.actor.evaluate(state_1)
        q_values_1, q_values_2 = self.critic.predict_q(state_1, new_action)
        q_values = np.minimum(q_values_1, q_values_2)
        target_v_value = q_values - log_prob
        self.critic.learn_v(state_1, target_v_value)

        # Training policy network
        q_action_grads = self.critic.get_action_grads(state_1, new_action)
        self.actor.learn(state_1, q_action_grads)

        # Update target v
        self.critic.update_target_v()

    def save_weights(self, domain, task):
        pass

    def load_weights(self, domain, task):
        pass


def convert_state(state):
    result = []

    for _, item in state.items():
        result.extend(item.flatten())

    return np.array(result)
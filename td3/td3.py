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
            memory_size=100000,
            batch_size=64,
            action_noise=1,
            target_noise=0.2,
            noise_clip=0.5,
            update_delay=2):
        self.action_dim = np.product(action_spec.shape)
        self.state_dim = 0
        for _, item in observation_spec.items():
            self.state_dim += np.product(item.shape)

        self.discount_factor = discount_factor
        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim,
                           learning_rate=actor_learning_rate, tau=tau, action_noise=action_noise)
        self.critic = Critic(state_dim=self.state_dim, action_dim=self.action_dim,
                             learning_rate=critic_learning_rate, tau=tau)

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.update_delay = update_delay
        self.update_counter = 1

    def choose_action(self, state):
        state = convert_state(state)[np.newaxis, :]
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
        # Compute target actions (nested clip)
        target_actions = np.clip(self.actor.predict(state_2) +
                                 np.clip(self.target_noise * np.random.randn(self.action_dim),
                                         -self.noise_clip, self.noise_clip), -1, 1)
        # Compute critic target values
        q_values_1, q_values_2 = self.critic.target_predict(state_2, target_actions)
        q_values = np.minimum(q_values_1, q_values_2)
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if end[i]:
                critic_target[i] = reward[i]
            else:
                critic_target[i] = reward[i] + self.discount_factor * q_values[i]
        # Let the critic learns
        self.critic.learn(states=state_1, actions=actions, critic_target=critic_target)
        # Let the actor learns
        if self.update_counter >= self.update_delay:
            action_grads = self.critic.get_action_grads(state_1, self.actor.predict(state_1))
            self.actor.learn(states=state_1, action_grads=np.array(action_grads).reshape(-1, self.action_dim))
            self.update_counter = 1
        else:
            self.update_counter += 1
        # Update the target models by tau factor
        self.actor.update_target()
        self.critic.update_target()

    def save_weights(self, domain, task):
        self.actor.model.save_weights("td3_actor_model_{}_{}.h5".format(domain, task))
        self.actor.target_model.save_weights("td3_actor_target_{}_{}.h5".format(domain, task))
        self.critic.model_1.save_weights("td3_critic_model_1_{}_{}.h5".format(domain, task))
        self.critic.target_model_1.save_weights("td3_critic_target_1_{}_{}.h5".format(domain, task))
        self.critic.model_2.save_weights("td3_critic_model_2_{}_{}.h5".format(domain, task))
        self.critic.target_model_2.save_weights("td3_critic_target_2_{}_{}.h5".format(domain, task))

    def load_weights(self, domain, task):
        self.actor.model.load_weights("td3_actor_model_{}_{}.h5".format(domain, task))
        self.actor.target_model.load_weights("td3_actor_target_{}_{}.h5".format(domain, task))
        self.critic.model_1.load_weights("td3_critic_model_1_{}_{}.h5".format(domain, task))
        self.critic.target_model_1.load_weights("td3_critic_target_1_{}_{}.h5".format(domain, task))
        self.critic.model_2.load_weights("td3_critic_model_2_{}_{}.h5".format(domain, task))
        self.critic.target_model_2.load_weights("td3_critic_target_2_{}_{}.h5".format(domain, task))


def convert_state(state):
    result = []

    for _, item in state.items():
        result.extend(item.flatten())

    return np.array(result)
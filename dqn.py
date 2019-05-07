import random
import numpy as np
from collections import deque
from tensorflow._api.v1.keras.models import Sequential
from tensorflow._api.v1.keras.layers import Dense
from tensorflow._api.v1.keras.optimizers import Adam


class Agent:

    def __init__(
            self,
            action_spec,
            observation_spec,
            action_size=11,    # Discretize action space
            learning_rate=0.001,
            discount_factor=0.95,
            e_greedy_start=1.0,
            e_greedy_decay=0.995,
            e_greedy_min=0.05,
            memory_size=10000,
            batch_size=64,
            replace_q_target_iteration=200,
            ):
        action_dim = np.product(action_spec.shape)
        if action_dim != 1:
            raise Exception("DQN only support action dimension of 1, given dimension is"
                            + action_dim.tostring())
        self.action_size = action_size

        # Get feature number
        self.feature_number = 0
        for _, item in observation_spec.items():
            self.feature_number += np.product(item.shape)

        # Set constants
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.e_greedy = e_greedy_start
        self.e_greedy_decay = e_greedy_decay
        self.e_greedy_min = e_greedy_min
        self.batch_size = batch_size
        self.replace_q_target_iteration = replace_q_target_iteration

        # Count used to update the q_target, starts with 1
        self.learning_count = 1

        # Create replay memory
        self.memory = deque(maxlen=memory_size)

        # Build neural network
        self.q_evaluate, self.q_target = self._build_network()

    def learn(self):
        if self.learning_count % self.replace_q_target_iteration == 0:
            self.q_target.set_weights(self.q_evaluate.get_weights())
        if len(self.memory) > self.batch_size:
            batch_memory = np.array(random.sample(list(self.memory), self.batch_size))
            batch_size = self.batch_size
        else:
            batch_memory = np.array(self.memory)
            batch_size = len(self.memory)
        state_1 = batch_memory[:, :self.feature_number]
        actions = ((batch_memory[:, self.feature_number]) + 1) / (2 / (self.action_size - 1))
        actions = actions.astype(int)
        reward = batch_memory[:, self.feature_number + 1]
        end = batch_memory[:, self.feature_number + 2]
        state_2 = batch_memory[:, -self.feature_number:]

        q_predict = self.q_evaluate.predict(state_1)
        q_future = self.q_target.predict(state_2)

        # Make a copy of the prediction
        target = q_predict.copy()

        # Only update the chosen action
        for i in range(batch_size):
            target[i, actions[i]] = reward[i] \
                if end[i] == 1 else reward[i] + \
                                    self.discount_factor * np.max(q_future[i])

        # Training
        self.q_evaluate.fit(x=state_1, y=target, epochs=1, verbose=0)

        if self.e_greedy > self.e_greedy_min:
            self.e_greedy *= self.e_greedy_decay

        self.learning_count += 1

    def store_transition(self, state_1, action, reward, end, state_2):
        memory = convert_state(state_1)
        memory = np.append(memory, action)
        memory = np.append(memory, reward)
        memory = np.append(memory, end)
        memory = np.append(memory, convert_state(state_2))
        self.memory.append(memory)

    def choose_action(self, state):
        state = convert_state(state)
        state = state[np.newaxis, :]

        if np.random.uniform() > self.e_greedy:
            action_value_list = self.q_evaluate.predict(state)
            action = np.argmax(action_value_list)
        else:
            action = np.random.randint(0, self.action_size)

        # Action now is between 0 and action_size - 1
        # Convert to action between -1 and 1
        return np.array([-1 + action * (2 / (self.action_size - 1))])

    def _build_network(self):
        # Create q-evaluate
        q_evaluate = Sequential()
        q_evaluate.add(Dense(32, input_dim=self.feature_number, activation='relu'))
        q_evaluate.add(Dense(32, activation='relu'))
        q_evaluate.add(Dense(32, activation='relu'))
        q_evaluate.add(Dense(self.action_size, activation='linear'))
        q_evaluate.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        # Create q-target
        q_target = Sequential()
        q_target.add(Dense(32, input_dim=self.feature_number, activation='relu'))
        q_target.add(Dense(32, activation='relu'))
        q_target.add(Dense(32, activation='relu'))
        q_target.add(Dense(self.action_size, activation='linear'))
        q_target.set_weights(q_evaluate.get_weights())
        return q_evaluate, q_target

    def save_weights(self, filename):
        self.q_evaluate.save_weights(filename)

    def load_weights(self, filename):
        self.q_evaluate.load_weights(filename)
        self.q_target.load_weights(filename)


def convert_state(state):
    result = []

    for _, item in state.items():
        result.extend(item.flatten())

    return np.array(result)
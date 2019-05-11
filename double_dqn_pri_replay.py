import random
import numpy as np
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
            memory_size=2000,
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
        self.memory = Memory(memory_size)

        # Build neural network
        self.q_evaluate, self.q_target = self._build_network()

    def learn(self):

        batch_size = self.batch_size
        batch = self.memory.sample(batch_size)
        batch = np.array(list(batch))

        idx = batch[:,0]
        batch_memory = batch[:,1]
        batch_memory = np.vstack(batch_memory).astype(np.float)

        state_1 = batch_memory[:, :self.feature_number]
        actions = ((batch_memory[:, self.feature_number]) + 1) / (2 / (self.action_size - 1))
        actions = actions.astype(int)
        reward = batch_memory[:, self.feature_number + 1]
        end = batch_memory[:, self.feature_number + 2]
        state_2 = batch_memory[:, -self.feature_number:]

        q_predict = self.q_evaluate.predict(state_1)
        q_predict_actions = np.argmax(self.q_evaluate.predict(state_2), axis=1)
        q_future = self.q_target.predict(state_2)

        # Make a copy of the prediction
        target = q_predict.copy()

        # Only update the chosen action
        for i in range(batch_size):
            target[i, actions[i]] = reward[i] \
                if end[i] == 1 else reward[i] + \
                                    self.discount_factor * q_future[i, q_predict_actions[i]]

        # Training
        self.q_evaluate.fit(x=state_1, y=target, epochs=1, verbose=0)

        # Update error in the sum tree
        for i in range(batch_size):
            state_1_ = state_1[i]
            state_1_ = state_1_[np.newaxis, :]
            action_ = actions[i]
            reward_ = reward[i]
            end_ = end[i]
            state_2_ = state_2[i]
            state_2_ = state_2_[np.newaxis, :]
            q_predict_ = self.q_evaluate.predict(state_1_)
            q_predict_action_ = np.argmax(self.q_evaluate.predict(state_2_))
            q_future_ = self.q_target.predict(state_2_)
            target_ = reward_ if end_ == 1 else reward_ + self.discount_factor * q_future_[0, q_predict_action_]
            target_error = abs(q_predict_[0, action_] - target_)
            idx_ = idx[i]
            self.memory.update(idx_, target_error)

        if self.e_greedy > self.e_greedy_min:
            self.e_greedy *= self.e_greedy_decay

        self.learning_count += 1

    def store_transition(self, state_1, action, reward, end, state_2):
        memory = convert_state(state_1)
        memory = np.append(memory, action)
        memory = np.append(memory, reward)
        memory = np.append(memory, end)
        memory = np.append(memory, convert_state(state_2))
        # Calculate Error
        state_1 = convert_state(state_1)[np.newaxis, :]
        state_2 = convert_state(state_2)[np.newaxis, :]
        action = (action + 1) / (2 / (self.action_size - 1))
        action = action.astype(int)
        q_predict = self.q_evaluate.predict(state_1)
        q_predict_action = np.argmax(self.q_evaluate.predict(state_2))
        q_future = self.q_target.predict(state_2)
        target = reward if end == 1 else reward + self.discount_factor * q_future[0, q_predict_action]
        target_error = abs(q_predict[0, action] - target)
        self.memory.add(target_error, memory)

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

    def __str__(self):
        return "double_dqn_pri_replay"


class SumTree:
    # Code from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    # Code from https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


def convert_state(state):
    result = []

    for _, item in state.items():
        result.extend(item.flatten())

    return np.array(result)
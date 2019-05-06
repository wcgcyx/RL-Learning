import tensorflow as tf
import numpy as np
import random
from collections import deque

random.seed(1)
tf.set_random_seed(1)


class Agent:

    def __init__(
            self,
            action_spec,
            observation_spec,
            learning_rate=0.01,
            discount_factor=0.9,
            e_greedy=0.95,
            fixed_q_iteration=200,
            memory_size=2000,
            batch_size=64
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.e_greedy = e_greedy
        self.fixed_q_iteration = fixed_q_iteration
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_count = 0

        action_dim = np.product(action_spec.shape)
        if action_dim != 1:
            raise Exception("DQN only support action dimension of 1, given dimension is"
                            + action_dim.tostring())

        # Discrete action space to 11 for each dimension.
        # -1 -0.8 -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0
        self.action_size = 11 ** action_dim

        # Get feature number
        self.feature_number = 0
        for _, item in observation_spec.items():
            self.feature_number += np.product(item.shape)

        # Create replay memory
        self.replay_memory = deque()

        # Build neural net
        self._build_network()

    def learn(self):
        if self.learn_count % self.fixed_q_iteration == 0:
            # It's time to update q_target_network
            self.sess.run(tf.assign(self.target_layer1_w, self.q_layer1_w))
            self.sess.run(tf.assign(self.target_layer1_b, self.q_layer1_b))
            self.sess.run(tf.assign(self.target_layer2_w, self.q_layer2_w))
            self.sess.run(tf.assign(self.target_layer2_b, self.q_layer2_b))

        if len(self.replay_memory) > self.batch_size:
            batch_memory = np.array(random.sample(list(self.replay_memory), self.batch_size))
            batch_size = self.batch_size
        else:
            batch_memory = np.array(self.replay_memory)
            batch_size = len(self.replay_memory)
        state_1 = batch_memory[:,:self.feature_number]
        actions = ((batch_memory[:,self.feature_number]) + 1) / 0.2
        actions = actions.astype(int)
        reward = batch_memory[:,self.feature_number + 1]
        end = batch_memory[:,self.feature_number + 2]
        state_2 = batch_memory[:,-self.feature_number:]

        # Calculate all the q values at state_1 using evaluate network
        q_this = self.sess.run(self.prediction, feed_dict={self.s: state_1})
        q_next = self.sess.run(self.target_evaluate_q, feed_dict={self.s_ : state_2})

        q_target = q_this.copy()
        batch_index = np.arange(batch_size, dtype=np.int)

        for i in batch_index:
            q_target[i, actions[i]] = reward[i]\
                                    if end[i] == 1 else reward[i] +\
                                    self.discount_factor + np.max(q_next[i])

        self.sess.run(self.train_step, feed_dict={self.s: state_1, self.q : q_target})
        self.learn_count += 1

    def choose_action(self, state):
        state = state_to_array(state)
        state = state[np.newaxis, :]

        if np.random.uniform() < self.e_greedy:
            actions_value = self.sess.run(self.target_evaluate_q, feed_dict={self.s_: state})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.action_size)

        # Action now is from 0 to action_size - 1
        # Return a value from -1 to 1
        return np.array([-1 + action * 0.2])

    def store_transition(self, state_1, action, reward, end, state_2):
        memory = state_to_array(state_1)
        memory = np.append(memory, action)
        memory = np.append(memory, reward)
        memory = np.append(memory, end)
        memory = np.append(memory, state_to_array(state_2))
        self.replay_memory.append(memory)
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory.popleft()

    def _build_network(self):
        # Training-Q-Network (Network 1)
        # Input (feature_number) (relu) -> hidden (100) -> output (action_size)

        # Input state and output a list of q values
        self.s = tf.placeholder(tf.float32, [None, self.feature_number])
        self.q = tf.placeholder(tf.float32, [None, self.action_size])

        # Layer 1
        self.q_layer1_w, self.q_layer1_b, q_layer_1 =\
            add_layer(self.s, self.feature_number, 100, tf.nn.relu)
        # Layer 2
        self.q_layer2_w, self.q_layer2_b, self.prediction =\
            add_layer(q_layer_1, 100, self.action_size, activation_function=None)

        loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.q - self.prediction)))

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        # Target-Q-Network (Network 2)
        # Input (feature_number) (relu) -> hidden (100) -> output (action_size)

        self.s_ = tf.placeholder(tf.float32, [None, self.feature_number])

        # Layer 1
        self.target_layer1_w, self.target_layer1_b, target_layer_1 =\
            add_layer(self.s_, self.feature_number, 100, tf.nn.relu)

        # Layer 2
        self.target_layer2_w, self.target_layer2_b, self.target_evaluate_q =\
            add_layer(target_layer_1, 100, self.action_size, activation_function=None)

        # Init session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.assign(self.target_layer1_w, self.q_layer1_w))
        self.sess.run(tf.assign(self.target_layer1_b, self.q_layer1_b))
        self.sess.run(tf.assign(self.target_layer2_w, self.q_layer2_w))
        self.sess.run(tf.assign(self.target_layer2_b, self.q_layer2_b))


def state_to_array(state):
    result = []

    for _, item in state.items():
        result.extend(item.flatten())

    return np.array(result)


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return Weights, biases, outputs
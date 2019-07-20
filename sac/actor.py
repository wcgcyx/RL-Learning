import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Lambda
from tensorflow.python.keras.optimizers import Adam
import tensorflow.python.keras.backend as K


class Actor:

    def __init__(
            self,
            action_dim,
            state_dim,
            learning_rate,
            min_std,
            max_std,
            epsilon):
        # Actor in SAC has one network:
        # One Policy Network
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.min_std = min_std
        self.max_std = max_std
        self.epsilon = epsilon
        # Build network
        self.policy_net = self.__build_net()
        # Get optimizer
        self.optimizer = self.__get_optimizer()
        # Need to do a initialization to avoid errors
        K.get_session().run(tf.global_variables_initializer())

    def predict(self, state):
        mean, log_std = self.policy_net.predict(state)
        std = tf.math.exp(log_std)
        # Perform reparameterization trick
        normal = tf.distributions.Normal(0, 1)
        z = normal.sample()
        action = tf.tanh(mean + std * z)
        return action

    def evaluate(self, state):
        mean, log_std = self.policy_net.predict(state)
        std = tf.math.exp(log_std)
        # Perform reparameterization trick
        normal = tf.distributions.Normal(0, 1)
        z = normal.sample()
        action = tf.tanh(mean + std * z)
        log_prob = \
            tf.distributions.Normal(mean, std).log_prob(mean + std * z) - tf.log(1 - tf.pow(action, 2) + self.epsilon)
        return action, log_prob

    def learn(self, states, q_action_grads):
        self.optimizer(inputs=[states, q_action_grads])

    def __build_net(self):
        state_input = Input(shape=[self.state_dim])
        layer1 = Dense(400, activation='relu')(state_input)
        layer2 = Dense(300, activation='relu')(layer1)
        mean_output = Dense(self.action_dim, activation='tanh')(layer2)
        raw_log_std_output = Dense(self.action_dim, activation='linear')(layer2)
        log_std_output = Lambda(lambda x: K.clip(x, self.min_std, self.max_std))(raw_log_std_output)
        model = Model(state_input, [mean_output, log_std_output])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def __get_optimizer(self):
        mean, log_std = self.policy_net.output
        std = tf.math.exp(log_std)
        # Perform reparameterization trick
        normal = tf.distributions.Normal(0.0, 1.0)
        z = normal.sample()
        action = tf.tanh(mean + std * z)
        log_prob =\
            tf.distributions.Normal(mean, std).log_prob(mean + std * z) - tf.log(1 - tf.pow(action, 2) + self.epsilon)
        # Grad of Q(s, a) wrt Action
        q_action_grads = K.placeholder(shape=(None, self.action_dim))
        # Grad of Log Prob wrt Weights
        prob_weight_grads = tf.gradients(log_prob, self.policy_net.trainable_weights)
        # Grad of Log Prob wrt Action
        prob_action_grads = tf.gradients(log_prob, action)
        # Grad of Action wrt Weights
        action_weight_grads = tf.gradients(action, self.policy_net.trainable_weights)
        # Grad of Loss wrt weights
        loss_weight_grads = prob_weight_grads + (prob_action_grads - q_action_grads) * action_weight_grads

        grads = zip(loss_weight_grads, self.policy_net.trainable_weights)
        return K.function(inputs=[self.policy_net.input, q_action_grads], outputs=[],
                          updates=[tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)])
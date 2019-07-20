import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Lambda
from tensorflow.python.keras.optimizers import Adam
import tensorflow._api.v1.keras.backend as K


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
        self.sess = K.get_session()
        self.policy_net = self.__build_net()

    def predict(self, state):
        # Need to add one extra dim
        state = state[np.newaxis, :]
        mean, log_std = self.policy_net.predict(state)
        std = tf.math.exp(log_std)
        # Perform reparameterization trick
        normal = tf.distributions.Normal(0.0, 1.0)
        z = normal.sample()
        action = tf.tanh(mean + std * z)
        return self.sess.run(action)

    def evaluate(self, state):
        mean, log_std = self.policy_net.predict(state)
        std = tf.math.exp(log_std)
        # Perform reparameterization trick
        normal = tf.distributions.Normal(0.0, 1.0)
        z = normal.sample()
        action = tf.tanh(mean + std * z)
        log_prob = \
            tf.distributions.Normal(mean, std).log_prob(mean + std * z) - tf.log(1 - tf.pow(action, 2) + self.epsilon)
        return self.sess.run(action), self.sess.run(log_prob)

    def learn(self, states, q_values):
        self.policy_net.train_on_batch([states], q_values)

    def __build_net(self):
        state_input = Input(shape=[self.state_dim])
        layer1 = Dense(400, activation='relu')(state_input)
        layer2 = Dense(300, activation='relu')(layer1)
        mean_output = Dense(self.action_dim, activation='tanh')(layer2)
        raw_log_std_output = Dense(self.action_dim, activation='linear')(layer2)
        log_std_output = Lambda(lambda x: K.clip(x, self.min_std, self.max_std))(raw_log_std_output)
        model = Model(state_input, [mean_output, log_std_output])
        model.compile(loss=self.__get_loss(), optimizer=Adam(lr=self.learning_rate))
        return model

    def __get_loss(self):

        # Here y_true is predicted new q value
        def loss(y_true, y_pred):
            mean = y_pred[0]
            log_std = y_pred[1]
            std = tf.math.exp(log_std)
            # Perform reparameterization trick
            normal = tf.distributions.Normal(0.0, 1.0)
            z = normal.sample()
            action = tf.tanh(mean + std * z)
            log_prob = \
                tf.distributions.Normal(mean, std).log_prob(mean + std * z) - tf.log(
                    1 - tf.pow(action, 2) + self.epsilon)
            loss_fuc = K.mean(log_prob - y_true)
            return loss_fuc
        return loss

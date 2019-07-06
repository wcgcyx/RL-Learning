import math
import tensorflow as tf
import tensorflow._api.v1.keras.backend as K
from tensorflow._api.v1.keras.models import Sequential
from tensorflow._api.v1.keras.layers import Dense
from tensorflow._api.v1.keras.initializers import RandomUniform


class Actor:

    def __init__(
            self,
            action_dim,
            state_dim,
            learning_rate,
            tau):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.tau = tau
        # Build network
        self.model = self.__build_net()
        self.target_model = self.__build_net()
        self.target_model.set_weights(self.model.get_weights())
        # Get optimizer
        self.optimizer = self.__get_optimizer()
        # Need to do a initialization to avoid errors
        K.get_session().run(tf.global_variables_initializer())

    def predict(self, state):
        return self.model.predict(state)

    def target_predict(self, state):
        return self.target_model.predict(state)

    def learn(self, states, action_grads):
        self.optimizer(inputs=[states, action_grads])

    def update_target(self):
        weights, target_weights = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def __build_net(self):
        model = Sequential()
        model.add(Dense(400, input_dim=self.state_dim, activation='relu',
                        kernel_initializer=RandomUniform(minval=-1.0 / math.sqrt(self.state_dim),
                                                         maxval=1.0 / math.sqrt(self.state_dim)),
                        bias_initializer=RandomUniform(minval=-1.0 / math.sqrt(self.state_dim),
                                                       maxval=1.0 / math.sqrt(self.state_dim))))
        model.add(Dense(300, activation='relu',
                        kernel_initializer=RandomUniform(minval=-1.0 / math.sqrt(400),
                                                         maxval=1.0 / math.sqrt(400)),
                        bias_initializer=RandomUniform(minval=-1.0 / math.sqrt(400),
                                                       maxval=1.0 / math.sqrt(400))))
        model.add(Dense(self.action_dim, activation='tanh',
                        kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                        bias_initializer=RandomUniform(minval=-3e-3, maxval=3e-3)))
        return model

    def __get_optimizer(self):
        action_grads = K.placeholder(shape=(None, self.action_dim))
        weight_grads = tf.gradients(self.model.output, self.model.trainable_weights, -action_grads)
        grads = zip(weight_grads, self.model.trainable_weights)
        return K.function(inputs=[self.model.input, action_grads], outputs=[],
                          updates=[tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)])

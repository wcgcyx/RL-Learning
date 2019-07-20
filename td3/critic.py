import math
import tensorflow._api.v1.keras.backend as K
from tensorflow._api.v1.keras.models import Model
from tensorflow._api.v1.keras.layers import Dense, Input, Concatenate, Flatten
from tensorflow._api.v1.keras.initializers import RandomUniform
from tensorflow._api.v1.keras.optimizers import Adam


class Critic:

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
        self.model_1 = self.__build_net()
        self.model_2 = self.__build_net()
        self.target_model_1 = self.__build_net()
        self.target_model_2 = self.__build_net()
        self.target_model_1.set_weights(self.model_1.get_weights())
        self.target_model_2.set_weights(self.model_2.get_weights())
        # Get action gradients calculator (From paper it uses gradient from critic network 1)
        self.action_grads = K.function(inputs=[self.model_1.input],
                                       outputs=[K.gradients(self.model_1.output, [self.model_1.input[1]])])

    def get_action_grads(self, states, actions):
        return self.action_grads([states, actions])

    def target_predict(self, state, action):
        return self.target_model_1.predict([state, action]), self.target_model_2.predict([state, action])

    def learn(self, states, actions, critic_target):
        self.model_1.train_on_batch([states, actions], critic_target)
        self.model_2.train_on_batch([states, actions], critic_target)

    def update_target(self):
        weights_1, target_weights_1, weights_2, target_weights_2 = \
            self.model_1.get_weights(), self.target_model_1.get_weights(), \
            self.model_2.get_weights(), self.target_model_2.get_weights()
        for i in range(len(weights_1)):
            target_weights_1[i] = self.tau * weights_1[i] + (1 - self.tau) * target_weights_1[i]
            target_weights_2[i] = self.tau * weights_2[i] + (1 - self.tau) * target_weights_2[i]
        self.target_model_1.set_weights(target_weights_1)
        self.target_model_1.set_weights(target_weights_2)

    def __build_net(self):
        state_input = Input(shape=[self.state_dim])
        action_input = Input(shape=[self.action_dim])
        combine_input = Concatenate(axis=-1)([state_input, action_input])
        layer1 = Dense(400, activation='relu',
                       kernel_initializer=RandomUniform(minval=-1.0 / math.sqrt(self.state_dim),
                                                        maxval=1.0 / math.sqrt(self.state_dim)),
                       bias_initializer=RandomUniform(minval=-1.0 / math.sqrt(self.state_dim),
                                                      maxval=1.0 / math.sqrt(self.state_dim)))(combine_input)
        layer2 = Dense(300, activation='relu',
                       kernel_initializer=RandomUniform(minval=-1.0 / math.sqrt(400),
                                                        maxval=1.0 / math.sqrt(400)),
                       bias_initializer=RandomUniform(minval=-1.0 / math.sqrt(400),
                                                      maxval=1.0 / math.sqrt(400)))(layer1)
        output = Dense(1, activation='linear',
                       kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                       bias_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))(layer2)
        model = Model([state_input, action_input], output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
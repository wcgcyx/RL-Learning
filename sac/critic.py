from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Concatenate
from tensorflow.python.keras.optimizers import Adam
import tensorflow._api.v1.keras.backend as K


class Critic:

    def __init__(
            self,
            action_dim,
            state_dim,
            learning_rate,
            tau):
        # Critic in SAC has four networks:
        # Two Q Network, a V Network and a Target-V Network
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.tau = tau
        # Build network
        self.q_1 = self.__build_q_net()
        self.q_2 = self.__build_q_net()
        self.v = self.__build_v_net()
        self.target_v = self.__build_v_net()
        self.target_v.set_weights(self.v.get_weights())
        # Get action gradients calculator (From open ai spinning up it uses gradient from critic q network 1)
        self.action_grads = K.function(inputs=[self.q_1.input],
                                       outputs=[K.gradients(self.q_1.output, [self.q_1.input[1]])])

    def get_action_grads(self, states, actions):
        return self.action_grads([states, actions])

    def predict_q(self, state, action):
        return self.q_1.predict([state, action]), self.q_2.predict([state, action])

    def predict_v(self, state):
        return self.v.predict([state])

    def predict_v_target(self, state):
        return self.target_v.predict([state])

    def learn_q(self, states, actions, q_targets):
        self.q_1.train_on_batch([states, actions], q_targets)
        self.q_2.train_on_batch([states, actions], q_targets)

    def learn_v(self, states, v_targets):
        self.v.train_on_batch([states], v_targets)

    def update_target_v(self):
        weights, target_weights = self.v.get_weights(), self.target_v.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_v.set_weights(target_weights)

    def __build_q_net(self):
        state_input = Input(shape=[self.state_dim])
        action_input = Input(shape=[self.action_dim])
        combine_input = Concatenate(axis=-1)([state_input, action_input])
        layer1 = Dense(400, activation='relu')(combine_input)
        layer2 = Dense(300, activation='relu')(layer1)
        output = Dense(1, activation='linear')(layer2)
        model = Model([state_input, action_input], output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def __build_v_net(self):
        state_input = Input(shape=[self.state_dim])
        layer1 = Dense(400, activation='relu')(state_input)
        layer2 = Dense(300, activation='relu')(layer1)
        output = Dense(1, activation='linear')(layer2)
        model = Model([state_input], output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

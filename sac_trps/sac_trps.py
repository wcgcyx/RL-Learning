import torch
import torch.nn as nn
import numpy as np
from .actor import Actor
from .critic import Critic
from .memory import ReplayBuffer
from torch.distributions import Normal

# Get device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Agent:

    def __init__(
            self,
            state_dim,
            action_dim,
            lr=3e-4,
            tr=0.01,
            discount=0.99,
            tau=1e-2,
            alpha=0.2,
            log_std_min=-20,
            log_std_max=2,
            memory_size=1000000,
            batch_size=128,
            debug_file=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tr = tr
        self.discount = discount
        self.alpha = alpha
        self.count = 0
        self.advantage = 0

        self.actor = Actor(state_dim, action_dim, lr, log_std_min, log_std_max)
        self.critic = Critic(state_dim, action_dim, lr, tau)

        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        if debug_file is None:
            self.debug_file = None
        else:
            self.debug_file = open(debug_file, "a")
            self.debug_file.write("KL,improvement\n")

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.actor.get_action(state)

    def store_transition(self, state, action, reward, next_state, end):
        self.memory.push(state, action, reward, next_state, end)

    def learn(self):
        state, action, reward, next_state, end = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        end = torch.FloatTensor(np.float32(end)).unsqueeze(1).to(device)

        # Training Q Networks
        predicted_q_value_1, predicted_q_value_2 = self.critic.predict_q(state, action)
        predicted_v_target = self.critic.predict_v_target(next_state)
        target_q_value = reward + (1 - end) * self.discount * predicted_v_target
        q_loss_1 = nn.MSELoss()(predicted_q_value_1, target_q_value.detach())
        q_loss_2 = nn.MSELoss()(predicted_q_value_2, target_q_value.detach())
        self.critic.learn_q(q_loss_1, q_loss_2)

        # Training V Network
        new_action, log_prob = self.actor.predict(state)
        predicted_new_q_value_1, predicted_new_q_value_2 = self.critic.predict_q(state, new_action)
        predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2)
        target_v_value = predicted_new_q_value - self.alpha * log_prob
        predicted_v_value = self.critic.predict_v(state)
        v_loss = nn.MSELoss()(predicted_v_value, target_v_value.detach())
        self.critic.learn_v(v_loss)

        # Training Policy Network

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        mean, log_std = self.actor.policy_net.forward(state)
        std = log_std.exp()
        old_action_raw = mean + std * z
        old_action = torch.tanh(old_action_raw)
        old_log_prob = Normal(mean, std).log_prob(old_action_raw) - torch.log(1 - old_action.pow(2) + 1e-6)
        old_log_prob = old_log_prob.sum(-1, keepdim=True)

        if self.count % 10 == 0:
            if self.debug_file is not None:
                old_reward = self.get_reward(state)

            mean = torch.nn.utils.parameters_to_vector(self.actor.policy_net.parameters()).detach()
            std = torch.FloatTensor([1e-5])
            iterations = 10
            N = 100
            Ne = 10
            t = 0
            trained = 0
            while t < iterations and std.max() > 1e-6:
                X = Normal(mean, std).sample((N, 1))
                S, valid = self.get_value(state, X, old_log_prob, old_action_raw, old_action)
                if not valid:
                    std /= 2
                else:
                    trained += 1
                    _, indices = torch.topk(S, k=Ne, dim=0)
                    indices = indices.flatten()
                    XNe = torch.index_select(X, dim=0, index=indices)
                    mean = XNe.mean(dim=0).squeeze(dim=0)
                    std = XNe.var(dim=0).sqrt().squeeze(dim=0)
                t += 1
            torch.nn.utils.vector_to_parameters(mean, self.actor.policy_net.parameters())

            if self.debug_file is not None:
                KL = self.get_KL(torch.nn.utils.parameters_to_vector(self.actor.policy_net.parameters()),
                                 old_log_prob, state, old_action_raw, old_action)
                new_reward = self.get_reward(state)
                self.debug_file.write("{},{}\n".format(abs(KL), new_reward - old_reward))
            self.advantage += new_reward - old_reward
            print(self.advantage, t, trained)
        self.count += 1
        # Updating Target-V Network
        self.critic.update_target_v()

    def get_value(self, state, x, old_log_prob, old_action_raw, old_action):
        result = torch.FloatTensor()
        for param in x:
            param = param.flatten()
            if self.get_KL(param, old_log_prob, state, old_action_raw, old_action) > self.tr:
                return None, False
            temp = self.get_single_value(state, param).reshape(1)
            result = torch.cat((result, temp), dim=0)
        result = result.unsqueeze(1)
        return result, True

    def get_single_value(self, state, params):
        same_z = Normal(0, 1).sample()
        torch.nn.utils.vector_to_parameters(params, self.actor.evaluate_net.parameters())
        mean, log_std = self.actor.evaluate_net.forward(state)
        std = log_std.exp()
        action_raw = mean + std * same_z
        action = torch.tanh(action_raw)
        log_prob = Normal(mean, std).log_prob(action_raw) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        predicted_new_q_value_1, predicted_new_q_value_2 = self.critic.predict_q(state, action)
        predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2)
        loss = (predicted_new_q_value - self.alpha * log_prob).mean()
        return loss.detach()

    def get_reward(self, state):
        same_z = Normal(0, 1).sample()
        mean, log_std = self.actor.policy_net.forward(state)
        std = log_std.exp()
        action_raw = mean + std * same_z
        action = torch.tanh(action_raw)
        log_prob = Normal(mean, std).log_prob(action_raw) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        predicted_new_q_value_1, predicted_new_q_value_2 = self.critic.predict_q(state, action)
        predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2)
        loss = (predicted_new_q_value - self.alpha * log_prob).mean()
        return loss.item()

    def get_KL(self, params, old_log_prob, state, old_action_raw, old_action):
        torch.nn.utils.vector_to_parameters(params, self.actor.evaluate_net.parameters())
        mean, log_std = self.actor.evaluate_net.forward(state)
        std = log_std.exp()
        new_log_prob = Normal(mean, std).log_prob(old_action_raw) - torch.log(1 - old_action.pow(2) + 1e-6)
        new_log_prob = new_log_prob.sum(-1, keepdim=True)
        KL = old_log_prob - new_log_prob
        KL = KL.mean()
        return KL.item()

    def save_weighs(self, task, identifier):
        torch.save(self.actor.policy_net.state_dict(), 'weights_history/' + task + '/sac_actor_policy_net_' + identifier + '.weights')
        torch.save(self.critic.q_net_1.state_dict(), 'weights_history/' + task + '/sac_critic_q_net_1_' + identifier + '.weights')
        torch.save(self.critic.q_net_2.state_dict(), 'weights_history/' + task + '/sac_critic_q_net_2_' + identifier + '.weights')
        torch.save(self.critic.v_net.state_dict(), 'weights_history/' + task + '/sac_critic_v_net_' + identifier + '.weights')
        torch.save(self.critic.target_v_net.state_dict(), 'weights_history/' + task + '/sac_critic_target_v_net_' + identifier + '.weights')

    def load_weights(self, task, identifier):
        self.actor.policy_net.load_state_dict(torch.load('weights_history/' + task + '/sac_actor_policy_net_' + identifier + '.weights'))
        self.critic.q_net_1.load_state_dict(torch.load('weights_history/' + task + '/sac_critic_q_net_1_' + identifier + '.weights'))
        self.critic.q_net_2.load_state_dict(torch.load('weights_history/' + task + '/sac_critic_q_net_2_' + identifier + '.weights'))
        self.critic.v_net.load_state_dict(torch.load('weights_history/' + task + '/sac_critic_v_net_' + identifier + '.weights'))
        self.critic.target_v_net.load_state_dict(torch.load('weights_history/' + task + '/sac_critic_target_v_net_' + identifier + '.weights'))

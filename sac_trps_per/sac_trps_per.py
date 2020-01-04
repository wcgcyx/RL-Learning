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
            N=100,
            Ne=5,
            t=10,
            size=0.33):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tr = tr
        self.discount = discount
        self.alpha = alpha
        self.count = 0
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.actor = Actor(state_dim, action_dim, lr, log_std_min, log_std_max)
        self.critic = Critic(state_dim, action_dim, lr, tau)

        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.N = N
        self.Ne = Ne
        self.t = t
        self.size = size

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.actor.get_action(state)

    def store_transition(self, state, action, reward, next_state, end):
        state_ = np.expand_dims(state, axis=0)
        state_ = torch.FloatTensor(state_).to(device)
        # Train policy Network
        old_mean, old_log_std = self.actor.policy_net.forward(state_)
        old_mean = old_mean.detach()
        old_log_std = old_log_std.detach()

        # Start cross-entropy search
        location = torch.cat((old_mean, old_log_std), dim=1)
        mean_scale = torch.zeros(1, self.action_dim).fill_(self.size).to(device)
        log_std_scale = torch.zeros(1, self.action_dim).fill_(self.size).to(device)
        scale = torch.cat((mean_scale, log_std_scale), dim=1)

        N = self.N
        Ne = self.Ne
        original_shape = N, 1
        compress_shape = N * 1
        states = state_.expand(N, 1, self.state_dim).reshape(compress_shape, self.state_dim)

        t = 0
        while t < self.t:
            t += 1
            X = self.sample(location, scale, N)
            S = self.get_value(X, states, original_shape, compress_shape)
            _, indices = torch.topk(S, k=Ne, dim=0)
            indices = indices.flatten()
            XNe = torch.index_select(X, dim=0, index=indices)
            location = XNe.mean(dim=0)
            scale = XNe.std(dim=0)

        len = location.shape[1] // 2
        target_mean, target_log_std = torch.split(location, len, dim=1)
        # self.actor.learn(state, target_mean, target_log_std)
        error = 0.5 * (old_mean - target_mean).pow(2) + 0.5 * (old_log_std - target_log_std).pow(2)
        error = error.item()

        self.memory.push(error, (state, action, reward, next_state, end))

    def learn(self):
        batch, idxs, is_weights = self.memory.sample(self.batch_size)
        state, action, reward, next_state, end = map(np.stack, zip(*batch))

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

        # Train policy Network
        old_mean, old_log_std = self.actor.policy_net.forward(state)
        old_mean = old_mean.detach()
        old_log_std = old_log_std.detach()

        # Start cross-entropy search
        location = torch.cat((old_mean, old_log_std), dim=1)
        mean_scale = torch.zeros(self.batch_size, self.action_dim).fill_(self.size).to(device)
        log_std_scale = torch.zeros(self.batch_size, self.action_dim).fill_(self.size).to(device)
        scale = torch.cat((mean_scale, log_std_scale), dim=1)

        N = self.N
        Ne = self.Ne
        original_shape = N, self.batch_size
        compress_shape = N * self.batch_size
        states = state.expand(N, self.batch_size, self.state_dim).reshape(compress_shape, self.state_dim)

        t = 0
        while t < self.t:
            t += 1
            X = self.sample(location, scale, N)
            S = self.get_value(X, states, original_shape, compress_shape)
            _, indices = torch.topk(S, k=Ne, dim=0)
            indices = indices.flatten()
            XNe = torch.index_select(X, dim=0, index=indices)
            location = XNe.mean(dim=0)
            scale = XNe.std(dim=0)

        len = location.shape[1] // 2
        target_mean, target_log_std = torch.split(location, len, dim=1)

        errors = 0.5 * (old_mean - target_mean).pow(2) + 0.5 * (old_log_std - target_log_std).pow(2)
        errors = errors.cpu().detach().numpy()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        self.actor.learn(state, target_mean, target_log_std)

        # Updating Target-V Network
        self.critic.update_target_v()

    def get_value(self, X, states, original_shape, compress_shape):
        len = X.shape[2] // 2
        mean, log_std = torch.split(X, len, dim=2)
        std = log_std.exp()
        noise = Normal(0, 1).sample()
        action_raw = mean + std * noise
        action = torch.tanh(action_raw)
        log_prob = Normal(mean, std).log_prob(action_raw) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        predicted_new_q_value_1, predicted_new_q_value_2 = self.critic.predict_q(states, action.reshape(compress_shape, self.action_dim))
        predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2).reshape(original_shape[0], original_shape[1], 1)
        loss = (predicted_new_q_value - self.alpha * log_prob).mean(dim=1)
        return loss.detach()

    def sample(self, location, scale, N):
        # Bound normal distribution by 3 std
        min = location - scale * 3
        max = location + scale * 3
        original_sample = torch.max(torch.min(Normal(location, scale).sample((N,)), max), min)
        len = original_sample.shape[2] // 2
        original_sample = torch.split(original_sample, len, dim=2)
        return torch.cat((original_sample[0], original_sample[1].clamp(self.log_std_min, self.log_std_max)), dim=2)

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

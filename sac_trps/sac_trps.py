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
            discount=0.99,
            tau=1e-2,
            alpha=0.2,
            log_std_min=-20,
            log_std_max=2,
            memory_size=1000000,
            batch_size=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.alpha = alpha

        self.actor = Actor(state_dim, action_dim, lr, log_std_min, log_std_max)
        self.critic = Critic(state_dim, action_dim, lr, tau)

        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size

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
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        mean, log_std = self.actor.policy_net.forward(state)
        std = log_std.exp()
        old_action_raw = mean + std * z
        old_action = torch.tanh(old_action_raw)
        old_log_prob = Normal(mean, std).log_prob(old_action_raw) - torch.log(1 - old_action.pow(2) + 1e-6)
        old_log_prob = old_log_prob.sum(-1, keepdim=True)

        params = torch.nn.utils.parameters_to_vector(self.actor.policy_net.parameters())
        iterations = 5
        search_size = torch.FloatTensor([0.005]).to(device)
        search_direction = torch.nn.utils.parameters_to_vector(torch.autograd.grad(policy_loss, self.actor.policy_net.parameters(), retain_graph=True))

        for i in range(iterations):
            test_params = params - search_direction * search_size
            KL = self.get_KL(test_params, old_log_prob, state, old_action_raw, old_action)
            if abs(KL) < 0.005:
                params = test_params
                torch.nn.utils.vector_to_parameters(params, self.actor.policy_net.parameters())
                # Compute new direction
                new_action, log_prob = self.actor.predict(state)
                predicted_new_q_value_1, predicted_new_q_value_2 = self.critic.predict_q(state, new_action)
                predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2)
                policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
                search_direction = torch.nn.utils.parameters_to_vector(torch.autograd.grad(policy_loss, self.actor.policy_net.parameters(), retain_graph=True))
            search_size /= 2

        # Updating Target-V Network
        self.critic.update_target_v()

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

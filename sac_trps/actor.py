import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

# Get device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Actor:

    def __init__(
            self,
            state_dim,
            action_dim,
            lr,
            log_std_min,
            log_std_max):
        # Build Networks
        # SAC has one network in critic
        # One Policy-Network
        self.policy_net = PolicyNetwork(state_dim, action_dim, log_std_min, log_std_max).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.evaluate_net = PolicyNetwork(state_dim, action_dim, log_std_min, log_std_max).to(device)

    def get_action(self, state):
        action, _ = self.policy_net.predict(state)
        return action.cpu()[0].detach()

    def predict(self, state):
        return self.policy_net.predict(state)

    def learn(self, state, target_mean, target_log_std):
        predicted_mean, predicted_log_std = self.policy_net.forward(state)
        predicted_dist = Normal(predicted_mean, predicted_log_std.exp())
        target_dist = Normal(target_mean, target_log_std.exp())
        policy_loss = kl_divergence(predicted_dist, target_dist).mean()
        # print(policy_loss)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, log_std_min, log_std_max):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)

        self.mean_layer3 = nn.Linear(300, action_dim)
        self.log_std_layer3 = nn.Linear(300, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        mean = self.mean_layer3(x)
        log_std = self.log_std_layer3(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def predict(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

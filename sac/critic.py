import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Get device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Critic:

    def __init__(
            self,
            state_dim,
            action_dim,
            lr,
            tau):
        self.tau = tau
        # Build Networks
        # SAC has four networks in critic
        # Two Q-Networks
        self.q_net_1 = QNetwork(state_dim, action_dim).to(device)
        self.q_net_2 = QNetwork(state_dim, action_dim).to(device)
        self.q_optimizer_1 = optim.Adam(self.q_net_1.parameters(), lr=lr)
        self.q_optimizer_2 = optim.Adam(self.q_net_2.parameters(), lr=lr)
        # One V-Network
        self.v_net = VNetwork(state_dim).to(device)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=lr)
        # One Target-V Network
        self.target_v_net = VNetwork(state_dim).to(device)
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(param.data)

    def predict_q(self, state, action):
        return self.q_net_1.forward(state, action), self.q_net_2.forward(state, action)

    def predict_v(self, state):
        return self.v_net.forward(state)

    def predict_v_target(self, state):
        return self.target_v_net.forward(state)

    def learn_q(self, q_loss_1, q_loss_2):
        self.q_optimizer_1.zero_grad()
        q_loss_1.backward()
        self.q_optimizer_1.step()
        self.q_optimizer_2.zero_grad()
        q_loss_2.backward()
        self.q_optimizer_2.step()

    def learn_v(self, v_loss):
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

    def update_target_v(self):
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class VNetwork(nn.Module):

    def __init__(self, state_dim):
        super(VNetwork, self).__init__()

        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

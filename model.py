import torch
import torch.nn as nn
import torch.nn.functional as F
'''
env_info = env.reset(train_mode=False)[brain_name]
state = env_info.vector_observations[0]
env_info = env.step(action)[brain_name]
next_state = env_info.vector_observations[0]
rewards =env_info.rewards[0]
done=env_info.local_dones[0]
'''



class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDQNetwork(nn.Module):
    """Architecture for Duelling DQN """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32,fcv_units = 64, fca_units = 64):
        super(DDQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fcv = nn.Linear(fc2_units,fcv_units)
        self.fca = nn.Linear(fc2_units,fca_units)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(fc1))
        value = F.relu(self.fcv(x))
        advantage = F.relu(self.fca(x))
        out = value + (advantage-advantage.mean())
        return out


import torch

class GradientNetwork(nn.Module):
    def __init__(self, state_dim, feature_size=64):
        super(GradientNetwork, self).__init__()
        self.l1 = nn.Linear(int(feature_size + state_dim), 100)
        self.l2 = nn.Linear(100, 100)
        # Grad output is same size as parameters (features)
        self.l3 = nn.Linear(100, int(feature_size))

    def forward(self, state, actor_params):
        x = torch.cat([state, actor_params], dim=1)
        y = torch.tanh(self.l1(x))
        y = torch.tanh(self.l2(y))
        grad_output = self.l3(y)

        return grad_output


class SigmoidPolicy(nn.Module):
    '''
    Simple Boltzmann policy with temperature = 1
    '''
    def __init__(self, num_states, num_actions, feature_size=64):
        super(SigmoidPolicy, self).__init__()
        self.fc1 = nn.Linear(num_states, num_actions, bias=False)
        # self.fc2 = nn.Linear(num_states, feature_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, one_hot_state):
        prob_a = self.softmax(self.fc1(one_hot_state))
        return prob_a

class Critic(nn.Module):
    '''
    Trying to implement tabular critic through NN
    '''
    def __init__(self, num_states, num_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_states, num_actions, bias=False)

    def forward(self, one_hot_state):
        q_s = self.fc1(one_hot_state)
        return q_s


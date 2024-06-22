import torch.nn as nn
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from rfc_utils.rfc_math import *
from rl_rfc.core.distributions import Categorical
from rl_rfc.core.policy import Policy


class PolicyDiscrete(Policy):
    def __init__(self, net, action_num, net_out_dim=None):
        super().__init__()
        self.type = 'discrete'
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.net = net
        self.action_head = nn.Linear(net_out_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        action_prob = torch.softmax(self.action_head(x), dim=1)
        return Categorical(probs=action_prob)

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}


import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.distributions import Categorical as TorchCategorical


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """This function should return a distribution to sample action from"""
        raise NotImplementedError


    def select_action(self, x, mean_action=False):
        dist = self.forward(x)
        if isinstance(mean_action, bool):  # Check if mean_action is a single boolean
            if mean_action:
                action = dist.mean_sample()
            else:
                action = dist.sample()
        
        else:
            actions = []
            for i, mean_act in enumerate(mean_action):
                if mean_act:
                    actions.append(dist.mean_sample()[i])
                else:
                    actions.append(dist.sample()[i])
            action = torch.stack(actions)
        
        return action
        

    def get_kl(self, x):
        dist = self.forward(x)
        return dist.kl()

    def get_log_prob(self, x, action):
        dist = self.forward(x)
        return dist.log_prob(action)
    
class PolicyGaussian(Policy):
    def __init__(self, net, action_dim, net_out_dim=None, log_std=0, fix_std=False):
        super().__init__()
        self.type = 'gaussian'
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.action_mean = nn.Linear(net_out_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std, requires_grad=not fix_std)

    def forward(self, x):
        x = self.net(x)
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return DiagGaussian(action_mean, action_std)

    def get_fim(self, x):
        dist = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), dist.loc, {'std_id': std_id, 'std_index': std_index}


class DiagGaussian(Normal):

    def __init__(self, loc, scale):
        super().__init__(loc, scale)

    def kl(self):
        loc1 = self.loc
        scale1 = self.scale
        log_scale1 = self.scale.log()
        loc0 = self.loc.detach()
        scale0 = self.scale.detach()
        log_scale0 = log_scale1.detach()
        kl = log_scale1 - log_scale0 + (scale0.pow(2) + (loc0 - loc1).pow(2)) / (2.0 * scale1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def log_prob(self, value):
        return super().log_prob(value).sum(1, keepdim=True)

    def mean_sample(self):
        return self.loc
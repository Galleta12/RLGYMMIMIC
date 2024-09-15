import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.distributions import Categorical as TorchCategorical




class PolicyGaussian(nn.Module):
    def __init__(self, net, action_dim, net_out_dim=None, log_std=0, fix_std=False):
        super().__init__()
        self.net = net
        
        if net_out_dim is None:
            net_out_dim = net.out_dim
        
        self.action_mean = nn.Linear(net_out_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std, requires_grad=not fix_std)
        
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        
        
    def forward(self, x):
        # mean of gaussian
        mean = self.action_mean(self.net(x))
        
        # Get the standard deviation by exponentiating the log standard deviation
        std = torch.exp(self.action_log_std).expand_as(mean)
        
        # normal distribution with the mean and std
        dist = Normal(mean, std)
        return dist

    def select_action(self, x, mean_action=False):
        # sample action from the diagonal gaussian
        dist = self.forward(x)
        action = dist.mean if mean_action else dist.sample()
        return action

    def get_log_prob_entropy(self, x, action):
        # Get the log probability of the action under the policy distribution
        dist = self.forward(x)
        return dist.log_prob(action).sum(dim=-1, keepdim=True), dist.entropy().sum(dim=-1,keepdim=True) 

    def get_kl(self, x):
        # KL divergence between the current policy and a fixed standard normal distribution
        dist = self.forward(x)
        kl = torch.distributions.kl_divergence(dist, Normal(torch.zeros_like(dist.mean), torch.ones_like(dist.stddev)))
        return kl.sum(dim=-1, keepdim=True)  # Summing KL divergence over action dimensions
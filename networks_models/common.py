

import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        
        #set the activtion layers
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        
        #the last dimension of the multilayer perceptron is the input for the
        #next neural networks
        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        
        last_dim = input_dim
        for nh in hidden_dims:
            #linear layers
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x



class Value(nn.Module):
    def __init__(self, net, net_out_dim=None):
        super().__init__()
        self.net = net
        if net_out_dim is None:
            #if this is the case is the same
            #as the last
            net_out_dim = net.out_dim
        self.value_head = nn.Linear(net_out_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        value = self.value_head(x)
        return value
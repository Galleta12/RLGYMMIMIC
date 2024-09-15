

import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        
        #set the activtion layers
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid()
        }

        self.activation = activations[activation]
        
        
        #populate the layers
        layers = []
        last_dim = input_dim
        
        for nh in hidden_dims:
            layers.append(nn.Linear(last_dim, nh))  # Add linear layer
            layers.append(activations[activation])  # Add corresponding activation
            last_dim = nh
        
        # Use nn.Sequential to create the model
        self.model = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]  # Last 

    def forward(self, x):
         
        #forward pass through the sequential layers
        return self.model(x)



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
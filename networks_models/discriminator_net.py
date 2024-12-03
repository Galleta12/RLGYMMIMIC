import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.distributions import Categorical as TorchCategorical
from torch import autograd

class Discriminator(nn.Module):
    def __init__(self, net,net_out_dim=None):
        super().__init__()
      

        self.net = net
        #print('net', net)
        if net_out_dim is None:
            net_out_dim = net.out_dim


        # Final linear layer for the discriminator output
        self.disc_output = nn.Linear(net_out_dim, 1)

        # Initialize layers
        self.disc_output.weight.data.mul_(0.1)
        self.disc_output.bias.data.mul_(0.0)

    def forward(self, x):
        # Forward pass through the network and output layer
        h = self.net(x)
        d = self.disc_output(h)
        return d

    
    
    

    def predict_reward(self, state, next_state):
        with torch.no_grad():
            #self.eval()  # Evaluation mode
            #print('state', state.device)
            #print('next state', next_state.device)
            # Concatenate states and compute discriminator output
            d = self.forward(torch.cat([state, next_state], dim=-1)).cpu()
            disc_reward =  torch.clamp(1 - 0.25 * torch.square(d - 1), min=0).cpu()

            
            #print("total device",total_reward.device)
            #self.train()  # Return to train mode
        
        return disc_reward.squeeze(), d
        #return d, d

    #gradient penaltiy cofficient 10 following amp paper
    def compute_grad_penalty(self, amp_state, amp_next_state, lambda_=10,device=None):
        # Concatenate expert and agent states, and enable gradient computation
        
        
        amp_state, amp_next_state = amp_state.to(device), amp_next_state.to(device)
        
     
        
        
        
        mixed_data = torch.cat([amp_state, amp_next_state], dim=-1).requires_grad_(True).to(device)

        # Forward pass
        d = self.forward(mixed_data).to(device)

        ones = torch.ones(d.size(), device=device)

        # Compute gradients
        grad = autograd.grad(outputs=d, inputs=mixed_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]
        # Gradient penalty term
        grad_penalty = (lambda_*0.5) * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_penalty

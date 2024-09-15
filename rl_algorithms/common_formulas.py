import torch
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.torch import batch_to

#GAE
def estimate_advantages(rewards, masks, values, gamma, gae_lambda):
    device = rewards.device
    rewards, masks, values = batch_to(torch.device('cpu'), rewards, masks, values)

    tensor_type = type(rewards)
    
    deltas = tensor_type(rewards.size(0), 1)
    
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    
    
    for i in reversed(range(rewards.size(0))):
       #multiply with mask that is terminated to avoid, returns when the episode ended
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        #calculate the advantage
        advantages[i] = deltas[i] + gamma * gae_lambda * prev_advantage * masks[i]
        
        #since the value and advantage tensor are just on dim
        #we can acces the value in a secure way in the following way
        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    
    
    returns = values + advantages
    
    #normalized advantage
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = batch_to(device, advantages, returns)
    return advantages, returns


import math
import numpy as np
from utils.torch import *
import time
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from rl_algorithms.agent_amp import AgentAMP
from rl_algorithms.common_formulas import estimate_advantages




class AmpAlg(AgentAMP):
    def __init__(self, clip_epsilon=0.2, mini_batch_size=64, use_mini_batch=False,
                 policy_grad_clip=None,gae_lambda=0.95, optimizer_policy=None, optimizer_value=None,
                 optimizer_disc=None,opt_num_epochs=1, value_opt_niter=1,entropy_coef= 0.01,value_loss_coef=1,**kwargs):
        super().__init__(**kwargs)
        self.gae_lambda = gae_lambda
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.optimizer_disc = optimizer_disc
        self.opt_num_epochs = opt_num_epochs
        self.value_opt_niter = value_opt_niter
        self.entropy_coef=entropy_coef
        self.value_loss_coef = value_loss_coef
        self.clip_epsilon = clip_epsilon
        self.mini_batch_size = mini_batch_size
        self.use_mini_batch = use_mini_batch
        self.policy_grad_clip = policy_grad_clip
    
    
    def update_policy(self, states, actions, returns, advantages, exps,amp_features,amp_next_features,amp_states,amp_next_states):
        
        """update policy"""
        #with to_test(*self.update_modules):
        self.policy_net.train(False)
        self.value_net.train(False)
        self.discriminator.train(False)
        # get the log probabilites of the current policy
        with torch.no_grad():
           
            old_log_probs, old_entropy = self.policy_net.get_log_prob_entropy(self.trans_policy(states), actions)

        
        
        #set it to train
        self.policy_net.train(True)
        self.value_net.train(True)
        self.discriminator.train(True)
        
        for _ in range(self.opt_num_epochs):
            #sample replay buffer
            if self.use_mini_batch:
                
                
                #shuffle shape 0 is the same for everything
                b_inds = np.arange(states.shape[0])
                np.random.shuffle(b_inds)
                
                #create a tensor, that will help us to access the shuffle indices
                #the shape will be the same as the batch size
                b_inds = torch.LongTensor(b_inds).to(self.device)
               
                
               # Get the data shuffled
                states, actions, returns, advantages, old_log_probs, exps, amp_features, amp_next_features, amp_states, amp_next_states = \
                    states[b_inds].clone(), actions[b_inds].clone(), returns[b_inds].clone(), advantages[b_inds].clone(), \
                    old_log_probs[b_inds].clone(), exps[b_inds].clone(), amp_features[b_inds].clone(), amp_next_features[b_inds].clone(), \
                    amp_states[b_inds].clone(), amp_next_states[b_inds].clone()

                optim_iter_num = int(math.floor(states.shape[0] / self.mini_batch_size))
                #print('optim_iter num', optim_iter_num)
                
                
                
                #batch size
                batch_size = states.shape[0]
                for start in range(0,batch_size,self.mini_batch_size):
                    
                    end = start + self.mini_batch_size
                    #get the minibatch indices
                    ind = b_inds[start:end]
                    
                    
                    # Select the data from the minibatch
                    states_b, actions_b, advantages_b, returns_b, old_log_probs_b, exps_b, amp_features_b, amp_next_features_b, amp_states_b, amp_next_states_b = \
                        states[ind], actions[ind], advantages[ind], returns[ind], old_log_probs[ind], exps[ind], \
                        amp_features[ind], amp_next_features[ind], amp_states[ind], amp_next_states[ind]
                
                    
                    #filter out invalid experience
                    #so if it is 1 it will return all the ind
                    ind = exps_b.nonzero(as_tuple=False).squeeze(1)
                    
                    #critic loss
                    critic_loss = self.critic_loss(states_b, returns_b)
                    
                    #surrogate loss 
                    surr_loss, entropy_regularization =self.actor_loss_entropy(states_b, actions_b, advantages_b, old_log_probs_b, ind)            
                    
                    
                    
               
                    # Discriminator loss for AMP features
                    policy_d = self.discriminator(torch.cat([amp_features_b, amp_next_features_b], dim=-1))
                    amp_d = self.discriminator(torch.cat([amp_states_b, amp_next_states_b], dim=-1))
                    # Expert loss for discriminator (pushes expert output towards 1)
                    expert_loss = torch.nn.MSELoss()(amp_d, torch.ones(amp_d.size(), device=self.device))

                    # Policy loss for discriminator (pushes policy output towards -1)
                    policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))

                    # Total discriminator loss
                    amp_loss = (expert_loss + policy_loss)
                    
                    # Compute gradient penalty for the discriminator
                    grad_penalty = self.discriminator.compute_grad_penalty(amp_states_b, amp_next_states_b, 10,self.device)
                    #print("grad penalty", grad_penalty.device)
                    #total_amp_loss = amp_loss+ grad_penalty
                    
                    self.update_networks(critic_loss,surr_loss,entropy_regularization,amp_loss,grad_penalty)
                    
                    
            else:
                print('error with the min batch(batch) and mini bathc')
    

    
    
    def update_networks(self,critic_loss,surr_loss,entropy_regularization,amp_loss,grad_penalty):
        #minimize the policy loss and the value loss and max the entropy
        #entropy is the measure of chaos on an action probablity distribuion, in theory maximize entropy encourage agent to explore more
        #this is the total loss function
        
        
        
        
        # print("critic_loss device:", critic_loss.device)
        # print("surr_loss device:", surr_loss.device)
        # print("entropy_regularization device:", entropy_regularization.device)
        # print("total_amp_loss device:", total_amp_loss.device)
        
        
        
        total_loss = surr_loss - entropy_regularization + (self.value_loss_coef*critic_loss) + amp_loss +grad_penalty
        #print("total_loss device:", total_loss.device)
        #be sure that we dont accumulate the gradient on each iter
        #so it is clear the gradient of the prev batch
        
        #print('zero grads')
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        self.optimizer_disc.zero_grad()
        #backpropagation
        total_loss.backward()
        #print('backward')
        
        
        self.clip_policy_grad()
        
        
        #print('value')
        self.optimizer_value.step()
        #print('value done')
        
        # critic_loss.backward()
        #update the parameters based on the gradient
    
        #surr_loss.backward()
        #clip policy gradient
        #update it
        self.optimizer_policy.step()
        self.optimizer_disc.step()
    

    
    
    
    
    def critic_loss(self, states, returns):
        
        # #this is one the value opt niter
        # for _ in range(self.value_opt_niter):
        #forward pass
        values_pred = self.value_net(self.trans_value(states))
        #means sqare error
        value_loss = (values_pred - returns).pow(2).mean()
        
        return value_loss
        
    
    def actor_loss_entropy(self,states_b, actions_b, advantages_b, old_log_probs_b, ind):
        #clip  surrogate loss
        surr_loss ,entropy_regularization= self.ppo_loss(states_b, actions_b, advantages_b, old_log_probs_b, ind)
        return surr_loss, entropy_regularization
        
    #clipp the policy net parameters
    def clip_policy_grad(self):
        #clip the policy with 40
        if self.policy_grad_clip is not None:
            for params, max_norm in self.policy_grad_clip:
                torch.nn.utils.clip_grad_norm_(params, max_norm)
    
    
    
    def ppo_loss(self, states, actions, advantages, old_log_probs_b, ind):
        #get the log probabilities
        log_probs, entropy = self.policy_net.get_log_prob_entropy(self.trans_policy(states)[ind], actions[ind])
        #fixed log prob is the old log prob ration
        log_ratio = log_probs - old_log_probs_b[ind]
        
        ratio = torch.exp(log_ratio)
        
        #get the current advantage
        advantages = advantages[ind]
        #clip 1 unclip objective
        surr1 = ratio * advantages
        #clip 2 cliiped objective
        surr2 = advantages* torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) 
        
        #We invert the sing since torch gradient descent but we want to maximizs
        #surrogate loss function
        surr_loss = -torch.min(surr1, surr2).mean()
        
        #entropy regularization
        entropy_regularization = self.entropy_coef * entropy.mean()
        
        return surr_loss, entropy_regularization

    
    def update_params(self, batch):
        t0 = time.time()
        
        #to_train(*self.update_modules)
        
        # #set test mode
        # self.policy_net.train(True)
        # self.value_net.train(True)
        
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        terminates = torch.from_numpy(batch.terminates).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        
        amp_features = torch.from_numpy(batch.amp_features).to(self.dtype).to(self.device)
        amp_next_features = torch.from_numpy(batch.amp_next_features).to(self.dtype).to(self.device)
        amp_states = torch.from_numpy(batch.amp_states).to(self.dtype).to(self.device)
        amp_next_states = torch.from_numpy(batch.amp_next_states).to(self.dtype).to(self.device)
        #with to_test(*self.update_modules):
       
        
        self.policy_net.train(False)
        self.value_net.train(False)
        self.discriminator.train(False)
        #get the value net
        #this is imporant since this will be the value function for computing GAES
        with torch.no_grad():
            #forward pass
            values = self.value_net(self.trans_value(states))
        #self.policy_net.train(True)
        #self.value_net.train(True)
        
        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, terminates, values, self.gamma, self.gae_lambda)

        self.update_policy(states, actions, returns, advantages, exps,amp_features,amp_next_features,amp_states,amp_next_states)

        return time.time() - t0

    # def update_params_replay(self, _states, _actions,  _rewards, _terminates, _exps,_amp_features,
    #                          _amp_next_features,_amp_states,_amp_next_states):
    #     t0 = time.time()
        
    #     #to_train(*self.update_modules)
        
    #     # #set test mode
    #     # self.policy_net.train(True)
    #     # self.value_net.train(True)
        
    #     states = torch.from_numpy(_states).to(self.dtype).to(self.device)
    #     actions = torch.from_numpy(_actions).to(self.dtype).to(self.device)
    #     rewards = torch.from_numpy(_rewards).to(self.dtype).to(self.device)
    #     terminates = torch.from_numpy(_terminates).to(self.dtype).to(self.device)
    #     exps = torch.from_numpy(_exps).to(self.dtype).to(self.device)
        
    #     amp_features = torch.from_numpy(_amp_features).to(self.dtype).to(self.device)
    #     amp_next_features = torch.from_numpy(_amp_next_features).to(self.dtype).to(self.device)
    #     amp_states = torch.from_numpy(_amp_states).to(self.dtype).to(self.device)
    #     amp_next_states = torch.from_numpy(_amp_next_states).to(self.dtype).to(self.device)
    #     #with to_test(*self.update_modules):
       
        
    #     self.policy_net.train(False)
    #     self.value_net.train(False)
    #     self.discriminator.train(False)
    #     #get the value net
    #     #this is imporant since this will be the value function for computing GAES
    #     with torch.no_grad():
    #         #forward pass
    #         values = self.value_net(self.trans_value(states))
    #     #self.policy_net.train(True)
    #     #self.value_net.train(True)
        
    #     """get advantage estimation from the trajectories"""
    #     advantages, returns = estimate_advantages(rewards, terminates, values, self.gamma, self.gae_lambda)

    #     self.update_policy(states, actions, returns, advantages, exps,amp_features,amp_next_features,amp_states,amp_next_states)

    #     return time.time() - t0


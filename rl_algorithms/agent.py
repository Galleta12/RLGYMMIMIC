import multiprocessing
import sys
import os
import torch
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import math
import time
import os
from rl_algorithms.logger_rl import LoggerRL
from utils.torch import *
from utils.memory import MemoryManager,TrajBatch
os.environ["OMP_NUM_THREADS"] = "1"


class Agent:
    def __init__(self, env, policy_net, value_net, dtype, device, gamma, custom_reward=None,
                 mean_action=False, running_state=None, 
                 num_envs=1, n_steps=1, batch_size=2048):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.device = device
        self.gamma = gamma
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        #num of envs
        self.noise_rate = 1.0
        # self.traj_cls = TrajBatch
        self.sample_modules = [policy_net]
        self.update_modules = [policy_net, value_net]
    
        self.num_timesteps = n_steps
        
        self.batch_size = batch_size
        
        self.num_envs = num_envs
    
        self.step =0
    
    def collect_rollout(self,memory):
        states, infos, norm_states = self.env.reset()
        
        for _ in range(0,self.num_timesteps+1):
            #select an actino from the policy
            actions,mean_actions = self.select_actions(norm_states)
            #be sure action is float 64
            actions = actions.astype(np.float64)
            #step on the vec env
            next_states, rewards_env, terminateds, truncateds, infos,norm_next_states, loggers = self.env.step(actions)
            
            rewards = rewards_env
            mask = np.where(terminateds, 0, 1)
            exp = 1 - mean_actions            
            #save the normalized states
            #prediciton/actions is not defined I will change this later
            memory.append(norm_states, actions, rewards, actions, terminateds, truncateds, norm_next_states,mask, exp, infos)
             
            
            norm_states = norm_next_states
            
            #print(norm_states.shape)
            
            for index in memory.done_indices():
                #print('index', index)
                
                env_memory = memory[index]
                states, actions, rewards, old_probs, dones, truncateds, next_state, masks, exps,infos = env_memory.get()
                #set the logger on the multi vec env
                logger = self.env.end_episode_log(index) 
                #print('num_episodes', logger.num_episodes)
                #print('num_steps', logger.num_steps)
                states, _, norm_states[index] = self.env.reset(index)

            
            
            self.step += self.num_envs
        
                            
    def trans_policy(self, states):
        """transform states before going into policy net"""
        #print('trans policy')
        return states
    
    #rollout
    def sample(self):
       
        t_start = time.time()
        to_test(*self.sample_modules)
        memory_manager = MemoryManager(num_envs=self.num_envs)
        
        with to_cpu(*self.sample_modules):
            with torch.no_grad(): 
                while self.step < self.batch_size:
                    self.collect_rollout(memory_manager)
                    #print('step end',self.step)
        
        sample_time = time.time() - t_start
        loggers = self.env.log_sample()
        logger = LoggerRL.merge(loggers)                                
        logger.sample_time = sample_time
        traj_batch = TrajBatch(memory_manager)
        #reset
        self.step = 0
        
        return traj_batch,logger
          
    def select_actions(self, states):
        state_vars = torch.tensor(states)
        trans_out = self.trans_policy(state_vars)
        
        # Generate mean_action flags for each environment
        if self.mean_action:
            mean_actions = np.ones(self.num_envs)
        else:
            #print('new noise rate', no)
            mean_actions = self.env.mean_actions(self.noise_rate)
            #print('mean action vec', mean_actions)
        #print('mean actions', mean_actions)
        
        actions = self.policy_net.select_action(trans_out, mean_action=mean_actions)
        return actions.cpu().numpy(), mean_actions
                    
                    
    def set_noise_rate(self, noise_rate):
        self.noise_rate = noise_rate

    def trans_value(self, states):
        """transform states before going into value net"""
        return states
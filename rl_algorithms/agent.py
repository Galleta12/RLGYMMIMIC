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
from rl_algorithms.replaybuffer import ReplayBuffer
from utils.torch import *
#from utils.memory import MemoryManager,TrajBatch
from rl_rfc.core import TrajBatch
from rfc_utils.memory import Memory
os.environ["OMP_NUM_THREADS"] = "1"


class Agent:

    def __init__(self, env, policy_net, value_net, dtype, device, gamma, custom_reward=None,
                 end_reward=True, mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.device = device
        self.gamma = gamma
        self.custom_reward = custom_reward
        self.end_reward = end_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads
        self.noise_rate = 1.0
        self.traj_cls = TrajBatch
        self.logger_cls = LoggerRL
        self.sample_modules = [policy_net]
        self.update_modules = [policy_net, value_net]
      
     

    def sample_worker(self, pid, queue, min_batch_size):
        torch.randn(pid)
        if hasattr(self.env, 'np_random'):
            #self.env.np_random.rand(pid)
            #self.env.np_random.seed(pid)
            #random_value = self.env.np_random.random()
            self.env.np_random.random()
        
       
        memory = Memory()
        logger = self.logger_cls()

        while logger.num_steps < min_batch_size:
            state , _ = self.env.reset()
            if self.running_state is not None:
                state = self.running_state.normalize(state)
            logger.start_episode(self.env)
           

            for t in range(10000):
                state_var = tensor(state).unsqueeze(0)
                trans_out = self.trans_policy(state_var)
                #mean_action = self.mean_action or self.env.np_random.binomial(1, 1 - self.noise_rate)
                mean_action = False
                if mean_action == True or mean_action == 1:
                    print('mean action new', mean_action)
                
                action = self.policy_net.select_action(trans_out, mean_action)[0]
                
                action = action.numpy()
                action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)
                next_state, env_reward, done,_,info = self.env.step(action)
                
                
                if self.running_state is not None:
                    #print('running state')
                    next_state = self.running_state.normalize(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    #print('cusrom reward')
                    
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    #get the pose error aswell
                    pose_error = self.env.pose_error()
                    
                    c_info = np.append(c_info,[pose_error])
                    
                    
                    reward = c_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                terminated = 0 if done else 1
                exp = 1 - mean_action
                
               
                
                self.push_memory(memory, state, action, terminated, next_state, reward, exp)
                
                
                
                if pid == 0 and self.render:
                    self.env.render()
                if done:
                    break
                state = next_state

            logger.end_episode(self.env)
            #print('print num epiode', logger.num_episodes)
            #print('print num stpes', logger.num_steps)
            
        logger.end_sampling()
        #print('print num epiode', logger.avg_episode_len)

        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger



        
    def push_memory(self, memory, state, action, terminated, next_state, reward, exp):
        memory.push(state, action, terminated, next_state, reward, exp)


    def sample(self, min_batch_size):
        t_start = time.time()
        
        #to_test(*self.sample_modules)
        #set to test
        self.policy_net.train(False)
        
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                #this is the batch size
                thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
                #print('thread batch size', thread_batch_size)
                queue = multiprocessing.Queue()
                memories = [None] * self.num_threads
                loggers = [None] * self.num_threads
                for i in range(self.num_threads-1):
                    worker_args = (i+1, queue, thread_batch_size)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                
                memories[0], loggers[0] = self.sample_worker(0, None, thread_batch_size)
                for i in range(self.num_threads - 1):
                    pid, worker_memory, worker_logger = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                
                
           
                traj_batch = self.traj_cls(memories)
                
                logger = self.logger_cls.merge(loggers)
                
                
 
                    
            
            logger.sample_time = time.time() - t_start
          
            
            return traj_batch, logger

    

    
    
    
    def trans_policy(self, states):
        """transform states before going into policy net"""
        #print('trans policy')
        return states

    def trans_value(self, states):
        """transform states before going into value net"""
        return states

    def set_noise_rate(self, noise_rate):
        self.noise_rate = noise_rate

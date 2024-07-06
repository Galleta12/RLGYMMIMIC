#got it from https://github.com/pythonlessons/RockRL

import numpy as np
import multiprocessing as mp
import torch
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.zfilter import ZFilter
from rl_algorithms.logger_rl import LoggerRL
def run_env(conn, env_object, kwargs,pid,custom_reward_func=None,running_state=None,seed=None):
    #seed for each thread env
    
    
    torch.randn(pid)
    env = env_object(**kwargs)
    if seed:
        env.seed(seed)
    env.np_random.random()
    logger = LoggerRL()
    
    while True:
        # Wait for a message on the connection
        message = conn[1].recv()

        if isinstance(message, str):
            if message == 'reset':
                state, info = env.reset()
                #start episode on the logger
                logger.start_episode(env)
                
                norm_state = running_state(state) if running_state else state
                
                conn[1].send((state, info,norm_state))
            elif message == 'close':
                env.close()
                break
            elif message == 'render':
                results = env.render()
                results = results if results is not None else []
                conn[1].send(results)
            
            elif message.startswith('set_noise_rate'):
                _, noise_rate = message.split(':')
                noise_rate = float(noise_rate)
                mean_action = env.np_random.binomial(1, 1 - noise_rate)
                
                conn[1].send(mean_action)
            
            
            elif message =='end_log':
                logger.end_episode(env)
                conn[1].send(logger)
            
            elif message == 'log_sample':
                logger.end_sampling()
                conn[1].send(logger)
    
        else:
            # Assume it's an action
            action = message
            # state, reward, done, info = env.step(action)[:4]
            next_state, env_reward, terminated, truncated, info = env.step(action)
            norm_next_state = running_state(next_state) if running_state else next_state
            
            if custom_reward_func is not None:
                c_reward, c_info = custom_reward_func(env, state, action, info)
                reward = c_reward
                logger.step(env, env_reward, c_reward, c_info, info)
            else:
                reward = env_reward
                logger.step(env, env_reward, env_reward, 0, info)
            
            
            
            conn[1].send((next_state, reward, terminated, truncated, info,norm_next_state,logger))


class VectorizedEnv:
    def __init__(self, env_object, num_envs: int=2, custom_reward=None,seed=1,**kwargs):
        self.env_object = env_object
        self.kwargs = kwargs
        self.env = self.env_object(**self.kwargs)  # Instantiate the environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        #self._max_episode_steps = self.env._max_episode_steps
        #create an env and close it
        self.running_state = ZFilter((self.env.observation_space.shape[0],), clip=5)
        self.env.close()
        self.seed = seed
        self.custom_reward_func = custom_reward
            
        self.conns = [mp.Pipe() for _ in range(num_envs)]
        self.envs = [mp.Process(target=run_env, args=(conn, env_object, self.kwargs, 
                                                      i,self.custom_reward_func,
                                                      self.running_state,self.seed)) for i, conn in enumerate(self.conns)]
        
        for env in self.envs:
            env.start()
            
        

    def reset(self, index=None): # return states
        if index is None:
            for conn in self.conns:
                conn[0].send('reset')
            states, infos,norm_states = zip(*[conn[0].recv() for conn in self.conns])
            return np.array(states), list(infos),np.array(norm_states)
        
        else:
            self.conns[index][0].send('reset')
            state, info,norm_state = self.conns[index][0].recv()
            return state, info,norm_state

    def step(self, actions):
        for conn, action in zip(self.conns, actions):
            conn[0].send(action)

        results = [conn[0].recv() for conn in self.conns]
        next_states, rewards, terminateds, truncateds,infos,norm_next_states,loggers = zip(*results)
        
        return np.array(next_states), np.array(rewards), np.array(terminateds), np.array(truncateds), list(infos),np.array(norm_next_states),list(loggers)
    
    def render(self, index=None):
        if index is None:
            for conn in self.conns:
                conn[0].send('render')
            results = [conn[0].recv() for conn in self.conns]
        else:
            self.conns[index][0].send('render')
            results = self.conns[index][0].recv()

        return results

    def mean_actions(self, noise_rate):
        message = f'set_noise_rate:{noise_rate}'
        for conn in self.conns:
            conn[0].send(message)
        
        mean_actions = [conn[0].recv() for conn in self.conns]
        return np.array(mean_actions)
    
    def log_sample(self):
        for conn in self.conns:
            conn[0].send('log_sample')
        
        logs = [conn[0].recv() for conn in self.conns]
        return list(logs)

    def end_episode_log(self,index):
        self.conns[index][0].send('end_log')
        logger = self.conns[index][0].recv()
        return logger

    
    
    
    def close(self):
        for conn, env in zip(self.conns, self.envs):
            conn[0].send('close')
            env.join()
            env.close()
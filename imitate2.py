import argparse
import os
import sys
import pickle
import time
import datetime
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.uniform import Uniform
from torch.distributions import Normal
sys.path.append(os.getcwd())
from rfc_utils.config import Config
from agent_envs.humanoid_env2 import HumanoidTemplate
from torch.utils.tensorboard import SummaryWriter
from utils.vectorizedEnv import VectorizedEnv
from networks_models.common import MLP,Value
from networks_models.policy_net import PolicyGaussian
from utils.torch import *
from rl_algorithms.ppo import PPO
from reward_function import reward_func
from utils.zfilter import ZFilter
from rfc_utils.logger import create_logger


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--num_envs', type=int, default=20)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--show_noise', action='store_true', default=False)
args = parser.parse_args()
if args.render:
    args.num_threads = 1
cfg = Config(args.cfg, args.test, create_dirs=not (args.render or args.iter > 0))

tb_logger = SummaryWriter(cfg.tb_dir) if not args.render else None
logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'), file_handle=not args.render)


"""environment"""
# env = HumanoidTemplate(cfg)
# env.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)
print('device', device)


expert_reward = reward_func[cfg.reward_id]
print('exper reward', expert_reward)

# def test():
#     env_vec = VectorizedEnv(env_object=HumanoidTemplate,num_envs=args.num_envs,cfg=cfg)
#     print("obs space", env_vec.observation_space)
#     print('action space', env_vec.action_space)
    
#     states, infos = env_vec.reset()
#     print("Reset states:", states)
#     print("Reset infos:", infos)

    
    
#     # Run the environment for 1 steps
#     for step in range(1):
#         print(f"\nStep {step + 1}:")
        
#         # Sample actions
#         actions = [env_vec.action_space.sample() for _ in range(args.num_envs)]
#         print("Actions:", actions)
        
#         # Take a step in the environment
#         next_states, rewards, terminateds, truncateds, infos = env_vec.step(actions)
#         print("Next states:", next_states)
#         print("Rewards:", rewards)
#         print("Terminateds:", terminateds)
#         print("Truncateds:", truncateds)
#         print("Infos:", infos)
        
#         # Check if any of the environments are done
#         if any(terminateds) or any(truncateds):
#             break

#     # Close the environment
#     env_vec.close()
#     print("Test complete")


def pre_iter_update(i_iter,agent,optimizer_policy,policy_net):
    cfg.update_adaptive_params(i_iter)
    agent.set_noise_rate(cfg.adp_noise_rate)

    # print('new noise rate', agent.noise_rate)
    # print('adp log std', cfg.adp_log_std)
    # print('cfg policy lr', cfg.adp_policy_lr)
        
    set_optimizer_lr(optimizer_policy, cfg.adp_policy_lr)
    if cfg.fix_std:
        policy_net.action_log_std.fill_(cfg.adp_log_std)
    return
def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return str(datetime.timedelta(seconds=round(eta)))

def create_nets(inputs,output):
    #define networks
    actor_net =  PolicyGaussian(MLP(inputs, cfg.policy_hsize, cfg.policy_htype), output, log_std=cfg.log_std, fix_std=cfg.fix_std)
    critic_net = Value(MLP(inputs, cfg.value_hsize, cfg.value_htype))

    to_device(device, actor_net, critic_net)
    
    print('actor net', actor_net)
    print('cricic net', critic_net)
    if cfg.policy_optimizer == 'Adam':
        optimizer_policy = torch.optim.Adam(actor_net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
    else:
        optimizer_policy = torch.optim.SGD(actor_net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
    if cfg.value_optimizer == 'Adam':
        optimizer_value = torch.optim.Adam(critic_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
    else:
        optimizer_value = torch.optim.SGD(critic_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)
    print('opt actor', optimizer_policy)
    print('opt critic', optimizer_value)
    
    
    
    
    return actor_net,critic_net,optimizer_policy,optimizer_value

def train():
    env_vec = VectorizedEnv(env_object=HumanoidTemplate,num_envs=args.num_envs,custom_reward=expert_reward,cfg=cfg)
    print("obs space", env_vec.observation_space)
    print('action space', env_vec.action_space)
    print("args iter", args.iter)
    print("max_iter num:", cfg.max_iter_num)
    print('min batch size', cfg.min_batch_size)
    print('mini batch size', cfg.mini_batch_size)
    print('cons len', len(env_vec.conns))
    
    actor_net,critic_net,opt_policy,opt_value=create_nets(env_vec.observation_space.shape[0],env_vec.action_space.shape[0])
    
    
    
    
    agent = PPO(env= env_vec, dtype=dtype, device=device, 
                running_state=env_vec.running_state,
                 custom_reward=expert_reward, 
                 mean_action=False,
                 policy_net=actor_net, 
                 value_net=critic_net,
                 optimizer_policy=opt_policy, 
                 optimizer_value=opt_value, 
                 opt_num_epochs=cfg.num_optim_epoch,
                 gamma=cfg.gamma, 
                 tau=cfg.tau, clip_epsilon=cfg.clip_epsilon,
                 policy_grad_clip=[(actor_net.parameters(), 40)], 
                 use_mini_batch=cfg.mini_batch_size < cfg.min_batch_size, 
                 mini_batch_size=cfg.mini_batch_size,
                 batch_size=cfg.min_batch_size,
                 num_envs=len(env_vec.conns),
                 n_steps= 4166)
                 #n_steps= 1)
    
    
    print('batch size', agent.batch_size)
    print('num envs', agent.num_envs)
    print('opt_num_epochs', agent.opt_num_epochs)
    print('mini batch', agent.use_mini_batch)
    
    
    
    
    #for i_iter in range(args.iter,1):
    for i_iter in range(args.iter,cfg.max_iter_num):
        
        pre_iter_update(i_iter,agent,opt_policy,actor_net)
        
        #rollout?
        batch,log = agent.sample()
        
        print('batch shapes:', batch.get_shapes())
        
        """update networks"""
        t0 = time.time()
        agent.update_params(batch)
        t1 = time.time()
        
        """logging"""
        c_info = log.avg_c_info
        logger.info(
            '{}\tT_sample {:.2f}\tT_update {:.2f}\tETA {}\texpert_R_avg {:.4f} {}'
            '\texpert_R_range ({:.4f}, {:.4f})\teps_len {:.2f}'
            .format(i_iter, log.sample_time, t1 - t0, get_eta_str(i_iter, cfg.max_iter_num, t1 - t0 + log.sample_time), log.avg_c_reward,
                    np.array2string(c_info, formatter={'all': lambda x: '%.4f' % x}, separator=','),
                    log.min_c_reward, log.max_c_reward, log.avg_episode_len))
        
        tb_logger.add_scalar('total_reward', log.avg_c_reward, i_iter)
        tb_logger.add_scalar('episode_len', log.avg_episode_reward, i_iter)
        for i in range(c_info.shape[0]):
            tb_logger.add_scalar('reward_%d' % i, c_info[i], i_iter)
            tb_logger.add_scalar('eps_reward_%d' % i, log.avg_episode_c_info[i], i_iter)
        if cfg.save_model_interval > 0 and (i_iter+1) % cfg.save_model_interval == 0:
            tb_logger.flush()
            with to_cpu(actor_net, critic_net):
                cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_iter + 1)
                model_cp = {'policy_dict': actor_net.state_dict(),
                            'value_dict': critic_net.state_dict(),
                            'running_state': env_vec.running_state}
                pickle.dump(model_cp, open(cp_path, 'wb'))
        """clean up gpu memory"""
        torch.cuda.empty_cache()

    env_vec.close()
    logger.info('training done!')




#pre iter update?




if __name__ == "__main__":

    #test()
    train()
    # print("obs space", env.observation_space)
    # print('action space', env.action_space)













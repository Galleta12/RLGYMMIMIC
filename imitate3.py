import argparse
import os
import sys
import pickle
import time
import datetime
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
parser.add_argument('--num_threads', type=int, default=20)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--show_noise', action='store_true', default=False)
args = parser.parse_args()
if args.render:
    args.num_threads = 1
cfg = Config(args.cfg, args.test, create_dirs=not (args.render or args.iter > 0))

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
print('device', device)



if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
tb_logger = SummaryWriter(cfg.tb_dir) if not args.render else None
logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'), file_handle=not args.render)
"""environment"""
env = HumanoidTemplate(cfg)
env.seed(cfg.seed)
#actuators = env.model.actuator_names
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
running_state = ZFilter((state_dim,), clip=5)

"""define actor and critic"""
policy_net = PolicyGaussian(MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
if args.iter > 0:
    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
    logger.info('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    policy_net.load_state_dict(model_cp['policy_dict'])
    value_net.load_state_dict(model_cp['value_dict'])
    running_state = model_cp['running_state']





#to_device(device, policy_net, value_net)
policy_net.to(device)
value_net.to(device)



if cfg.policy_optimizer == 'Adam':
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
else:
    optimizer_policy = torch.optim.SGD(policy_net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
if cfg.value_optimizer == 'Adam':
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
else:
    optimizer_value = torch.optim.SGD(value_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)

# reward functions
expert_reward = reward_func[cfg.reward_id]


#print('xd',args.render and not args.show_noise)


"""create agent"""
agent = PPO(env=env, dtype=dtype, device=device, running_state=running_state,
                 custom_reward=expert_reward, mean_action=args.render and not args.show_noise,
                 render=args.render, num_threads=args.num_threads,
                 policy_net=policy_net, value_net=value_net,
                 optimizer_policy=optimizer_policy, optimizer_value=optimizer_value, opt_num_epochs=cfg.num_optim_epoch,
                 gamma=cfg.gamma, tau=cfg.tau, clip_epsilon=cfg.clip_epsilon,
                 policy_grad_clip=[(policy_net.parameters(), 40)], end_reward=cfg.end_reward,
                 use_mini_batch=cfg.mini_batch_size < cfg.min_batch_size, mini_batch_size=cfg.mini_batch_size)

print('opt_num_epochs', agent.opt_num_epochs)
print('mini batch', agent.use_mini_batch)
print('mean action', agent.mean_action)


def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return str(datetime.timedelta(seconds=round(eta)))


def pre_iter_update(i_iter):
    cfg.update_adaptive_params(i_iter)
    agent.set_noise_rate(cfg.adp_noise_rate)
    
    # print('new noise rate', agent.noise_rate)
    # print('adp log std', cfg.adp_log_std)
    # print('cfg policy lr', cfg.adp_policy_lr)
    
    set_optimizer_lr(optimizer_policy, cfg.adp_policy_lr)
    if cfg.fix_std:
        policy_net.action_log_std.fill_(cfg.adp_log_std)
    return

def main_loop():


    #for i_iter in range(args.iter, 1):
    for i_iter in range(args.iter, cfg.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        pre_iter_update(i_iter)
        batch, log = agent.sample(cfg.min_batch_size)
        
            
        print('batch shapes:', batch.get_shapes())
        if cfg.end_reward:
            print('end reward lol')
            agent.env.end_reward = log.avg_c_reward * cfg.gamma / (1 - cfg.gamma)   
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
            with to_cpu(policy_net, value_net):
                cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_iter + 1)
                model_cp = {'policy_dict': policy_net.state_dict(),
                            'value_dict': value_net.state_dict(),
                            'running_state': running_state}
                pickle.dump(model_cp, open(cp_path, 'wb'))
        """clean up gpu memory"""
        torch.cuda.empty_cache()
    logger.info('training done!')

main_loop()

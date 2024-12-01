import argparse
import os
import sys
import pickle
import time
import datetime
sys.path.append(os.getcwd())

from rfc_utils.config import Config



#change this
from agent_envs.humanoid_amp_env_2 import HumanoidTemplate
from torch.utils.tensorboard import SummaryWriter
from networks_models.common import MLP,Value
from networks_models.policy_net import PolicyGaussian
from networks_models.discriminator_net import Discriminator
from utils.torch import *
from rl_algorithms.amp_ppo import AmpAlg
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
env = HumanoidTemplate(cfg,isExpert=False)
#env = HumanoidTemplate(cfg,isExpert=False,render_mode = 'human')
env.seed(cfg.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
#for the policy states
running_state = ZFilter((state_dim,), clip=5)
#for the amp features
running_state_amp_features = ZFilter((env.amp_features_size,), clip=5)
#for the amp states
running_state_amp = ZFilter((env.amp_features_size,), clip=5)
#for the amp next states
running_next_state_amp = ZFilter((env.amp_features_size,), clip=5)


print('observation space', state_dim)
print('action dim', action_dim)

"""size of the input for the disc"""
amp_feature_size = env.amp_features_size
print('amp features size for states', env.amp_features_size)

"""define actor and critic"""
policy_net = PolicyGaussian(MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
"""define the disc net"""
disc_net = Discriminator(net=MLP(amp_feature_size*2,cfg.value_hsize, cfg.value_htype),amp_reward_coef=0.5) 


if args.iter > 0:
    print('loading iter', args.iter)
    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
    logger.info('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    policy_net.load_state_dict(model_cp['policy_dict'])
    value_net.load_state_dict(model_cp['value_dict'])
    disc_net.load_state_dict(model_cp['disc_dict'])
    running_state = model_cp['running_state']
    running_state_amp_features = model_cp['running_state_amp_features']
    running_state_amp = model_cp['running_state_amp']
    running_next_state_amp = model_cp['running_next_state_amp']









print('policy net',policy_net)
print('valu net',value_net)
print('disc net',disc_net)






policy_net.to(device)
value_net.to(device)
disc_net.to(device)



if cfg.policy_optimizer == 'Adam':
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
else:
    optimizer_policy = torch.optim.SGD(policy_net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
if cfg.value_optimizer == 'Adam':
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
    #for now the same as value
    optimizer_disc = torch.optim.Adam(disc_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
    
else:
    optimizer_value = torch.optim.SGD(value_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)
    #same as value
    optimizer_disc = torch.optim.SGD(disc_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)



# reward functions
expert_reward = reward_func[cfg.reward_id]
print('this is the reward', expert_reward)



agent = AmpAlg(env=env, dtype=dtype, device=device, running_state=running_state, running_state_amp=running_state_amp,running_next_state_amp=running_next_state_amp,
                 running_state_amp_features=running_state_amp_features,
                 custom_reward=expert_reward, mean_action=args.render and not args.show_noise,
                 render=args.render, num_threads=args.num_threads,
                 policy_net=policy_net, value_net=value_net, discriminator=disc_net ,
                 optimizer_policy=optimizer_policy, optimizer_value=optimizer_value, optimizer_disc=optimizer_disc,opt_num_epochs=cfg.num_optim_epoch,
                 gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_epsilon=cfg.clip_epsilon,
                 policy_grad_clip=[(policy_net.parameters(), 40)], end_reward=cfg.end_reward,
                 use_mini_batch=cfg.mini_batch_size < cfg.min_batch_size, mini_batch_size=cfg.mini_batch_size)


print('opt_num_epochs', agent.opt_num_epochs)
print('mini batch', agent.use_mini_batch)
print('mean action', agent.mean_action)

def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return str(datetime.timedelta(seconds=round(eta)))

#here we define the algorithm



def main_loop():


    #for i_iter in range(args.iter, 1):
    for i_iter in range(args.iter, cfg.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        #pre_iter_update(i_iter)
        batch, log = agent.sample(cfg.min_batch_size)
        
            
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
            with to_cpu(policy_net, value_net):
                cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_iter + 1)
                model_cp = {'policy_dict': policy_net.state_dict(),
                            'value_dict': value_net.state_dict(),
                            'disc_dict': disc_net.state_dict(),
                            'running_state': running_state,
                            'running_state_amp_features': running_state_amp_features,
                            'running_state_amp': running_state_amp,
                            'running_next_state_amp': running_next_state_amp}
                pickle.dump(model_cp, open(cp_path, 'wb'))
        """clean up gpu memory"""
        torch.cuda.empty_cache()
    logger.info('training done!')

main_loop()




# def visualize():
#     obs= env.reset()
    
#     for i in range(2000):
#         #random action
#         action = env.action_space.sample()    
#         observation, reward,done,_, info = env.step(action)
#         env.render()
#         if done:
#             print('done')
#             #break
#             obs= env.reset()


# visualize()

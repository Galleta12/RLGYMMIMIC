import argparse
import os
import sys
import pickle
import time
import datetime
sys.path.append(os.getcwd())

from rfc_utils.config import Config




from agent_envs.humanoid_amp_env import HumanoidTemplate
from torch.utils.tensorboard import SummaryWriter
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
env = HumanoidTemplate(cfg,isExpert=False)
env.seed(cfg.seed)


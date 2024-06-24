import argparse
import os
import sys
import pickle
import time
import datetime
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Normal
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
sys.path.append(os.getcwd())
from rfc_utils.config import Config
from agent_envs.humanoid_gym import HumanoidGymTemplate
from rl_rfc.core.policy_gaussian import PolicyGaussian
from rl_rfc.core.critic import Value
from rl_rfc.agents import AgentPPO
from rfc_models.mlp import MLP



model_dir = "models_ppo_gym"
log_dir = "logs_ppo_gym"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

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



env = HumanoidGymTemplate(cfg)





# class CustomPolicy(ActorCriticPolicy):
#     def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
#         super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

#         state_dim = env.observation_space.shape[0]  # This retrieves the dimension of the state space
#         action_dim = env.action_space.shape[0]      # This retrieves the dimension of the action space

#         # Initialize the policy and value functions
#         self.policy_net = PolicyGaussian(MLP(state_dim, cfg.policy_hsize, cfg.policy_htype),
#                                          action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
#         self.value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))

#         # Setting up optimizers
#         if cfg.policy_optimizer == 'Adam':
#             self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
#         else:
#             self.optimizer_policy = torch.optim.SGD(self.policy_net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)

#         if cfg.value_optimizer == 'Adam':
#             self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
#         else:
#             self.optimizer_value = torch.optim.SGD(self.value_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)



#     def forward(self, obs, deterministic=False):
#         # Directly use the forward methods from PolicyGaussian and Value
#         action_distribution = self.policy_net(obs)
#         actions = action_distribution.mean_sample() if deterministic else action_distribution.sample()
#         values = self.value_net(obs)  # Use forward from Value class directly
#         return actions, values





class ActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.log_std_dist = Uniform(torch.tensor([-2.3]), torch.tensor([2.3]))
         # Set log_std as a fixed value (not a parameter, not trainable)
        #self.log_std = torch.tensor([cfg.log_std] * env.action_space.shape[0], dtype=torch.float32)
        super(ActorCriticPolicy, self).__init__(*args, **kwargs,
                                                net_arch={'vf': [512, 256], 'pi': [512, 256]},
                                                activation_fn=nn.ReLU,
                                                log_std_init=self.log_std_dist.sample().float())

        # Manually update the optimizer to enable per-layer hyperparameters to be set.
        self.optimizer = torch.optim.Adam([{'params': self.pi_features_extractor.parameters(),
                                           'lr':cfg.policy_lr,
                                           'weight_decay': cfg.policy_weightdecay},
                                          {'params': self.vf_features_extractor.parameters(),
                                           'lr': cfg.value_lr,
                                           'weight_decay': cfg.value_weightdecay}])

    # def _distribution(self, latent_pi):
    #     # Use the fixed log_std to create the action distribution
    #     mean_actions = self.action_net(latent_pi)
    #     std_actions = torch.exp(self.log_std).expand_as(mean_actions)
    #     return torch.distributions.Normal(mean_actions, std_actions)

    # def forward(self, obs, deterministic=False):
    #     latent_pi = self.extract_features(obs)
    #     distribution = self._distribution(latent_pi)
    #     actions = distribution.mean if deterministic else distribution.sample()
    #     log_prob = distribution.log_prob(actions).sum(axis=-1)
    #     return actions, log_prob


def make_env(cfg):
    def _init():
        env = HumanoidGymTemplate(cfg,render_mode="human")
        env.seed(cfg.seed)
        return env
    return _init


def train():
    mul_env = make_vec_env(make_env(cfg), n_envs=2)
    # env = VecNormalize(env,norm_obs=True)
    
    model = PPO(
        policy=ActorCriticPolicy,
        env=mul_env,
        #n_steps=4096,             # m = 4096 samples per policy update
        n_steps=4096,             # m = 4096 samples per policy update
        batch_size=256 ,           # n = 256 for minibatches
        n_epochs=cfg.num_optim_epoch,               # Number of epochs per policy update
        gamma=cfg.gamma,               # Discount factor
        gae_lambda=cfg.tau,          # Lambda for GAE
        clip_range=cfg.clip_epsilon,           # Likelihood ratio clipping 
        learning_rate=cfg.policy_lr,       # Policy learning rate
        tensorboard_log=log_dir,
        verbose=1,
        device='cuda'
    )

    
    
    iters = 0
   
    while True:
        iters += 1
        TIMESTEPS = 100_000
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True,reset_num_timesteps=False)
        
        model.save(f"{model_dir}/PPO_{TIMESTEPS*iters}")


train()
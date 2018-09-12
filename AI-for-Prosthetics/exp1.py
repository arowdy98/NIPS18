from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
from baselines.ppo2 import ppo2
import gym
import gym.spaces
from gym.spaces import Box
import time
import os
from osim.env import ProstheticsEnv
import numpy as np

env = ProstheticsEnv(visualize=False)
observation = env.reset()

# def policy_fn(name, ob_space, ac_space):
#         return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
#             hid_size=32, num_hid_layers=2)

log_dir = "/home/aditya/NIPS18/AI-for-Prosthetics"
if not os.path.exists(log_dir + '/output'):
        os.makedirs(log_dir + '/output')
logger.configure(log_dir)
trpo_mpi.learn(env=env, network = "mlp",total_timesteps=100)
import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv
import numpy as np
import os
import sys
import numpy as np
import gym
from gym import wrappers

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "06a7a9af9a879495223bcf5b9f0be33c"

client = Client(remote_base)

# Create environment
observation = client.env_create(crowdai_token, env_id='ProstheticsEnv')

# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)
def my_controller(observation,theta,n,mean,mean_diff,var):
    obs_std = np.sqrt(var)
    state = (observation-mean)/obs_std
    return theta.dot(state)


theta = np.genfromtxt('policy.out', delimiter = ' ', dtype = np.float32)
print("Loading from policy matrix.")
n = np.genfromtxt('n.out', delimiter = ' ', dtype = np.float32)
print("Loading from n matrix.")
mean = np.genfromtxt('mean.out', delimiter = ' ', dtype = np.float32)
print("Loading from mean matrix.")
mean_diff = np.genfromtxt('mean_diff.out', delimiter = ' ', dtype = np.float32)
print("Loading from Mean diff matrix.") 
var = np.genfromtxt('var.out', delimiter = ' ', dtype = np.float32)
print("Loading from Variance matrix.")

tot_reward = 0

while True:
    [observation, reward, done, info] = client.env_step(my_controller(observation,theta,n,mean,mean_diff,var), True)
    tot_reward+=reward
    print(tot_reward)
    if done:
        observation = client.env_reset()
        if not observation:
            break
print(tot_reward)
client.submit()
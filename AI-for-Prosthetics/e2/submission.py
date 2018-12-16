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

def dict_list(observation):
	obs = [];
	for key in observation:
		obs.append(float(observation[key]))
	return np.array(obs)

def _dict_to_list(state_desc):
    res = []

    # Body Observations
    for info_type in ['body_pos', 'body_pos_rot',
                      'body_vel', 'body_vel_rot',
                      'body_acc', 'body_acc_rot']:
        for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                          'femur_l', 'femur_r', 'head', 'pelvis',
                          'torso', 'pros_foot_r', 'pros_tibia_r']:
            res += state_desc[info_type][body_part]

    # Joint Observations
    # Neglecting `back_0`, `mtp_l`, `subtalar_l` since they do not move
    for info_type in ['joint_pos', 'joint_vel', 'joint_acc']:
        for joint in ['ankle_l', 'ankle_r', 'back', 'ground_pelvis',
                      'hip_l', 'hip_r', 'knee_l', 'knee_r']:
            res += state_desc[info_type][joint]

    # Muscle Observations
    for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r', 
                   'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                   'glut_max_l', 'glut_max_r', 
                   'hamstrings_l', 'hamstrings_r',
                   'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                   'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
        res.append(state_desc['muscles'][muscle]['activation'])
        res.append(state_desc['muscles'][muscle]['fiber_force'])
        res.append(state_desc['muscles'][muscle]['fiber_length'])
        res.append(state_desc['muscles'][muscle]['fiber_velocity'])

    # Force Observations
    # Neglecting forces corresponding to muscles as they are redundant with
    # `fiber_forces` in muscles dictionaries
    for force in ['AnkleLimit_l', 'AnkleLimit_r',
                  'HipAddLimit_l', 'HipAddLimit_r',
                  'HipLimit_l', 'HipLimit_r', 'KneeLimit_l', 'KneeLimit_r']:
        res += state_desc['forces'][force]

        # Center of Mass Observations
        res += state_desc['misc']['mass_center_pos']
        res += state_desc['misc']['mass_center_vel']
        res += state_desc['misc']['mass_center_acc']

    return np.array(res)

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
    observation = _dict_to_list(observation)
    [observation, reward, done, info] = client.env_step(my_controller(observation,theta,n,mean,mean_diff,var), True)
    tot_reward+=reward
    print(tot_reward)
    if done:
        observation = client.env_reset()
        if not observation:
            break
print("Episode reward:",tot_reward)
client.submit()
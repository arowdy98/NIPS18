import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv
import numpy as np
import os
import sys
import numpy as np
import gym
from gym import wrappers

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
env = ProstheticsEnv(visualize=False)
obs = env.reset(project=True)
#print(len(_dict_to_list(obs)))
print(len(obs))
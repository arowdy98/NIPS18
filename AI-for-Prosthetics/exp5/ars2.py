import os
import sys
import numpy as np
import gym
from gym import wrappers
from osim.env import ProstheticsEnv
#import roboschool
#import mujoco_py
#import pybullet_envs

# hyper parameters
class Hp():
    def __init__(self):
        self.main_loop_size = 500
        self.horizon = 1000
        self.step_size = 0.035
        self.n_directions = 125
        self.b = 125
        assert self.b<=self.n_directions, "b must be <= n_directions"
        self.noise = 0.01#0.0075
        self.seed = 1
        ''' chose your favourite '''
        #self.env_name = 'Reacher-v1'
        #self.env_name = 'Pendulum-v0'
        #self.env_name = 'HalfCheetahBulletEnv-v0'
        #self.env_name = 'Hopper-v1'#'HopperBulletEnv-v0'
        #self.env_name = 'Ant-v1'#'AntBulletEnv-v0'#
        #self.env_name = 'HalfCheetah-v1'
        #self.env_name = 'Swimmer-v1'
        #self.env_name = 'Humanoid-v1'

# observation filter
class Normalizer():
    def __init__(self, num_inputs):
        try:
            self.n = np.genfromtxt('n.out', delimiter = ' ', dtype = np.float32)
            print("Loading from old n matrix.")
        except:
            self.n = np.zeros(num_inputs)
            print("Generating new n matrix.")

        try:
            self.mean = np.genfromtxt('mean.out', delimiter = ' ', dtype = np.float32)
            print("Loading from old mean matrix.")
        except:
            self.mean = np.zeros(num_inputs)
            print("Generating new mean matrix.")

        try:
            self.mean_diff = np.genfromtxt('mean_diff.out', delimiter = ' ', dtype = np.float32)
            print("Loading from old Mean diff matrix.")
        except:
            self.mean_diff = np.zeros(num_inputs)
            print("Generating new Mean diff matrix.")

        try:
            self.var = np.genfromtxt('var.out', delimiter = ' ', dtype = np.float32)
            print("Loading from old Variance matrix.")
        except:
            self.var = np.zeros(num_inputs)
            print("Generating new Variance matrix.")

        # self.n = np.zeros(num_inputs)
        # self.mean = np.zeros(num_inputs)
        # self.mean_diff = np.zeros(num_inputs)
        # self.var = np.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = (self.mean_diff/self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs-obs_mean)/obs_std

# linear policy
class Policy():
    def __init__(self, input_size, output_size):
        try:
            self.theta = np.genfromtxt('policy.out', delimiter = ' ', dtype = np.float32)
            print("Loading from old policy matrix.")
        except:
            self.theta = np.zeros((output_size, input_size))
            print("Generating new policy matrix.")
        # print(self.theta)
        # print(self.theta.shape)

    def evaluate(self, input):
        return self.theta.dot(input)

    def positive_perturbation(self, input, delta):
        return (self.theta + hp.noise*delta).dot(input)

    def negative_perturbation(self, input, delta):
        return (self.theta - hp.noise*delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.n_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg)*d
        self.theta += hp.step_size * step / (sigma_r*hp.b)

# training loop
def train(env, policy, normalizer, hp):
    print("Starting Training")
    for episode in range(hp.main_loop_size):
        #print("******")
        # init deltas and rewards
        deltas = policy.sample_deltas()
        reward_positive = [0]*hp.n_directions
        reward_negative = [0]*hp.n_directions

        # positive directions
        print("Positive Start")
        for k in range(hp.n_directions):
            #print("???????????")
            state = env.reset()
            done = False
            num_plays = 0.
            while not done and num_plays<hp.horizon:
                normalizer.observe(state)
                state = normalizer.normalize(state)
                action = policy.positive_perturbation(state, deltas[k])
                state, reward, done, _ = env.step(action)
                reward = max(min(reward, 1), -1)
                reward_positive[k] += reward
                num_plays += 1
            print(k,reward_positive[k])

        # negative directions
        print("Negative Start")
        for k in range(hp.n_directions):
            state = env.reset()
            done = False
            num_plays = 0.
            while not done and num_plays<hp.horizon:
                normalizer.observe(state)
                state = normalizer.normalize(state)
                action = policy.negative_perturbation(state, deltas[k])
                state, reward, done, _ = env.step(action)
                reward = max(min(reward, 1), -1)
                reward_negative[k] += reward
                num_plays += 1
            print(k,reward_negative[k])

        all_rewards = np.array(reward_negative + reward_positive)
        sigma_r = all_rewards.std()

        # sort rollouts wrt max(r_pos, r_neg) and take (hp.b) best
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(reward_positive,reward_negative))}
        order = sorted(scores.keys(), key=lambda x:scores[x])[-hp.b:]
        rollouts = [(reward_positive[k], reward_negative[k], deltas[k]) for k in order[::-1]]

        # update policy:
        policy.update(rollouts, sigma_r)

        # evaluate
        state = env.reset()
        done = False
        num_plays = 1.
        reward_evaluation = 0
        while not done and num_plays<hp.horizon:
            normalizer.observe(state)
            state = normalizer.normalize(state)
            action = policy.evaluate(state)
            state, reward, done, _ = env.step(action)
            reward_evaluation += reward
            num_plays += 1

        # finish, print:
        print('episode',episode,'reward_evaluation',reward_evaluation)

        f = open("policy.out",'w')
        for i in range(num_outputs):
            for j in range(num_inputs):
                f.write(str(policy.theta[i][j])+" ")
            f.write("\n")
        f.close()

        f = open("n.out",'w')
        for j in range(num_inputs):
            f.write(str(normalizer.n[j])+" ")
        f.write("\n")
        f.close()

        f = open("mean.out",'w')
        for j in range(num_inputs):
            f.write(str(normalizer.mean[j])+" ")
        f.write("\n")
        f.close()

        f = open("mean_diff.out",'w')
        for j in range(num_inputs):
            f.write(str(normalizer.mean_diff[j])+" ")
        f.write("\n")
        f.close()

        f = open("var.out",'w')
        for j in range(num_inputs):
            f.write(str(normalizer.var[j])+" ")
        f.write("\n")
        f.close()

if __name__ == '__main__':
    hp = Hp()
    np.random.seed(hp.seed)
    #work_dir = mkdir('exp', 'brs')
    #monitor_dir = mkdir(work_dir, 'monitor')
    env = ProstheticsEnv(visualize=False)
    #env = gym.make("RoboschoolHalfCheetah-v0")
    #env = gym.make(hp.env_name)
    #env = wrappers.Monitor(env, monitor_dir, force=True)
    #num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    num_inputs = 160
    policy = Policy(num_inputs, num_outputs)
    normalizer = Normalizer(num_inputs)
    train(env, policy, normalizer, hp)
    # f = open("policy.out",'w')
    # f.write(policy.theta)
    # f.close()

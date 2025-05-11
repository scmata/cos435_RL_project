# Imports (after restarting runtime)
# Note: Most of this code is from the TD3 implementation Sean wrote for Homework 6.



import gym
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import copy
from typing import Tuple
import glfw
import imageio
import mujoco
from dm_control import suite
from dm_env import specs
import matplotlib.pyplot as plt

from utilities_DMC import DMCWrapper, make_dmc_env, flatten_obs
from utilities_TD3 import ReplayBuffer_TD3,eval_policy_TD3_DMC,  eval_policy_TD3_gym, Actor_TD3, Critic_TD3
import os
import csv
import pandas as pd




print(f" Current numpy version: {np.__version__}, gym version: {gym.__version__}")
env_name='cartpole/swingup'
env_type= "DMC" #either DMC or gym


#otherwise it's Ant-v4, or MountainCarContinuous


DMC_TASKS = [
    'acrobot/swingup', 'ball_in_cup/catch', 'cartpole/balance', 'cartpole/balance_sparse',
    'cartpole/swingup', 'cartpole/swingup_sparse', 'cheetah/run', 'dog_run', 'dog_stand', 'dog_trot',
    'dog_walk', 'finger/spin', 'finger/turn_easy', 'finger/turn_hard', 'fish/swim', 'hopper/hop',
    'hopper/stand', 'humanoid/run', 'humanoid/stand', 'humanoid/walk', 'pendulum/swingup',
    'quadruped/run', 'quadruped/walk', 'reacher/easy', 'reacher/hard',
    'walker/run', 'walker/stand', 'walker/walk'
]

GYM_TASKS = [
     'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 'humanoid-v2', 'Walker2d-v3'
]


def init_flags():

    flags = {
        "env": env_name,
        "seed":0,
        "start_timesteps": 25e3,
        "max_timesteps": 8e4,
        "expl_noise": 0.1,
        "batch_size": 256,
        "discount":0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip":0.5,
        "policy_freq": 2,
        "save_model": "store_true"
    }

    return flags

# The following main() function is provided to you. It can a run for both DDPG and TD3..
def main(policy_name='TD3'):
    args = init_flags()

    if env_type == "DMC":
        domain, task = args["env"].split("/")
        env, action_spec = make_dmc_env(domain, task, args["seed"])

        # Set seeds
        torch.manual_seed(args["seed"])
        np.random.seed(args["seed"])

        time_step = env.reset()
        state = flatten_obs(time_step.observation)
        state_dim = state.shape[0]
        action_dim = action_spec.shape[0]
        max_action = action_spec.maximum[0]

    elif env_type == "gym":
        env = gym.make(args["env"])
        env.seed(args["seed"] + 100)
        env.action_space.seed(args["seed"])
        torch.manual_seed(args["seed"])
        np.random.seed(args["seed"])

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args["discount"],
        "tau": args["tau"],
    }

    if policy_name == "TD3":
        kwargs["policy_noise"] = args["policy_noise"] * max_action
        kwargs["noise_clip"] = args["noise_clip"] * max_action
        kwargs["policy_freq"] = args["policy_freq"]
        policy = TD3(**kwargs)
    elif policy_name == "RandomPolicy":
        policy = RandomPolicy(**kwargs)

    replay_buffer = ReplayBuffer_TD3(state_dim, action_dim)

    if env_type == "DMC":
        evaluations = [eval_policy_TD3_DMC(policy, args["env"], args["seed"])]
    elif env_type == "gym":
        evaluations = [eval_policy_TD3_gym(policy, args["env"], args["seed"])]
        state, done = env.reset(), False

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args["max_timesteps"])):
        episode_timesteps += 1

        # Select action
        if t < args["start_timesteps"]:
            if env_type == "DMC":
                action = np.random.uniform(
                    low=action_spec.minimum,
                    high=action_spec.maximum,
                    size=action_dim
                )
            elif env_type == "gym":
                action = env.action_space.sample()
        else:
            if policy_name != 'RandomPolicy':
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args["expl_noise"], size=action_dim)
                ).clip(-max_action, max_action)
            else:
                if env_type == "DMC":
                    action = np.random.uniform(
                        low=action_spec.minimum,
                        high=action_spec.maximum,
                        size=action_dim
                    )
                elif env_type == "gym":
                    action = env.action_space.sample()

        # Step environment
        if env_type == "DMC":
            time_step = env.step(action)
            next_state = flatten_obs(time_step.observation)
            reward = time_step.reward or 0.0
            done = time_step.last()
            done_bool = float(done)
        elif env_type == "gym":
            next_state, reward, done, _ = env.step(action)
            done_bool = float(done) if episode_timesteps < env.spec.max_episode_steps else 0

        # Store data
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent
        if t >= args["start_timesteps"]:
            policy.train(replay_buffer, args["batch_size"])

        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            evaluations.append(episode_reward)

            # Reset environment
            if env_type == "DMC":
                time_step = env.reset()
                state = flatten_obs(time_step.observation)
            elif env_type == "gym":
                state, done = env.reset(), False

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    return evaluations


#kiri
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		max_action: float,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor_TD3(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic_TD3(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise.
			clip_noise = (torch.randn_like(action)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			next_action = self.actor_target(next_state) + clip_noise # Fixed: Using next_state instead of state, and assigning to next_action
			next_action = next_action.clamp(-self.max_action, self.max_action)


			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)

			# 2. Compute the target_Q here

			min_q = torch.min(target_Q1, target_Q2)
			target_Q = reward + self.discount * min_q * not_done

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			actor_loss = - self.critic.Q1(state, self.actor(state)).mean()
			############################

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models using weighted mean
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				new_target_params = self.tau*param + (1-self.tau) * target_param
				target_param.data.copy_(new_target_params)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				new_target_params = self.tau*param + (1-self.tau) * target_param
				target_param.data.copy_(new_target_params)
                    


class Actor_Random():
    def __init__(self, action_dim: int, max_action: float):
        self.max_action = max_action
        self.action_dim = action_dim

    def GetAction(self):
        return (torch.rand(self.action_dim) * self.max_action).cpu().numpy()


class RandomPolicy(object):
	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		max_action: float,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor_Random(action_dim, max_action)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		return self.actor.GetAction()


	def train(self, replay_buffer, batch_size=256):
		pass
     

'''
evaluations_random = main(policy_name = 'RandomPolicy') #either TD3 or RandomPolicy
evaluations_td3 = main(policy_name = 'TD3') #either TD3 or RandomPolicy

# Create the plots directory if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

if not os.path.exists("results"):
    os.makedirs("results")


plt.plot(evaluations_td3)
plt.xlabel('Episode Num')
plt.ylabel('reward')
plt.savefig("plots/reward_plot.png")

# Save evaluations to a DataFrame and then to CSV

# Create a DataFrame from the evaluations
df = pd.DataFrame({
    "Episode": range(1, len(evaluations_td3) + 1),
    "Reward": evaluations_td3
})

# Save the DataFrame to a CSV file
df.to_csv(f"results/{env_name.replace('/', '_')}_evaluationsTD3.csv", index=False)
'''


env_type = "gym" #either DMC or gym
for task in GYM_TASKS:
    env_name = task
    print(f"Running MRQ on {env_name}")
    evaluations_MRQ = main(policy_name='TD3', _env_name=env_name)

    # Replace "/" with "_" in the environment name for safe file indexing
    safe_env_name = env_name.replace("/", "_")

    # Plot and save the reward plot
    plt.figure()
    plt.plot(evaluations_MRQ)
    plt.xlabel('Episode Num')
    plt.ylabel('Reward')
    plt.title(f"MRQ {safe_env_name} Reward Plot")
    plt.savefig(f"plots/reward_plot_{safe_env_name}_MRQ_A.png")
    plt.close()

    # Save evaluations to a DataFrame and then to CSV
    df = pd.DataFrame({
        "Episode": range(1, len(evaluations_MRQ) + 1),
        "Reward": evaluations_MRQ
    })
    df.to_csv(f"results/{safe_env_name}_evaluations_MRQ_A.csv", index=False)
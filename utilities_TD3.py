
import numpy as np
import torch
from dm_control import suite
from dm_env import specs
import gym
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from utilities_DMC import flatten_obs


class ReplayBuffer_TD3(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	

# policy evaluation with Monte Carlo
def eval_policy_TD3_DMC(policy, env_name, seed, eval_episodes=10):
    domain, task = env_name.split("/")
    env = suite.load(domain, task, task_kwargs={"random": seed})


    avg_reward = 0.
    for _ in range(eval_episodes):
        time_step = env.reset()
        state = flatten_obs(time_step.observation)
        done = False
        while not time_step.last():
            action = policy.select_action(np.array(state))
            time_step = env.step(action)
            reward = time_step.reward or 0.0
            avg_reward += reward
            state = flatten_obs(time_step.observation)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


# policy evaluation with Monte Carlo
def eval_policy_TD3_gym(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _= eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


class Actor_TD3(nn.Module):
	def __init__(self, state_dim: int, action_dim: int, max_action: float):
		super(Actor_TD3, self).__init__()
		"""
    Inputs: same as DDPG actor
		Outputs of forward: torch.Tensor that represents the chosen action
		"""
		self.fc1 = nn.Linear(state_dim, 256)
		self.fc2 = nn.Linear(256,256)
		self.fc3 = nn.Linear(256, action_dim)
		self.max_action = max_action


	def forward(self, state):
		x = torch.relu(self.fc1(state))
		x = torch.relu(self.fc2(x))
		x = torch.tanh(self.fc3(x))
		return x * self.max_action


class Critic_TD3(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic_TD3, self).__init__()
        """
        Inputs: same as the actor
        Outputs: two torch.Tensors that represent Q1 and Q2
        """ 
        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
		
        self.fc4 = nn.Linear(state_dim+action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.fc1(sa))
        q1 = torch.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = torch.relu(self.fc4(sa))
        q2 = torch.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.fc1(sa))
        q1 = torch.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1
	


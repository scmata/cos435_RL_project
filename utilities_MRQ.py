import numpy as np
import torch
from collections import deque
import torch.nn.functional as F
from collections.abc import Callable

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


def ln_activ(x: torch.Tensor, activ: Callable):
    #Referenced from MR.Q Github Repo
      x = F.layer_norm(x, (x.shape[-1],))
      return activ(x)



# We include some optimizations in this buffer to storing states multiple times when history or horizon > 1.
class ReplayBuffer_MRQ:
    #Referenced from MR.Q Github Repo
    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, max_action: float, pixel_obs: bool,
        device: torch.device, history: int=1, horizon: int=1, max_size: int=1e6, batch_size: int=256,
        prioritized: bool=True, initial_priority: float=1, normalize_actions: bool=True):

        self.max_size = int(max_size)
        self.batch_size = batch_size

        self.obs_shape = obs_shape # Size of individual frames.
        self.obs_dtype = torch.uint8 if pixel_obs else torch.float

        # Size of state given to network
        self.state_shape = [obs_shape[0] * history] # Channels or obs dim.
        if pixel_obs: self.state_shape += [obs_shape[1], obs_shape[2]] # Image size.
        self.num_channels = obs_shape[0] # Used to grab only the most recent obs (history) or channels.

        self.device = device

        # Store obs on GPU if they are sufficient small.
        memory, _ = torch.cuda.mem_get_info()
        obs_space = np.prod((self.max_size, *self.obs_shape)) * 1 if pixel_obs else 4
        ard_space = self.max_size * (action_dim + 2) * 4
        if obs_space + ard_space < memory:
            self.storage_device = self.device
        else:
            self.storage_device = torch.device('cpu')

        self.action_dim = action_dim
        self.action_scale = max_action if normalize_actions else 1.

        # Tracking
        self.ind, self.size = 0, 0
        self.ep_timesteps = 0
        self.env_terminates = False # Used to track if there are any terminal transitions in the buffer.

        # History (used even if history = 1)
        self.history = history
        self.state_ind = np.zeros((self.max_size, self.history), dtype=np.int32) # Tracks the indices of the current state.
        self.next_ind = np.zeros((self.max_size, self.history), dtype=np.int32) # Tracks the indices of the next state.

        self.history_queue = deque(maxlen=self.history)
        for _ in range(self.history): # Initialize with self.ind=0.
            self.history_queue.append(0)

        # Multi-step
        self.horizon = horizon

        # Prioritization
        self.prioritized = prioritized
        self.priority = torch.empty(self.max_size, device=self.device) if self.prioritized else []
        self.max_priority = initial_priority

        # Sampling mask, used to hide states that we don't want to sample, either due to truncation or replacing states in the horizon.
        self.mask = torch.zeros(self.max_size, device=self.device if prioritized else torch.device('cpu'))

        # Actual storage
        self.obs = torch.zeros((self.max_size, *self.obs_shape), device=self.storage_device, dtype=self.obs_dtype)
        self.action_reward_notdone = torch.zeros((self.max_size, action_dim + 2), device=self.device, dtype=torch.float)


    # Extract the most recent obs from the state that includes history.
    def extract_obs(self, state: np.array):
        return torch.tensor(
            state[-self.num_channels:].reshape(self.obs_shape),
            dtype=self.obs_dtype, device=self.storage_device
        )


    # Used to map discrete actions to one hot or normalize continuous actions.
    def one_hot_or_normalize(self, action: int | float):
        if isinstance(action, int):
            one_hot_action = torch.zeros(self.action_dim, device=self.device)
            one_hot_action[action] = 1
            return one_hot_action
        return torch.tensor(action/self.action_scale, dtype=torch.float, device=self.device)


    def add(self, state: np.array, action: int | float, next_state: np.array, reward: float, terminated: bool, truncated: bool):
        self.obs[self.ind] = self.extract_obs(state)
        self.action_reward_notdone[self.ind,0] = reward
        self.action_reward_notdone[self.ind,1] = 1. - terminated
        self.action_reward_notdone[self.ind,2:] = self.one_hot_or_normalize(action)

        if self.prioritized:
            self.priority[self.ind] = self.max_priority

        # Tracking
        self.size = max(self.size, self.ind + 1)
        self.ep_timesteps += 1
        if terminated: self.env_terminates = True

        # Masking
        self.mask[(self.ind + self.history - 1) % self.max_size] = 0
        if self.ep_timesteps > self.horizon: # Allow states that have a completed horizon to be sampled.
            self.mask[(self.ind - self.horizon) % self.max_size] = 1

        # History
        next_ind = (self.ind + 1) % self.max_size
        self.state_ind[self.ind] = np.array(self.history_queue, dtype=np.int32) # Track last x=history obs for the state.
        self.history_queue.append(next_ind) # Update history queue with incremented ind.
        self.next_ind[self.ind] = np.array(self.history_queue, dtype=np.int32)
        self.ind = next_ind

        if terminated or truncated:
            self.terminal(next_state, truncated)


    def terminal(self, state: np.array, truncated: bool):
        self.obs[self.ind] = self.extract_obs(state)

        self.mask[(self.ind + self.history - 1) % self.max_size] = 0
        past_ind = (self.ind - np.arange(min(self.ep_timesteps, self.horizon)) - 1) % self.max_size
        self.mask[past_ind] = 0 if truncated else 1 # Mask out truncated subtrajectories.

        self.ind = (self.ind + 1) % self.max_size
        self.ep_timesteps = 0

        # Reset queue
        for _ in range(self.history):
            self.history_queue.append(self.ind)


    def sample_ind(self):
        if self.prioritized:
            csum = torch.cumsum(self.priority * self.mask, 0)
            self.sampled_ind = torch.searchsorted(
                csum,
                torch.rand(size=(self.batch_size,), device=self.device)*csum[-1]
            ).cpu().data.numpy()
        else:
            nz = torch.nonzero(self.mask).reshape(-1)
            self.sampled_ind = np.random.randint(nz.shape[0], size=self.batch_size)
            self.sampled_ind = nz[self.sampled_ind]
        return self.sampled_ind


    def sample(self, horizon: int, include_intermediate: bool=False):
        ind = self.sample_ind()
        ind = (ind.reshape(-1,1) + np.arange(horizon).reshape(1,-1)) % self.max_size

        ard = self.action_reward_notdone[ind]

        # Sample subtrajectory (with horizon dimension) for unrolling dynamics.
        if include_intermediate:
            # Group (state, next_state) to speed up CPU -> GPU transfer.
            state_ind = np.concatenate([
                self.state_ind[ind],
                self.next_ind[ind[:,-1].reshape(-1,1)]
            ], 1)
            both_state = self.obs[state_ind].reshape(self.batch_size,-1,*self.state_shape).to(self.device).type(torch.float)
            state = both_state[:,:-1]       # State: (batch_size, horizon, *state_dim)
            next_state = both_state[:,1:]   # Next state: (batch_size, horizon, *state_dim)
            action = ard[:,:,2:]            # Action: (batch_size, horizon, action_dim)

        # Sample at specific horizon (used for multistep rewards).
        else:
            state_ind = np.concatenate([
                self.state_ind[ind[:,0].reshape(-1,1)],
                self.next_ind[ind[:,-1].reshape(-1,1)]
            ], 1)
            both_state = self.obs[state_ind].reshape(self.batch_size,2,*self.state_shape).to(self.device).type(torch.float)
            state = both_state[:,0]         # State: (batch_size, *state_dim)
            next_state = both_state[:,1]    # Next state: (batch_size, *state_dim)
            action = ard[:,0,2:]            # Action: (batch_size, action_dim)

        return state, action, next_state, ard[:,:,0].unsqueeze(-1), ard[:,:,1].unsqueeze(-1)


    def update_priority(self, priority: torch.Tensor):
        self.priority[self.sampled_ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)


    def reward_scale(self, eps: float=1e-8):
        return float(self.action_reward_notdone[:self.size,0].abs().mean().clamp(min=eps))


    def save(self, save_folder: str):
        np.savez_compressed(f'{save_folder}/buffer_data',
            obs = self.obs.cpu().data.numpy(),
            ard = self.action_reward_notdone.cpu().data.numpy(),
            state_ind = self.state_ind,
            next_ind = self.next_ind,
            priority = self.priority.cpu().data.numpy(),
            mask = self.mask.cpu().data.numpy()
        )

        v = ['ind', 'size', 'env_terminates', 'history_queue', 'max_priority']
        var_dict = {k: self.__dict__[k] for k in v}

        np.save(f'{save_folder}/buffer_var.npy', var_dict)


    def load(self, save_folder: str):
        buffer_data = np.load(f'{save_folder}/buffer_data.npz')

        self.obs = torch.tensor(buffer_data['obs'], device=self.storage_device, dtype=self.obs_dtype)
        self.action_reward_notdone = torch.tensor(buffer_data['ard'], device=self.device, dtype=torch.float)
        self.state_ind = buffer_data['state_ind']
        self.next_ind = buffer_data['next_ind']
        if self.prioritized: self.priority = torch.tensor(buffer_data['priority'], device=self.device)
        self.mask = torch.tensor(buffer_data['mask'], device=self.device if self.prioritized else torch.device('cpu'))

        var_dict = np.load(f'{save_folder}/buffer_var.npy', allow_pickle=True).item()
        for k, v in var_dict.items(): self.__dict__[k] = v



# policy evaluation with Monte Carlo
def eval_policy_MRQ_gym(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action_MRQ(np.array(state), True)
			state, reward, done, truncated = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


# policy evaluation with Monte Carlo
def eval_policy_MRQ_DMC(policy, env_name, seed, eval_episodes=10):
    domain, task = env_name.split("/")
    env = suite.load(domain, task, task_kwargs={"random": seed})

    avg_reward = 0.
    for _ in range(eval_episodes):
        time_step = env.reset()
        state = flatten_obs(time_step.observation)
        done = False
        while not time_step.last():
            action = policy.select_action_MRQ(np.array(state), True)
            time_step = env.step(action)
            reward = time_step.reward or 0.0
            avg_reward += reward
            state = flatten_obs(time_step.observation)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

class TwoHot:
    #Referenced from MR.Q Github Repo
    def __init__(self, device: torch.device, lower: float=-10, upper: float=10, num_bins: int=101):
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1)
        self.num_bins = num_bins


    def transform(self, x: torch.Tensor):
        diff = x - self.bins.reshape(1,-1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind+1).clamp(0, self.num_bins-1)]
        weight = (x - lower)/(upper - lower)

        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind+1).clamp(0, self.num_bins), weight)
        return two_hot


    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)

    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.transform(target)
        return -(target * pred).sum(-1, keepdim=True)
    

class Encoder(nn.Module):
    """
    3-layer multilayer perceptron (MLP) with 512 hidden units.
    Uses ELU activations, as in the original paper.
    """
    def __init__(self, state_dim: int, action_dim: int,
        num_bins: int=65, zs_dim: int=512, za_dim: int=256, zsa_dim: int=512):

        super(Encoder, self).__init__()
        self.SEl1 = nn.Linear(state_dim, 512)
        self.SEl2 = nn.Linear(512, 512)
        self.SEl3 = nn.Linear(512, zs_dim)

        self.za = nn.Linear(action_dim, za_dim)
        self.zsa1 = nn.Linear(zs_dim + za_dim, 512)
        self.zsa2 = nn.Linear(512, 512)
        self.zsa3 = nn.Linear(512, zsa_dim)

        self.activ = F.elu

        self.model = nn.Linear(zsa_dim, num_bins + zs_dim + 1)

        self.zs_dim = zs_dim
        self.za_dim = za_dim
        self.zsa_dim = zsa_dim
        self.num_bins = num_bins

        self.model = nn.Linear(self.zsa_dim, self.num_bins + self.zs_dim + 1)



    def state(self, state):
        zs = F.elu(self.SEl1(state))
        zs = F.elu(self.SEl2(zs))
        zs = self.SEl3(zs)
        return zs

    def state_action(self, zs, action):
      za = F.elu(self.za(action))
      zsa = torch.cat([zs, za], dim=-1)
      zsa = ln_activ(self.zsa1(zsa), self.activ)
      zsa = ln_activ(self.zsa2(zsa), self.activ)
      zsa = self.zsa3(zsa)
      return self.model(zsa), zsa

    def model_all(self, zs, action):
      _, zsa = self.state_action(zs, action)
      dzr =  self.model(zsa)
      return dzr[:,0:1], dzr[:,1:self.zs_dim+1], dzr[:,self.zs_dim+1:] # done, zs, reward

class QNetwork(nn.Module):
  def __init__(self, zsa_dim: int):
    super(QNetwork, self).__init__()

    self.l1 = nn.Linear(zsa_dim, 512)
    self.l2 = nn.Linear(512, 512)
    self.l3 = nn.Linear(512, 512)
    self.l4 = nn.Linear(512, 1)
    self.activ = F.elu

  def forward(self, zsa):
    q = ln_activ(self.l1(zsa), self.activ)
    q = ln_activ(self.l2(q), self.activ)
    q = ln_activ(self.l3(q), self.activ)
    return self.l4(q)
  

class PolicyNetwork(nn.Module):
  """
    Policy network is a three layer MLP hidden dimension 512,
    LayerNorm followed by ReLU activations after the first two layers.
    Final activation is a tanh function.
  """
  def __init__(self, zs_dim: int, action_dim: int):
    super(PolicyNetwork, self).__init__()

    self.l1 = nn.Linear(zs_dim, 512)
    self.l2 = nn.Linear(512, 512)
    self.l3 = nn.Linear(512, action_dim)
    self.activ = F.relu

  def action(self, zs):
    a = ln_activ(self.l1(zs), self.activ)
    a = ln_activ(self.l2(a), self.activ)
    return torch.tanh(self.l3(a))

  def forward(self, zs):
    a = ln_activ(self.l1(zs), self.activ)
    pre_a = ln_activ(self.l2(a), self.activ)
    return torch.tanh(self.l3(pre_a)), pre_a

class StochasticPolicyNetwork(nn.Module):
  """
    Policy network is a three layer MLP hidden dimension 512,
    LayerNorm followed by ReLU activations after the first two layers.
    Final activation is a tanh function.
  """
  def __init__(self, zs_dim: int, action_dim: int):
    super(PolicyNetwork, self).__init__()

    self.l1 = nn.Linear(zs_dim, 512)
    self.l2 = nn.Linear(512, 512)
    self.mean_layer = nn.Linear(512, action_dim)
    self.log_std_layer = nn.Linear(512, action_dim)

    self.LOG_STD_MIN = -5
    self.LOG_STD_MAX = 2

    self.activ = F.relu
    self.std_bound = [-20, 2]

  def forward(self, zs):
    a = ln_activ(self.l1(zs), self.activ)
    a = ln_activ(self.l2(a), self.activ)
    mean = self.mean_layer(a)
    log_std = self.log_std_layer(a)
    log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
    return mean, log_std

  def sample(self, zs):
    mean, log_std = self.forward(zs)
    std = log_std.exp()

    normal = torch.distributions.Normal(mean, std)
    x_t = normal.rsample()  # Sample with noise and track gradients
    a_t = torch.tanh(x_t)

    #Log-prob with correction Tanh correction (change of variables formula)
    log_prob = normal.log_prob(x_t)
    log_prob = log_prob - torch.log(1 - a_t.pow(2) + 1e-6)
    log_prob = log_prob.sum(1, keepdim=True)

    return a_t, x_t, log_prob, torch.tanh(mean)  # return also the mean action for eval mode

  def action(self, zs):
      mean, log_std = self.forward(zs)
      return torch.tanh(mean)
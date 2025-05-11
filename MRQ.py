import gym
import random
import imageio
import os
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
from collections.abc import Callable
from collections import deque

from dm_control import suite
from dm_env import specs
from gym import spaces
import pandas as pd
from utilities_DMC import DMCWrapper, make_dmc_env, flatten_obs

from utilities_MRQ import ReplayBuffer_MRQ, Encoder, PolicyNetwork, QNetwork, TwoHot, eval_policy_MRQ_DMC, eval_policy_MRQ_gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f" Current numpy version: {np.__version__}, gym version: {gym.__version__}")
env_name='cartpole/swingup'
env_type= "DMC" #either DMC or gym

#env_name = 'BipedalWalker-v3'
#env_type = "gym" #either DMC or gym

class MRQ_agent(object):
  def __init__(self, state_dim: int, action_dim: int, max_action: float, discount: float=0.99):
    # env settings
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.max_action = max_action
    self.total_it = 0

    # experiment settings
    self.random_exploration_steps = 10000
    self.encoder_horizon = 5
    self.noise_level = 0.2
    self.policy_noise = 0.2
    self.noise_clip = 0.3
    self.use_exploration = True
    self.discount = discount
    self.reward_scale = 1 #update
    self.target_reward_scale = 1 #update
    self.two_hot = TwoHot(device, -10, 10, 65)
    self.value_grad_clip = 20
    self.pre_activ_weight = 1e-5
    self.encoder_steps = 250

    self.dyn_weight = 1
    self.reward_weight = 0.1
    self.done_weight = 0.1

    # encoder setting
    self.zs_dim = 512
    self.za_dim = 256
    self.zsa_dim = 512
    self.enc_horizon = 5

    # State Encoder
    self.encoder = Encoder(state_dim, action_dim).to(device)
    self.encoder_target = copy.deepcopy(self.encoder)
    self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=3e-4)

    # Policy
    self.policy = PolicyNetwork(self.zs_dim, action_dim).to(device)
    self.policy_target = copy.deepcopy(self.policy)
    self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

    # Value
    self.Q = QNetwork(self.zsa_dim).to(device)
    self.Q1_target = copy.deepcopy(self.Q)
    self.Q2_target = copy.deepcopy(self.Q)
    self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=3e-4)

    self.TwoHot = TwoHot(device, -10, 10, 65)

  def select_action_MRQ(self, state, use_exploration):
    with torch.no_grad():
      state = torch.FloatTensor(state.reshape(1, -1)).to(device)
      zs = self.encoder.state(state)
      action, _ = self.policy(zs)
      if self.use_exploration:
        noise = torch.randn_like(action) * self.noise_level
        action += noise
      return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

  def train(self, replay_buffer, batch_size=64):
    self.total_it += 1

    # TODO: random exploration
    if replay_buffer.size <= self.random_exploration_steps:
      return

    # Sample replay buffer
    state, action, next_state, reward, not_done = replay_buffer.sample(horizon=3, include_intermediate = False)
    # multiple step reward
    term_discount = 1
    ms_reward = 0
    for i in range(reward.shape[1]):
      ms_reward += term_discount * reward[:,i]
      term_discount *= self.discount * not_done[:,i]

    # optimise QNetwork
    Q, Q_tgt = self.train_Q(state, action, next_state, ms_reward,
                            term_discount, self.reward_scale, self.target_reward_scale)

    # update networks and train encoder
    if self.total_it % self.encoder_steps == 0:
      self.policy_target.load_state_dict(self.policy.state_dict())
      self.Q1_target.load_state_dict(self.Q.state_dict())
      self.Q2_target.load_state_dict(self.Q.state_dict())
      self.encoder_target.load_state_dict(self.encoder.state_dict())
      # TODO: reward scale update

      for _ in range(self.encoder_steps):
        state, action, next_state, reward, not_done = replay_buffer.sample(horizon=self.encoder_horizon, include_intermediate=True)
        self.train_encoder(state, action, next_state, reward, not_done, replay_buffer.env_terminates)


  def train_encoder(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                  reward: torch.Tensor, not_done: torch.Tensor, env_terminates: bool):
    with torch.no_grad():
      #May need to re-shape next_state.reshape(-1,*self.state_shape)
      encoder_target = self.encoder.state(next_state)

    pred_zs = self.encoder.state(state[:,0])
    prev_not_done = 1
    encoder_loss = 0
    for i in range(self.enc_horizon):
      pred_d, pred_zs, pred_r = self.encoder.model_all(pred_zs, action[:, i])
      dynamic_loss = (F.mse_loss(pred_zs, encoder_target[:,i], reduction='none') * prev_not_done).mean()
      #CE between the predicted reward and a two-hot encoding of the reward r:
      reward_loss = self.TwoHot.cross_entropy_loss(pred_r, reward[:,i]).mean()
      done_loss = (F.mse_loss(pred_d, 1- not_done[:,i].reshape(-1,1), reduction='none') * prev_not_done).mean() if env_terminates else 0

      #Update encoder loss
      encoder_loss = encoder_loss + self.dyn_weight * dynamic_loss + self.reward_weight * reward_loss + self.done_weight * done_loss
      prev_not_done = not_done[:,i].reshape(-1,1) * prev_not_done # Adjust termination mask.

    self.encoder_optimizer.zero_grad(set_to_none=True)
    encoder_loss.backward()
    self.encoder_optimizer.step()

  def train_Q(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                    reward: torch.Tensor, term_discount: torch.Tensor, reward_scale: float, target_reward_scale: float):
    with torch.no_grad():
      next_zs = self.encoder_target.state(next_state)

      clipped_noise = (torch.randn_like(action)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
      next_action = (self.policy_target.action(next_zs) + clipped_noise).clamp(-self.max_action, self.max_action)

      _, next_zsa = self.encoder_target.state_action(next_zs, next_action)

      Q1_target_val = self.Q1_target(next_zsa)
      Q2_target_val = self.Q2_target(next_zsa)

      target_Q = torch.min(Q1_target_val, Q2_target_val)

      #Need to think about
      target_Q = (reward + term_discount * target_Q * target_reward_scale)/reward_scale

      zs = self.encoder.state(state)
      _, zsa = self.encoder.state_action(zs, action)

    Q_val = self.Q(zsa)
    Q_loss = F.smooth_l1_loss(Q_val, target_Q)

    self.Q_optimizer.zero_grad(set_to_none=True)
    Q_loss.backward()
    #Again need to think about also re-write
    torch.nn.utils.clip_grad_norm_(self.Q.parameters(), self.value_grad_clip)
    self.Q_optimizer.step()

    action, pre_activation = self.policy(zs)
    _, zsa = self.encoder.state_action(zs, action)
    Q_policy = self.Q(zsa)
    policy_loss = -Q_policy.mean() + self.pre_activ_weight * pre_activation.pow(2).mean()

    self.policy_optimizer.zero_grad(set_to_none=True)
    policy_loss.backward()
    self.policy_optimizer.step()

    return Q_val, target_Q



def init_flags():

  flags = {
        "env": env_name,
        "seed":0 ,
        "start_timesteps": 1e4, #needs to be 100k at some point
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
def main(policy_name='MRQ'):

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


        time_step = env.reset()
        state = flatten_obs(time_step.observation)
        state_dim = state.shape[0]
        state_dim_vec = state.shape

    elif env_type == "gym":
        env = gym.make(args["env"])
        env.seed(args["seed"] + 100)
        env.action_space.seed(args["seed"])
        torch.manual_seed(args["seed"])
        np.random.seed(args["seed"])

        state_dim = env.observation_space.shape[0]
        state_dim_vec = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args["discount"],
    }

    if policy_name == "MRQ":
        policy = MRQ_agent(**kwargs)
        replay_buffer = ReplayBuffer_MRQ(state_dim_vec, action_dim, max_action= max_action, pixel_obs= False, device= device)
        
        if env_type == "DMC":
            evaluations = [eval_policy_MRQ_DMC(policy, args["env"], args["seed"])]
        elif env_type == "gym":
            evaluations = [eval_policy_MRQ_gym(policy, args["env"], args["seed"])]
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
                    policy.select_action_MRQ(np.array(state),use_exploration=False)
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
            #need to get the truncated value
            truncated = 0.0  # DMC environments do not have a truncated flag
        elif env_type == "gym":
            next_state, reward, done, truncated = env.step(action)
            done_bool = float(done) if episode_timesteps < env.spec.max_episode_steps else 0

        # Store data in replay buffer
        if policy_name == "MRQ":
            replay_buffer.add(state, action, next_state, reward, done_bool, truncated)

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




evaluations_MRQ = main(policy_name = 'MRQ')


# Create the plots directory if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

if not os.path.exists("results"):
    os.makedirs("results")


# Replace "/" with "_" in the environment name for safe file indexing
safe_env_name = env_name.replace("/", "_")

plt.plot(evaluations_MRQ)
plt.xlabel('Episode Num')
plt.ylabel('reward')
plt.title(f"MRQ {safe_env_name} reward plot")
plt.savefig(f"plots/reward_plot_{safe_env_name}.png")

# Save evaluations to a DataFrame and then to CSV

# Create a DataFrame from the evaluations
df = pd.DataFrame({
    "Episode": range(1, len(evaluations_MRQ) + 1),
    "Reward": evaluations_MRQ
})

# Save the DataFrame to a CSV file
df.to_csv(f"results/{env_name.replace('/', '_')}_evaluationsMRQ.csv", index=False)



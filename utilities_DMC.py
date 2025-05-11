from dm_control import suite
from dm_env import specs
import numpy as np


class DMCWrapper:
    def __init__(self, domain_name, task_name, frame_skip=2, seed=42):
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})
        self.frame_skip = frame_skip
        self._reset()

        # Define action and observation spaces
        self.action_spec = self.env.action_spec()
        self.obs_spec = self.env.observation_spec()
        self.observation_size = sum(np.prod(v.shape) for v in self.obs_spec.values())
        self.action_size = np.prod(self.action_spec.shape)

    def _flatten_obs(self, obs_dict):
        return np.concatenate([v.ravel() for v in obs_dict.values()])

    def _reset(self):
        self.time_step = self.env.reset()
        return self._flatten_obs(self.time_step.observation)

    def reset(self):
        return self._reset()

    def step(self, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            self.time_step = self.env.step(action)
            total_reward += self.time_step.reward or 0.0
            if self.time_step.last():
                break
        obs = self._flatten_obs(self.time_step.observation)
        done = self.time_step.last()
        return obs, total_reward, done, {}
    

def make_dmc_env(domain_name, task_name, seed):
    env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={"random": seed})
    spec = env.action_spec()
    return env, spec

def flatten_obs(obs_dict):
    return np.concatenate([v.ravel() for v in obs_dict.values()])
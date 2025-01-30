from typing import Dict, List, NoReturn, Optional, Tuple, Union

import gym
import numpy as np
import torch
from gym import spaces

from .const import *
from ebb.env.luxenv import ACTION_SPACE


class VecEnv(gym.Env):

  def __init__(self, envs: List[gym.Env], use_single_player=False):
    assert len(envs) > 0
    self.envs = envs
    self.last_outs = [() for _ in range(len(self.envs))]
    self.use_single_player = use_single_player

  @staticmethod
  def _stack_dict(x: List[Union[Dict, np.ndarray]],
                  is_observation=False) -> Union[Dict, np.ndarray]:
    if isinstance(x[0], dict):
      return {
          key: VecEnv._stack_dict([i[key] for i in x], is_observation)
          for key in x[0].keys()
      }
    else:
      if is_observation:
        return np.concatenate([arr for arr in x], axis=0)
      return np.stack([arr for arr in x], axis=0)

  def _vectorize_env_outs(self,
                          env_outs: List[Tuple],
                          reset: bool = False) -> Tuple:
    """Double the env out."""

    def _unzip_env_out():
      "For each env_out of [x1, x2], flatten it to x1, x2 ..."
      for env_out in env_outs:
        obs, reward, done, info = env_out
        if self.use_single_player:
          yield obs[0], reward[0], done, info[0]
          continue

        done = [done] * 2
        for out in zip(obs, reward, done, info):
          yield out

    obs_list, reward_list, done_list, info_list = zip(*_unzip_env_out())

    obs_stacked = VecEnv._stack_dict(obs_list, is_observation=True)
    reward_stacked = np.array(reward_list)
    done_stacked = np.array(done_list)
    info_stacked = VecEnv._stack_dict(info_list)

    if not self.use_single_player:
      assert len(obs_list) == 2 * len(env_outs)
      assert len(reward_list) == 2 * len(env_outs)
      assert len(done_list) == 2 * len(env_outs)
      assert len(info_list) == 2 * len(env_outs)

    return obs_stacked, reward_stacked, done_stacked, info_stacked

  def reset(self, force: bool = False, **kwargs):
    if force:
      # noinspection PyArgumentList
      self.last_outs = [env.reset(**kwargs) for env in self.envs]
      ret = self._vectorize_env_outs(self.last_outs)
      return ret

    for i, env in enumerate(self.envs):
      # Check if env finished
      if self.last_outs[i][2]:
        # noinspection PyArgumentList
        self.last_outs[i] = env.reset(**kwargs)
    ret = self._vectorize_env_outs(self.last_outs)
    return ret

  def step(self, actions: Dict[str, torch.Tensor]):
    # assert len(actions) == len(
    # ACTION_SPACE), 'number of  actions match len(ACTION_SPACE)'

    def groupby(iterable, n):
      """
      for x in groupby(range(9), 3):
        print(x)

      (0, 1, 2)
      (3, 4, 5)
      (6, 7, 8)
      """
      a = iter(iterable)
      return zip(*[a] * n)

    def _d(x):
      return x.detach().cpu().numpy()

    def merged_actions(actions):
      action_keys = list(ACTION_SPACE.keys())

      groups = [groupby(_d(actions[k]), 2) for k in action_keys]
      for g in zip(*groups):
        x = {key: g[i][0] for i, key in enumerate(action_keys)}
        y = {key: g[i][1] for i, key in enumerate(action_keys)}
        yield x, y

    def single_player_actions(actions):
      action_keys = list(ACTION_SPACE.keys())
      groups = [groupby(_d(actions[k]), 1) for k in action_keys]
      for g in zip(*groups):
        x = {key: g[i][0] for i, key in enumerate(action_keys)}
        yield x

    if self.use_single_player:
      self.last_outs = [
          env.step(a)
          for env, a in zip(self.envs, single_player_actions(actions))
      ]
    else:
      self.last_outs = [
          env.step(a) for env, a in zip(self.envs, merged_actions(actions))
      ]
    return self._vectorize_env_outs(self.last_outs)

  def render(self, idx: int, mode: str = None, **kwargs):
    # noinspection PyArgumentList
    return self.envs[idx].render(mode, **kwargs)

  def close(self):
    return [env.close() for env in self.envs]

  def seed(self, seed: Optional[int] = None) -> list:
    if seed is not None:
      return [env.seed(seed + i) for i, env in enumerate(self.envs)]
    else:
      return [env.seed(seed) for i, env in enumerate(self.envs)]

  @property
  def unwrapped(self) -> List[gym.Env]:
    return [env.unwrapped for env in self.envs]

  @property
  def action_space(self) -> List[gym.spaces.Dict]:
    return [env.action_space for env in self.envs]

  @property
  def observation_space(self) -> List[gym.spaces.Dict]:
    return [env.observation_space for env in self.envs]

  @property
  def single_action_space(self):
    return self.envs[0].action_space

  @property
  def single_observation_space(self):
    return self.envs[0].observation_space

  @property
  def metadata(self) -> List[Dict]:
    return [env.metadata for env in self.envs]


class PytorchEnv(gym.Wrapper):

  def __init__(self,
               env: Union[gym.Env, VecEnv],
               device: torch.device = torch.device("cpu")):
    super(PytorchEnv, self).__init__(env)
    self.device = device

  @property
  def single_action_space(self):
    return self.env.single_action_space

  @property
  def single_observation_space(self):
    return self.env.single_observation_space

  def reset(self, **kwargs):
    return tuple([
        self._to_tensor(out, key_hint='reset')
        for out in super(PytorchEnv, self).reset(**kwargs)
    ])

  def step(self, action: torch.Tensor):
    return tuple([
        self._to_tensor(out, key_hint='step')
        for out in super(PytorchEnv, self).step(action)
    ])

  def _to_tensor(self,
                 x: Union[Dict, np.ndarray],
                 key_hint=None) -> Dict[str, Union[Dict, torch.Tensor]]:
    if isinstance(x, dict):
      return {
          key: self._to_tensor(val, key_hint=key)
          for key, val in x.items()
      }
    else:
      dtype = torch.float32
      if x.dtype == np.int32:
        dtype = torch.int32

      return torch.from_numpy(x).to(self.device, non_blocking=True).to(dtype)
      # return torch.from_numpy(x).to(self.device).to(dtype)


class DictEnv(gym.Wrapper):

  @staticmethod
  def _dict_env_out(env_out: tuple) -> dict:
    obs, reward, done, info = env_out
    assert "obs" not in info.keys()
    assert "reward" not in info.keys()
    assert "done" not in info.keys()
    return dict(obs=obs, reward=reward, done=done, info=info)

  @property
  def single_action_space(self):
    return self.env.single_action_space

  @property
  def single_observation_space(self):
    return self.env.single_observation_space

  def reset(self, **kwargs):
    return DictEnv._dict_env_out(super(DictEnv, self).reset(**kwargs))

  def step(self, action):
    v = DictEnv._dict_env_out(super(DictEnv, self).step(action))
    # print([(i, x) for i, x in enumerate(zip(v['obs'].items(), v['done']))])
    return v

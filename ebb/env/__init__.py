import torch
from typing import Optional

from .luxenv import LuxS3Env, MapManager
from .wrappers import VecEnv, PytorchEnv, DictEnv


def create_env(flags, device: torch.device) -> DictEnv:
  envs = []
  for _ in range(flags.n_actor_envs):
    env = LuxS3Env(flags.reward_schema,
                   obs_space_kwargs=flags.obs_space_kwargs,
                   reward_shaping_params=flags.reward_shaping_params)
    envs.append(env)
  env = VecEnv(envs)
  env = PytorchEnv(env, device)
  env = DictEnv(env)
  return env

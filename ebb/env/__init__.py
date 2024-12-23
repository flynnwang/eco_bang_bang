import torch
from typing import Optional

from .luxenv import LuxS3Env, MapManager
from .wrappers import VecEnv, PytorchEnv, DictEnv
from .const import *


def create_env(flags, device: torch.device) -> DictEnv:
  envs = []
  for _ in range(flags.n_actor_envs):
    env = LuxS3Env(reward_schema=flags.reward_schema)
    envs.append(env)
  env = VecEnv(envs)
  env = PytorchEnv(env, device)
  env = DictEnv(env)
  return env

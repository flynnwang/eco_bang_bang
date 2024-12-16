import torch
from typing import Optional

from .delivery import DeliveryEnv
from .wrappers import VecEnv, PytorchEnv, DictEnv
from .const import *


def create_env(flags, device: torch.device) -> DictEnv:
  envs = []
  for _ in range(flags.n_actor_envs):
    env = DeliveryEnv(reward_schema=flags.reward_schema)
    envs.append(env)
  # env = VecEnv(envs, N_ACTION_PLAYERS * 2) # player model
  env = VecEnv(envs, 1)  # team model
  env = PytorchEnv(env, device)
  env = DictEnv(env)
  return env

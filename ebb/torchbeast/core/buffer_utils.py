from copy import copy
import gym
import numpy as np
from typing import Any, Callable, Dict, List, Tuple, Union

import torch

from ebb.env.const import *

Buffers = List[Dict[str, Union[Dict, torch.Tensor]]]


def fill_buffers_inplace(buffers: Union[Dict, torch.Tensor],
                         fill_vals: Union[Dict, torch.Tensor],
                         step: int,
                         key_hint=None):
  if isinstance(fill_vals, dict):
    for key, val in copy(fill_vals).items():
      fill_buffers_inplace(buffers[key], val, step, key_hint=key)
  else:
    # if key_hint == "unit_move_cost":
    # __import__('ipdb').set_trace()
    buffers[step, ...] = fill_vals[:]


def stack_buffers(buffers: Buffers,
                  dim: int) -> Dict[str, Union[Dict, torch.Tensor]]:
  stacked_buffers = {}
  for key, val in copy(buffers[0]).items():
    if isinstance(val, dict):
      stacked_buffers[key] = stack_buffers([b[key] for b in buffers], dim)
    else:
      stacked_buffers[key] = torch.cat([b[key] for b in buffers], dim=dim)
  return stacked_buffers


def split_buffers(
    buffers: Dict[str, Union[Dict, torch.Tensor]],
    split_size_or_sections: Union[int, List[int]],
    dim: int,
    contiguous: bool,
) -> List[Union[Dict, torch.Tensor]]:
  buffers_split = None
  for key, val in copy(buffers).items():
    if isinstance(val, dict):
      bufs = split_buffers(val, split_size_or_sections, dim, contiguous)
    else:
      bufs = torch.split(val, split_size_or_sections, dim=dim)
      if contiguous:
        bufs = [b.contiguous() for b in bufs]

    if buffers_split is None:
      buffers_split = [{} for _ in range(len(bufs))]
    assert len(bufs) == len(
        buffers_split
    ), f"len(bufs)={len(bufs)},  len(buffers_split)={len(buffers_split)}"
    buffers_split = [
        dict(**{key: buf}, **d) for buf, d in zip(bufs, buffers_split)
    ]
  return buffers_split


def buffers_apply(buffers: Union[Dict, torch.Tensor],
                  func: Callable[[torch.Tensor], Any],
                  key_hint=None) -> Union[Dict, torch.Tensor]:
  if isinstance(buffers, dict):
    return {
        key: buffers_apply(val, func, key_hint=key)
        for key, val in copy(buffers).items()
    }
  else:
    return func(buffers)


def _create_buffers_from_specs(
    specs: Dict[str, Union[Dict, Tuple, torch.dtype]]
) -> Union[Dict, torch.Tensor]:
  if isinstance(specs, dict) and "dtype" not in specs.keys():
    new_buffers = {}
    for key, val in specs.items():
      new_buffers[key] = _create_buffers_from_specs(val)
    return new_buffers
  else:
    return torch.empty(**specs).share_memory_()


def _create_buffers_like(buffers: Union[Dict, torch.Tensor],
                         t_dim: int) -> Union[Dict, torch.Tensor]:
  if isinstance(buffers, dict):
    torch_buffers = {}
    for key, val in buffers.items():
      torch_buffers[key] = _create_buffers_like(val, t_dim)
    return torch_buffers
  else:
    buffers = buffers.unsqueeze(0).expand(
        t_dim, *[-1 for _ in range(len(buffers.shape))])
    return torch.empty_like(buffers).share_memory_()


def create_buffers(
    flags, obs_space, example_info: Dict[str, Union[Dict, np.ndarray,
                                                    torch.Tensor]]) -> Buffers:
  t = flags.unroll_length
  n = flags.n_actor_envs
  p = 2
  # n = flags.n_actor_envs * p  # two players
  # n = flags.batch_size * 2  # two players
  # n = flags.n_actor_envs  # single player mode

  # observation_space is expected to be a dict of simple spaces.
  obs_specs = {}
  for key, spec in obs_space.spaces.items():
    if isinstance(spec, gym.spaces.MultiBinary):
      dtype = torch.int64
    elif isinstance(spec, gym.spaces.MultiDiscrete):
      dtype = torch.int64
    elif isinstance(spec, gym.spaces.Box):
      dtype = torch.float32
    else:
      raise NotImplementedError(
          f"{type(spec)} is not an accepted observation space.")
    shape = spec.shape[1:]  # drop first dimension of size 1
    # shape = spec.shape
    obs_specs[key] = dict(size=(t + 1, n * p, *shape), dtype=dtype)
    print(key, '(t + 1, n, *shape)=', (t + 1, n * p, *shape))

  # TODO: create state-action also by example?
  # create_buffers is related to action space is sutle ways, move it to env?
  specs = dict(
      obs=obs_specs,
      #
      # action needs to be int64 for torch.gather
      actions={
          UNITS_ACTION:
          dict(size=(t + 1, n, p, MAX_UNIT_NUM, 1), dtype=torch.int64),
      },
      policy_logits={
          UNITS_ACTION:
          dict(size=(t + 1, n, p, MAX_UNIT_NUM, ALL_ACTION_NUM),
               dtype=torch.float32),
      },
      #
      baseline=dict(size=(t + 1, n, p), dtype=torch.float32),
      #
      reward=dict(size=(t + 1, n, p), dtype=torch.float32),
      done=dict(size=(t + 1, n), dtype=torch.bool),
  )

  buffers: Buffers = []
  for _ in range(flags.num_buffers):
    new_buffer = _create_buffers_from_specs(specs)
    new_buffer["info"] = _create_buffers_like(example_info, t + 1)
    buffers.append(new_buffer)
  return buffers

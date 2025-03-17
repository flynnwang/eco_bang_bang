from typing import Callable, Dict, Optional, Tuple, Union, NamedTuple, Any, List
from argparse import Namespace
import os
import sys

import numpy as np
import torch

from .env.const import *
from .env.luxenv import (
    MapManager,
    LuxS3Env,
    SapIndexer,
    anti_diag_sym,
    anti_diag_sym_i,
    EXT_ACTION_SHAPE,
)
from .model import create_model

SUBMIT_AGENT = True

MODEL_FILE_NAME = "063230272_weights.pt"

DO_SAMPLE = True
USE_MIRROR_TRANS = False

DEVICE = 'cpu'
if not SUBMIT_AGENT:
  # import random
  # DEVICE = random.sample(['cuda:0', 'cuda:1'], k=1)[0]
  DEVICE = 'cuda:0'


def _to_tensor(x: Union[Dict, np.ndarray],
               device) -> Dict[str, Union[Dict, torch.Tensor]]:
  if isinstance(x, dict):
    return {key: _to_tensor(val, device) for key, val in x.items()}
  else:
    dtype = torch.float32
    if x.dtype == np.int32:
      dtype = torch.int32
    if x.dtype == np.int64:
      dtype = torch.int64
    return torch.from_numpy(x).to(device, non_blocking=True).to(dtype)


def _stack_dict(x: List[Union[Dict, np.ndarray]],
                is_observation=False) -> Union[Dict, np.ndarray]:
  if isinstance(x[0], dict):
    return {
        key: _stack_dict([i[key] for i in x], is_observation)
        for key in x[0].keys()
    }
  else:
    if is_observation:
      return np.concatenate([arr for arr in x], axis=0)
    # for i in range(4):
    # print(i, x[i].shape, file=sys.stderr)
    return np.stack([arr for arr in x], axis=0)


def mirror_transpose_obs(x, mirror, transpose):
  if isinstance(x, dict):
    return {
        key: mirror_transpose_obs(val, mirror, transpose)
        for key, val in x.items()
    }
  else:

    if x.shape == (1, MAP_WIDTH, MAP_HEIGHT):
      assert x.shape[0] == 1

      # print(f'orig shape: {x.shape}', file=sys.stderr)
      if mirror:
        x = np.expand_dims(anti_diag_sym(x[0]), axis=0)

      # print(f'pos1 shape: {x.shape}', file=sys.stderr)
      if transpose:
        x = np.expand_dims(x[0].T, axis=0)
      # print(f'pos2 shape: {x.shape}', file=sys.stderr)
    return x


def mirror_transpose_action_masks(a, mirror, transpose, sap_indexer):
  a = a[UNITS_ACTION]
  actions_mask = np.zeros(EXT_ACTION_SHAPE, dtype=bool)
  for i in range(MAX_UNIT_NUM):
    actions_mask[i][ACTION_CENTER] = a[i][ACTION_CENTER]
    for j in range(1, MAX_MOVE_ACTION_IDX + 1):
      j2 = j
      if mirror:
        j2 = MIRRORED_ACTION[j2]

      if transpose:
        j2 = TRANSPOSED_ACTION[j2]

      actions_mask[i][j2] = a[i][j]

    for j in range(SAP_ACTION_NUM):
      x, y = sap_indexer.idx_to_position[j]
      if mirror:
        x, y = y, x

      if transpose:
        x, y = -y, -x

      j2 = sap_indexer.position_to_idx[(x, y)]
      actions_mask[i][j2 + MOVE_ACTION_NUM] = a[i][j + MOVE_ACTION_NUM]
  return {UNITS_ACTION: actions_mask}


def restore_action_probs(action_probs, mirror, transpose, sap_indexer):
  restored_probs = torch.zeros_like(action_probs)
  for i in range(MAX_UNIT_NUM):
    restored_probs[i][ACTION_CENTER] = action_probs[i][ACTION_CENTER]
    for j in range(1, MAX_MOVE_ACTION_IDX + 1):
      j2 = j
      if transpose:
        j2 = TRANSPOSED_ACTION[j2]

      if mirror:
        j2 = MIRRORED_ACTION[j2]

      restored_probs[i][j2] = action_probs[i][j]

    for j in range(SAP_ACTION_NUM):
      x, y = sap_indexer.idx_to_position[j]
      if transpose:
        x, y = -y, -x

      if mirror:
        x, y = y, x

      j2 = sap_indexer.position_to_idx[(x, y)]
      restored_probs[i][j2 +
                        MOVE_ACTION_NUM] = action_probs[i][j + MOVE_ACTION_NUM]
  return restored_probs


class Agent:

  def __init__(self, player: str, env_cfg) -> None:
    self.player = player
    self.env_cfg = env_cfg
    # np.random.seed(0)

    obs_space_kwargs = {'use_energy_cost_map': True}

    self.mm = MapManager(player,
                         env_cfg,
                         transpose=False,
                         sap_indexer=SapIndexer(),
                         use_mirror=False,
                         use_hidden_relic_estimator=True)
    self.env = LuxS3Env("", obs_space_kwargs=obs_space_kwargs,
                        game_env=1)  # for calling _convert_observation
    self.env.sap_indexer = self.mm.sap_indexer
    assert self.env.sap_indexer is not None

    self.prev_model_action = None
    self.md = self.load_model()

  def load_model(self):
    flags = dict(n_blocks=16,
                 hidden_dim=128,
                 base_out_channels=128,
                 embedding_dim=32,
                 kernel_size=5,
                 use_separate_base=False,
                 reward_schema="game_win_loss2")
    flags = Namespace(**flags)
    model = create_model(flags, self.env.observation_space, device=DEVICE)
    # print(f"Model created", file=sys.stderr)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              MODEL_FILE_NAME)
    if not os.path.exists(model_path):
      print(f"Model file not found: {model_path}", file=sys.stderr)
      raise RuntimeError("model not found")

    # print(f"Loading model...", file=sys.stderr)
    checkpoint_state = torch.load(model_path,
                                  map_location=DEVICE,
                                  weights_only=True)
    model.load_state_dict(checkpoint_state["model_state_dict"])
    print(f"Model loaded", file=sys.stderr)
    return model

  def convert_observation(self, raw_obs):
    obs = self.env._convert_observation(raw_obs, self.mm, skip_check=True)
    action_mask = self.env._get_available_action_mask(self.mm)

    if USE_MIRROR_TRANS:
      ob3 = mirror_transpose_obs(obs, mirror=True, transpose=True)
      am3 = mirror_transpose_action_masks(action_mask,
                                          mirror=True,
                                          transpose=True,
                                          sap_indexer=self.mm.sap_indexer)
      ob2 = mirror_transpose_obs(obs, mirror=True, transpose=False)
      am2 = mirror_transpose_action_masks(action_mask,
                                          mirror=True,
                                          transpose=False,
                                          sap_indexer=self.mm.sap_indexer)
      ob1 = mirror_transpose_obs(obs, mirror=False, transpose=True)
      am1 = mirror_transpose_action_masks(action_mask,
                                          mirror=False,
                                          transpose=True,
                                          sap_indexer=self.mm.sap_indexer)

      obs_list = [obs, ob1, ob2, ob3]
      am_list = [action_mask, am1, am2, am2]
    else:
      obs_list = [obs]
      am_list = [action_mask]

    o = _stack_dict(obs_list, is_observation=True)
    a = _stack_dict(am_list)

    dev = torch.device(DEVICE)
    model_input = {
        "obs": _to_tensor(o, dev),
        "info": {
            "available_action_mask": _to_tensor(a, dev),
        }
    }
    return model_input

  def get_avg_model_action(self, action_probs):
    """model_probs=torch.Size([64, 118])"""
    if USE_MIRROR_TRANS:
      action_probs = action_probs.reshape(4, MAX_UNIT_NUM, -1)
      action_probs[1] = restore_action_probs(action_probs[1],
                                             mirror=False,
                                             transpose=True,
                                             sap_indexer=self.mm.sap_indexer)
      action_probs[2] = restore_action_probs(action_probs[2],
                                             mirror=True,
                                             transpose=False,
                                             sap_indexer=self.mm.sap_indexer)
      action_probs[3] = restore_action_probs(action_probs[3],
                                             mirror=True,
                                             transpose=True,
                                             sap_indexer=self.mm.sap_indexer)
      action_probs = action_probs.mean(dim=0)

    actions = torch.multinomial(action_probs, num_samples=1, replacement=False)
    return actions

  def act(self, step: int, raw_obs, remainingOverageTime: int = 60):
    """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
    self.mm.update(raw_obs, self.prev_model_action)
    self._available_action_mask = self.env._get_available_action_mask(self.mm)

    model_input = self.convert_observation(raw_obs)
    # print(model_input, file=sys.stderr)

    model_output = self.md(model_input, sample=DO_SAMPLE, probs_output=True)
    self.model_output = model_output
    action_probs = self.get_avg_model_action(model_output["probs"])

    model_action = {UNITS_ACTION: action_probs}
    self.prev_model_action = model_action

    # print(
    # f'model_action.shape={model_action[UNITS_ACTION].shape}, model_probs={action_probs.shape}',
    # file=sys.stderr)

    # model_action[UNITS_ACTION] = model_action[UNITS_ACTION].squeeze(0)
    action_taken_mask = self.env.get_actions_taken_mask(model_action, self.mm)
    action = self.env._encode_action(model_action, self.mm, action_taken_mask)
    return action

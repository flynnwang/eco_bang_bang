from typing import Callable, Dict, Optional, Tuple, Union, NamedTuple, Any
from argparse import Namespace
import os
import sys

import numpy as np
import torch

from .env.const import UNITS_ACTION
from .env.luxenv import MapManager, LuxS3Env, SapIndexer
from .model import create_model

SUBMIT_AGENT = False

MODEL_FILE_NAME = "051062272_weights.pt"

DO_SAMPLE = True

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
    return torch.from_numpy(x).to(device, non_blocking=True).to(dtype)


class Agent:

  def __init__(self, player: str, env_cfg) -> None:
    self.player = player
    self.env_cfg = env_cfg
    # np.random.seed(0)

    self.mm = MapManager(player, env_cfg, transpose=False)
    self.env = LuxS3Env(game_env=1)  # for calling _convert_observation
    self.env.sap_indexer = SapIndexer()
    assert self.env.sap_indexer is not None

    self.prev_model_action = None
    self.md = self.load_model()

  def load_model(self):
    flags = dict(n_blocks=12,
                 hidden_dim=128,
                 base_out_channels=128,
                 embedding_dim=32,
                 kernel_size=5,
                 reward_schema="shaping")
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
    if not self.env.prev_raw_obs:
      self.env.prev_raw_obs = {self.mm.player: raw_obs}

    obs = self.env._convert_observation(raw_obs, self.mm, skip_check=True)
    action_mask = self.env._get_available_action_mask(self.mm)
    dev = torch.device(DEVICE)
    model_input = {
        "obs": _to_tensor(obs, dev),
        "info": {
            "available_action_mask": _to_tensor(action_mask, dev),
        }
    }
    return model_input

  def act(self, step: int, raw_obs, remainingOverageTime: int = 60):
    """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
    self.mm.update(raw_obs, self.prev_model_action)

    model_input = self.convert_observation(raw_obs)
    model_action = self.md(model_input, sample=DO_SAMPLE)["actions"]

    # print(model_input, file=sys.stderr)
    self.prev_model_action = model_action

    model_action[UNITS_ACTION] = model_action[UNITS_ACTION].squeeze(0)
    action_taken_mask = self.env.get_actions_taken_mask(model_action, self.mm)
    action = self.env._encode_action(model_action, self.mm, action_taken_mask)
    return action

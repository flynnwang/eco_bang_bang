from copy import deepcopy
from collections import OrderedDict, deque, defaultdict
from functools import lru_cache

import gym
import numpy as np
from gym import spaces
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from .const import *

PLAYER0 = 'player_0'
PLAYER1 = 'player_1'

MAX_UNIT_NUM = 16

MAX_GAME_STEPS = 504
MAX_MATCH_STEPS = 100

# 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap
MOVE_ACTION_NUM = 5

# Let's use move action only first
ACTION_SPACE = spaces.MultiDiscrete(
    np.zeros((MAX_UNIT_NUM, ), dtype=int) + MOVE_ACTION_NUM)

OB = OrderedDict([
    ('map_features',
     spaces.MultiDiscrete(np.zeros(VIEW_SHAPE) + MAP_FEATURE_NUM)),
    ('time_feature', spaces.Box(low=0, high=1, shape=VIEW_SHAPE)),
    ('delta_reward', spaces.Box(low=0, high=1, shape=VIEW_SHAPE)),

    # agent
    ('agent_position', spaces.Box(low=0, high=1, shape=VIEW_SHAPE)),
    ('agent_can_grab_num', spaces.Box(low=0, high=1, shape=VIEW_SHAPE)),
    ('agent_order_to_pick_num', spaces.Box(low=0, high=1, shape=VIEW_SHAPE)),
    ('agent_order_to_deliver_num', spaces.Box(low=0, high=1,
                                              shape=VIEW_SHAPE)),
    ('agent_total_reward', spaces.Box(low=0, high=1, shape=VIEW_SHAPE)),
    # potential_map
    ('potential_map', spaces.Box(low=0, high=1, shape=VIEW_SHAPE)),
])
OBSERVATION_SPACE = spaces.Dict(OB)


def game_win_loss(scores, player_id):
  if scores[0] == scores[1]:
    return 0
  return np.argmax(scores) == player_id and 1 or -1


def manhatten_distance(p1, p2):
  return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def gen_dummy_action():
  # TODO:
  pass


class LuxS3Env(gym.Env):

  def __init__(self, reward_schema=None):
    self.reward_schema = reward_schema
    self.game = LuxAIS3GymEnv(numpy_output=True)

  @property
  def total_agent_controls(self):
    return 2

  @property
  def action_space(self):
    """Action space of single player."""
    return ACTION_SPACE

  @property
  def observation_space(self):
    """Obervation space of single player"""
    return OBSERVATION_SPACE

  def seed(self, seed):
    pass

  def reset(self, seed=None):
    raw_obs, info = self.game.reset()

    self.prev_raw_obs = raw_obs
    done = False
    reward = self._convert_reward(raw_obs, info)
    action = [gen_dummy_action(), gen_dummy_action()]
    info = self.get_info(action, raw_obs, model_action=None)

    return self.observation(raw_obs), reward, done, info

  def _encode_action(self, action):
    """Translate the model action into game env action.

    TODO: to encode SAP action, observation is required.
    """

    def _encode(single_player_actions):
      unit_actions = np.zeros((MAX_UNIT_NUM, 3), dtype=np.int16)

      for i, a in enumerate(single_player_actions):
        unit_actions[i][0] = np.int16(a)

      return _encode(action[0], self.prev_raw_obs[0])

  def step(self, model_action):
    action = {
        PLAYER0: self._encode_action(model_action[0]),
        PLAYER1: self._encode_action(model_action[1]),
    }
    raw_obs, step_reward, _, info = self.game.step(action)
    done = False
    if raw_obs[PLAYER0]['steps'] >= MAX_GAME_STEPS:
      done = True

    obs = self.observation(raw_obs)
    reward = self._convert_reward(raw_obs, info)
    info = self.get_info(action, raw_obs, model_action)

    self.prev_raw_obs = raw_obs
    return obs, reward, done, info

  def observation(self, raw_obs, action=None):
    assert len(
        raw_obs
    ) == 2, f"len(raw_obs)={len(raw_obs)}, self.total_agent_controls={self.total_agent_controls}"

    return [
        self._convert_observation(raw_obs[0]),
        self._convert_observation(raw_obs[1])
    ]

  def _convert_observation(self, raw_obs1):
    # TODO:
    obs = {}
    return obs

  def _convert_reward(self, raw_obs, info):
    """Use the match win-loss reward for now."""
    assert self.reward_schema == 'match_win_loss'

    team_wins = raw_obs[PLAYER0]['team_wins']
    prev_team_wins = self.prev_raw_obs[PLAYER0]['team_wins']
    reward = team_wins - prev_team_wins
    return reward

  def _get_action_mask(self, raw_obs1):
    """Mask for unit action."""
    return {}

  def get_actions_taken_mask(self, action):
    return {}

  def get_info(self, action, raw_obs, model_action):

    def _info(agent_action, raw_obs1, prev_obs1, model_action):
      info = {}

      # action mask matches with given action for last state (for compute logits)
      max_grab_num = MAX_DISTRIBUTE_NUM - len(agent['orders_to_pick'])
      info['actions_taken_mask'] = self.get_actions_taken_mask(
          model_action, max_grab_num)

      # action mask for current state, (for sample action)
      info['available_action_mask'] = self._get_action_mask(raw_obs1)

      return info

    return [
        _info(action[i], raw_obs[i], self.prev_raw_obs[i], model_action)
        for i in range(len(raw_obs))
    ]

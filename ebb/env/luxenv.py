from copy import deepcopy
from collections import OrderedDict, deque, defaultdict
from functools import lru_cache

import gym
import numpy as np
from gym import spaces
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from .const import *

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


from scipy.ndimage import maximum_filter


def anti_diag_sym(A):
  A = np.flipud(A)
  A = np.fliplr(A)
  A = A.T
  return A


class MapManager:

  def __init__(self, player, env_cfg):
    self.player_id = int(player[-1])
    self.env_cfg = env_cfg
    self.cell_type = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.visible = None
    self.observed = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.visited = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.is_relic_node = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.is_relic_neighbour = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)

    self.prev_team_point = 0
    self.team_point_mass = np.ones((MAP_WIDTH, MAP_HEIGHT), np.float32)

  def update(self, ob):
    # Match restarted
    if ob['match_steps'] == 0:
      self.prev_team_point = 0

    self.visible = ob['sensor_mask'].astype(np.int32)
    self.observed |= self.visible

    # adding 1 to start cell type from 0
    cells = ob['map_features']['tile_type'] + 1

    # Update map cell type
    c = cells > CELL_UNKONWN
    self.cell_type[c] = cells[c]

    # also update the symmetrical positions
    cells_sym = anti_diag_sym(cells)
    ct = cells_sym > CELL_UNKONWN
    self.cell_type[ct] = cells_sym[ct]

    unit_masks = ob['units_mask'][self.player_id]
    unit_positions = ob['units']['position'][self.player_id][unit_masks]
    self.visited[unit_positions[:, 0], unit_positions[:, 1]] = 1

    relic_nodes_mask = ob['relic_nodes_mask']
    relic_nodes_positions = ob['relic_nodes'][relic_nodes_mask]
    self.is_relic_node[relic_nodes_positions[:, 0],
                       relic_nodes_positions[:, 1]] = 1
    self.is_relic_node |= anti_diag_sym(self.is_relic_node)
    self.is_relic_neighbour = maximum_filter(
        (self.is_relic_node == 1).astype(np.int32), size=RELIC_NB_SIZE)

    self.update_team_point_mass(ob, unit_positions)
    self.prev_team_point = team_point

  def update_team_point_mass(self, ob, unit_positions):
    """Update team point confidence"""
    team_point = ob['team_points'][self.player_id]
    # print(ob['steps'], ob['match_steps'], 'team_point=', team_point, 'prev_point', self.prev_team_point)
    if ob['match_steps'] == 0 or len(unit_positions):
      return

    unit_pos_mask = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    unit_pos_mask[unit_positions[:, 0], unit_positions[:, 1]] = True
    unit_nearby_relic = (self.is_relic_neighbour > 0) & (unit_pos_mask)
    if unit_nearby_relic.sum() == 0:
      return

    must_be_team_point = (self.team_point_mass
                          >= TEAM_POINT_MASS) & (unit_nearby_relic)
    non_team_point = (self.team_point_mass
                      <= NON_TEAM_POINT_MASS) & (unit_nearby_relic)

    delta = team_point - self.prev_team_point
    delta -= must_be_team_point.sum()

    team_point_candidate = unit_nearby_relic & (~must_be_team_point) & (
        ~non_team_point)
    num = team_point_candidate.sum()
    if num > 0:
      if delta == 0:
        self.team_point_mass[team_point_candidate] = NON_TEAM_POINT_MASS
        self.team_point_mass[anti_diag_sym(
            team_point_candidate)] = NON_TEAM_POINT_MASS
      elif num == delta:
        self.team_point_mass[team_point_candidate] = TEAM_POINT_MASS
        self.team_point_mass[anti_diag_sym(
            team_point_candidate)] = TEAM_POINT_MASS
      else:
        assert delta < num
        # print('>>>>>>>>>>>>>>', ob['steps'], delta, num, must_be_team_point.sum(), non_team_point.sum())
        self.team_point_mass[anti_diag_sym(
            team_point_candidate)] += delta / num
      assert delta >= 0


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

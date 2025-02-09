from collections import OrderedDict, deque, defaultdict, Counter
from functools import cached_property
import copy
import random
import sys

import gym

gym.logger.set_level(40)

import numpy as np
from gym import spaces
from luxai_s3.wrappers import LuxAIS3GymEnv
from scipy.ndimage import maximum_filter, minimum_filter

from random import randint

from .const import *
from .mapmanager import *
from ..agent import Agent

# Let's use move action only first
ACTION_SPACE = spaces.Dict({
    UNITS_ACTION:
    spaces.MultiDiscrete(np.zeros(MAX_UNIT_NUM, dtype=int) + ALL_ACTION_NUM)
})

OB = OrderedDict([
    # Game params
    # Time & Match
    ('game_step', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('match_step', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    #  Game params
    ('unit_move_cost',
     spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + (MAX_MOVE_COST + 1))),
    ('unit_sensor_range',
     spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + (MAX_SENSOR_RANGE + 1))),
    ('unit_sap_cost',
     spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + (MAX_SAP_COST + 1))),
    ('unit_sap_range',
     spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + (MAX_SAP_RANGE + 1))),
    ('nebula_vision_reduction',
     spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + N_NEBULA_VISION_REDUCTION)),
    ('nebula_energy_reduction',
     spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + N_NEBULA_ENERGY_REDUCTION)),
    ('unit_sap_dropoff_factor',
     spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + N_SAP_DROPOFF_FACTOR)),
    # Scores
    ('player_id', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_team_points', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('team_points_delta', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_team_points_growth', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('team_points_growth_delta', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_wins', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('match_wins_delta', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_total_energy', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),

    # Map info
    ('_a_cell_type', spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + N_CELL_TYPES)),
    ('_b_cell_type', spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + N_CELL_TYPES)),
    ('visible', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('last_observed_age', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('last_visited_age', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('_a_is_relic_node', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('_b_is_relic_node', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),

    # use this to indicate nodes of unvisited relic cells (and its neighbour)
    ('_a_is_relic_neighbour', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('_b_is_relic_neighbour', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    # use this to indicate the hidden place of relc nodes.
    ('_a_team_point_prob', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('_b_team_point_prob', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    # hints for where to go
    ('_a_energy_cost_map', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('_b_energy_cost_map', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    #
    ('_a_cell_energy', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('_b_cell_energy', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),

    # units team map
    ('units_loc_t0', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_energy_t0', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_loc_t1', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_energy_t1', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('enemy_loc_t0', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('enemy_energy_t0', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('enemy_loc_t1', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('enemy_energy_t1', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),

    # Extra baseline mode feature
    ("_baseline_extras",
     spaces.Box(
         np.zeros((1, N_BASELINE_EXTRA_DIM)) - 1,
         np.zeros((1, N_BASELINE_EXTRA_DIM)) + 1)),

    # Per Units info
    ('_units_info',
     spaces.MultiDiscrete([[(MAP_WIDTH, MAP_HEIGHT, MAX_UNIT_ENERGY)] *
                           MAX_UNIT_NUM])),  # shape: (1, 16, 3)
])


def get_ob_sapce(obs_space_kwargs):
  ob = copy.copy(OB)
  if obs_space_kwargs.get('use_energy_cost_map'):
    ob['_a_energy_cost_map'] = spaces.Box(low=0, high=1, shape=MAP_SHAPE)
    ob['_b_energy_cost_map'] = spaces.Box(low=0, high=1, shape=MAP_SHAPE)
  return spaces.Dict(ob)


class LuxS3Env(gym.Env):

  def __init__(self,
               reward_schema,
               obs_space_kwargs,
               game_env=None,
               reward_shaping_params=None):
    self.reward_schema = reward_schema
    self.obs_space_kwargs = obs_space_kwargs
    self.use_agent = obs_space_kwargs.get("use_agent")
    self.game = game_env or LuxAIS3GymEnv(numpy_output=True)
    self.reward_shaping_params = reward_shaping_params
    self.mms = None
    self.prev_raw_obs = None
    self._seed = None
    self.last_sap_locations = [[], []]
    self.agent_actions = [None, None]

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
    return get_ob_sapce(self.obs_space_kwargs)

  @property
  def use_single_player(self):
    """Obervation space of single player"""
    return self.obs_space_kwargs["use_single_player"]

  def seed(self, seed):
    self._seed = seed

  def _update_mms(self, obs, model_actions, env_state=None):
    a0, a1 = None, None
    if model_actions is not None:
      if self.use_agent:
        self.last_sap_locations[0] = self.agents[0].last_sap_locations
        self.last_sap_locations[1] = self.agents[1].last_sap_locations
      else:
        a0, a1 = model_actions
        self.last_sap_locations[0] = self.mms[0].to_last_sap_actions(a0)
        self.last_sap_locations[1] = self.mms[1].to_last_sap_actions(a1)

    self.mms[0].update(obs[PLAYER0], a0, env_state)
    self.mms[1].update(obs[PLAYER1], a1, env_state)

    self.mms[0].add_sap_locations(self.last_sap_locations[0])
    self.mms[1].add_sap_locations(self.last_sap_locations[1])

  def reset(self, seed=None):
    if seed is None:
      self._seed = randint(-(1 << 31), 1 << 31)
    else:
      self._seed = seed
    self._sum_r = 0.0

    raw_obs, info = self.game.reset(seed=self._seed)

    # TODO: delete
    # from luxai_s3.params import EnvParams
    # env_params = EnvParams(nebula_tile_vision_reduction=3,
    # nebula_tile_energy_reduction=25,
    # unit_sensor_range=2)
    # raw_obs, info = self.game.reset(seed=self._seed,
    # options={'params': env_params})

    final_state = info['state']

    env_cfg = info['params']
    full_params = info['full_params']

    sap_indexer = SapIndexer()

    # tr1 = (self._seed % 2 == 0)
    # tr2 = ((self._seed // 2) % 2 == 0)
    # mirror1 = ((self._seed // 4) % 2 == 0)
    # mirror2 = ((self._seed // 8) % 2 == 0)
    tr1 = tr2 = mirror1 = mirror2 = False
    use_hidden_relic_estimator = self.reward_shaping_params[
        'use_hidden_relic_estimator']
    self.mms = [
        MapManager(PLAYER0,
                   env_cfg,
                   transpose=tr1,
                   sap_indexer=sap_indexer,
                   use_mirror=mirror1,
                   use_hidden_relic_estimator=use_hidden_relic_estimator,
                   full_params=full_params),
        MapManager(PLAYER1,
                   env_cfg,
                   transpose=tr2,
                   sap_indexer=sap_indexer,
                   use_mirror=mirror2,
                   use_hidden_relic_estimator=use_hidden_relic_estimator,
                   full_params=full_params),
    ]
    self._update_mms(raw_obs, model_actions=None, env_state=final_state)
    self.sap_indexer = sap_indexer
    self.mms[0].env = self
    self.mms[1].env = self

    self.agents = [None, None]
    if self.use_agent:
      self.agents = [
          Agent(self.mms[0].player, env_cfg),
          Agent(self.mms[1].player, env_cfg),
      ]
      self.agents[0].env = self
      self.agents[1].env = self

    self.prev_raw_obs = raw_obs
    done = False
    reward = self._convert_reward(raw_obs, final_state)

    self._actions_taken_mask = [{
        UNITS_ACTION:
        np.zeros(EXT_ACTION_SHAPE, dtype=bool)
    }, {
        UNITS_ACTION:
        np.zeros(EXT_ACTION_SHAPE, dtype=bool)
    }]
    self.agent_actions = [
        np.zeros((MAX_UNIT_NUM, 3), dtype=int),
        np.zeros((MAX_UNIT_NUM, 3), dtype=int),
    ]
    model_action = [{
        UNITS_ACTION: np.zeros((MAX_UNIT_NUM, 1), dtype=int)
    }, {
        UNITS_ACTION: np.zeros((MAX_UNIT_NUM, 1), dtype=int)
    }]
    info = self.get_info(model_action,
                         raw_obs,
                         reward,
                         done,
                         env_state=final_state)

    self.prev_raw_ob = self.raw_obs = raw_obs
    return self.observation(raw_obs, final_state), reward, done, info

  def _encode_action(self, action, mm, action_taken_mask):
    """Translate the model action into game env action.

    TODO: to encode SAP action, prev  observation is required.
    """
    action_taken_mask = action_taken_mask[UNITS_ACTION]

    def encode(i, a):
      # Skip random action generated by model.
      if not action_taken_mask[i][a]:
        return 0, 0, 0

      if a < MOVE_ACTION_NUM:
        x, y = 0, 0
      else:
        a -= MOVE_ACTION_NUM
        x, y = self.sap_indexer.idx_to_position[a]
        a = ACTION_SAP
        # print(f'sap: dx={x}, dy={y}')

      # Note: transpose shoud happen before mirror
      if mm.transpose:
        a = TRANSPOSED_ACTION[a]
        x, y = y, x

      if mm.use_mirror:
        a = MIRRORED_ACTION[a]
        x, y = -y, -x

      return a, x, y

    action = action[UNITS_ACTION]
    unit_actions = np.zeros((MAX_UNIT_NUM, 3), dtype=np.int32)
    for i in range(MAX_UNIT_NUM):
      # uid = mm.unit_idx_to_id[i]
      uid = i

      a = int(action[i][0])  # unbox [a] => a
      a, x, y = encode(i, a)

      # Set the unit action based on its real unit info.
      unit_actions[uid][:] = (a, x, y)
    return unit_actions

  def compute_actions_taken(self, model_actions):
    return [
        self.get_actions_taken_mask(model_actions[i], self.mms[i])
        for i, player in enumerate([PLAYER0, PLAYER1])
    ]

  def is_game_done(self, raw_obs, player):
    team_wins = raw_obs[player]['team_wins']
    game_step = raw_obs[player]['steps']
    # return (game_step >= MAX_GAME_STEPS) or (max(team_wins) >= MIN_TEAM_WINS)
    return (game_step >= MAX_GAME_STEPS)

  def step(self, model_action):
    if self.use_single_player:
      model_action = [
          model_action,
          {
              UNITS_ACTION:
              np.zeros((MAX_UNIT_NUM, 1),
                       np.int32)  # use dummy action for the other player
          }
      ]

    self._actions_taken_mask = self.compute_actions_taken(model_action)
    action = {
        PLAYER0:
        self._encode_action(model_action[0], self.mms[0],
                            self._actions_taken_mask[0]),
        PLAYER1:
        self._encode_action(model_action[1], self.mms[1],
                            self._actions_taken_mask[1]),
    }
    # Use agent to generate action to overwrite the model action.
    if self.use_agent:
      action = {
          PLAYER0: self.agent_actions[0],
          PLAYER1: self.agent_actions[1],
      }
      model_action = [
          {
              UNITS_ACTION:
              self.agent_action_do_model_action(self.agent_actions[0],
                                                self.mms[0])
          },
          {
              UNITS_ACTION:
              self.agent_action_do_model_action(self.agent_actions[1],
                                                self.mms[1])
          },
      ]
      self._actions_taken_mask = self.compute_actions_taken(model_action)

    raw_obs, step_reward, terminated, truncated, info = self.game.step(action)
    final_state = info['final_state']

    print(
        f"step={raw_obs[PLAYER0]['steps']} final_state.energy_nodes={final_state.energy_nodes}, final_state.energy_nodes_mask={final_state.energy_nodes_mask}"
    )
    self._update_mms(raw_obs,
                     model_actions=model_action,
                     env_state=final_state)

    done = self.is_game_done(raw_obs, PLAYER0)

    obs = self.observation(raw_obs, final_state)
    reward = self._convert_reward(raw_obs, final_state)
    info = self.get_info(model_action, raw_obs, reward, done, final_state)

    self.prev_raw_obs = raw_obs
    self.raw_obs = raw_obs
    return obs, reward, done, info

  def agent_action_do_model_action(self, agent_action, mm):
    units_action = np.zeros((MAX_UNIT_NUM, 1), dtype=int)
    for i in range(MAX_UNIT_NUM):
      a = agent_action[i][0]
      if a < MOVE_ACTION_NUM:
        units_action[i] = a
      else:
        dx = agent_action[i][1]
        dy = agent_action[i][2]
        sap_id = mm.sap_indexer.position_to_idx[(dx, dy)]
        sap_id += MOVE_ACTION_NUM
        units_action[i] = sap_id
    return units_action

  def _convert_observation(self, ob, mm, final_state=None, skip_check=False):
    """Construct all features using MAP_SHAPE2."""
    o = {}

    def scalar(v, maxv):
      return np.zeros(MAP_SHAPE2) + (v / maxv)

    def nebula_params(v):
      param = np.zeros(MAP_SHAPE2, dtype=np.int32)
      param[mm.cell_type == CELL_NEBULA] = int(v)
      return param

    def game_params(v):
      param = np.zeros(MAP_SHAPE2, dtype=np.int32)
      return param + int(v)

    def get_units_total_energy(env_state, pid):
      return env_state.units.energy[pid].sum()

    def extract_baseline_extras(mm, env_state):
      extras = np.zeros(N_BASELINE_EXTRA_DIM)
      if env_state is None:
        return extras

      hidden_relics_num = count_relic_score_nodes_num(env_state)
      extras[0] = hidden_relics_num / (6 * 25 * 0.25)
      if hidden_relics_num == 0:
        hidden_relics_num = 1
      team_points = env_state.team_points[mm.player_id]
      enemy_points = env_state.team_points[mm.enemy_id]
      extras[1] = team_points / 100 / hidden_relics_num

      team_energy = get_units_total_energy(env_state, mm.player_id)
      enemy_energy = get_units_total_energy(env_state, mm.enemy_id)
      extras[2] = team_energy / 4000
      extras[3] = np.clip((team_energy - enemy_energy) / 1000, -1, 1)

      relic_num = env_state.relic_nodes_mask.sum()
      extras[4] = relic_num / MAX_RELIC_NODE_NUM

      params = mm.full_params
      extras[10] = params['unit_sensor_range'] / MAX_SENSOR_RANGE
      extras[11] = params['unit_move_cost'] / MAX_MOVE_COST
      extras[12] = params['unit_sap_cost'] / MAX_SAP_COST
      extras[13] = params['unit_sap_range'] / MAX_SAP_RANGE
      extras[
          14] = params['nebula_tile_vision_reduction'] / MAX_VISION_REDUCTION
      extras[
          15] = params['nebula_tile_energy_reduction'] / MAX_ENERGY_REDUCTION
      extras[16] = params['unit_sap_dropoff_factor']
      extras[17] = params['unit_energy_void_factor']
      extras[18] = params['nebula_tile_drift_speed']
      extras[19] = params['energy_node_drift_speed']
      extras[20] = params['energy_node_drift_magnitude'] / 6

      mm2 = self.mms[mm.enemy_id]
      extras[21] = (mm.is_relic_node.sum() -
                    mm2.is_relic_node.sum()) / MAX_RELIC_NODE_NUM
      extras[22] = (mm.get_game_visited_relic_nb_num() -
                    mm2.get_game_visited_relic_nb_num()) / (3 * 25)
      extras[23] = ((mm.game_observed_num - mm2.game_observed_num) /
                    (24 * 24 / 2))

      return extras

    # Time & Match
    o['game_step'] = scalar(mm.game_step, MAX_GAME_STEPS)
    o['match_step'] = scalar(mm.match_step, MAX_MATCH_STEPS)

    # Game params
    o['unit_move_cost'] = game_params(mm.unit_move_cost)
    o['unit_sensor_range'] = game_params(mm.unit_sensor_range)

    o['unit_sap_cost'] = game_params(mm.unit_sap_cost)
    o['unit_sap_range'] = game_params(mm.unit_sap_range)

    o['nebula_vision_reduction'] = nebula_params(mm.nebula_vision_reduction)
    o['nebula_energy_reduction'] = nebula_params(
        mm._nebula_energy_reduction.index())

    o['unit_sap_dropoff_factor'] = game_params(
        mm.sap_dropoff_factor_estimator.index())

    # Scores
    o['player_id'] = scalar(mm.player_id, 1)

    team_points = ob['team_points']
    units_points = team_points[mm.player_id]
    enemy_points = team_points[mm.enemy_id]
    o['units_team_points'] = scalar(units_points, MAX_TEAM_POINTS)
    o['team_points_delta'] = scalar(units_points - enemy_points,
                                    MAX_TEAM_POINTS)
    prev_team_points = self.prev_raw_obs[mm.player]['team_points']
    units_team_points_growth = max(
        units_points - prev_team_points[mm.player_id], 0)
    o['units_team_points_growth'] = scalar(units_team_points_growth,
                                           MAX_UNIT_NUM)
    enemy_team_points_growth = max(
        enemy_points - prev_team_points[mm.enemy_id], 0)
    team_points_growth_delta = units_team_points_growth - enemy_team_points_growth
    o['team_points_growth_delta'] = scalar(team_points_growth_delta,
                                           MAX_UNIT_NUM)
    team_wins = ob['team_wins']
    units_wins = team_wins[mm.player_id]
    enemy_wins = team_wins[mm.enemy_id]
    o['units_wins'] = scalar(units_wins, TEAM_WIN_NORM)
    o['match_wins_delta'] = scalar(units_wins - enemy_wins, TEAM_WIN_NORM)

    # Map info
    o['_a_cell_type'] = mm.cell_type.copy()
    o['_b_cell_type'] = mm.true_cell_type.copy()

    o['visible'] = mm.visible.astype(np.float32)
    ob_age = np.minimum((mm.game_step - mm.last_observed_step),
                        MAX_MATCH_STEPS)
    o['last_observed_age'] = ob_age / MAX_MATCH_STEPS
    visit_age = np.minimum((mm.game_step - mm.last_observed_step),
                           MAX_MATCH_STEPS)
    o['last_visited_age'] = visit_age / MAX_MATCH_STEPS

    o['_a_is_relic_node'] = mm.is_relic_node.astype(np.float32)
    o['_b_is_relic_node'] = mm.true_relic_map.astype(np.float32)

    o['_a_is_relic_neighbour'] = mm.is_relic_neighbour.astype(np.float32)
    true_relic_neighbour = maximum_filter((mm.true_relic_map == 1),
                                          size=RELIC_NB_SIZE)
    o['_b_is_relic_neighbour'] = true_relic_neighbour.astype(np.float32)

    # places need unit stay
    o['_a_team_point_prob'] = mm.team_point_mass.astype(np.float32)
    o['_b_team_point_prob'] = mm.true_team_point_map.astype(np.float32)

    if self.obs_space_kwargs.get('use_energy_cost_map'):
      o['_a_energy_cost_map'] = mm.get_erengy_cost_map_feature(
          mm.energy_cost_map)

      energy_reduction = mm.nebula_energy_reduction
      if mm.full_params:
        energy_reduction = mm.full_params['nebula_tile_energy_reduction']
      true_cost_map = mm.compute_energy_cost_map(mm.true_cell_type,
                                                 mm.true_cell_energy,
                                                 mm.true_relic_map,
                                                 energy_reduction)
      o['_b_energy_cost_map'] = mm.get_erengy_cost_map_feature(true_cost_map)

    energy_map = mm.cell_energy.copy()
    energy_map[mm.cell_type == CELL_NEBULA] -= mm.nebula_energy_reduction
    o['_a_cell_energy'] = energy_map / MAX_ENERTY_PER_TILE

    true_energy_map = mm.true_cell_energy.copy()
    true_energy_map[mm.true_cell_type == CELL_NEBULA] -= energy_reduction
    o['_b_cell_energy'] = true_energy_map / MAX_ENERTY_PER_TILE

    # print(
    # f"nebula_energy_reduction={mm.nebula_energy_reduction}, vision_reduction={mm.nebula_vision_reduction}"
    # )

    def add_unit_feature(prefix, player_id, t):
      unit_pos = np.zeros(MAP_SHAPE2)
      unit_energy = np.zeros(MAP_SHAPE2)
      total_energy = 0
      for i in range(MAX_UNIT_NUM):
        mask, pos, energy = mm.get_unit_info(player_id, i, t)
        if mask and energy >= 0:
          unit_energy[pos[0]][pos[1]] += (energy / MAX_UNIT_ENERGY /
                                          MAX_UNIT_NUM)
          unit_pos[pos[0]][pos[1]] += (1 / MAX_UNIT_NUM)
          total_energy += energy

      o[f'{prefix}_loc_t{t}'] = unit_pos
      o[f'{prefix}_energy_t{t}'] = unit_energy
      return total_energy

    def add_unit_info(player_id, t):
      units_info = np.zeros((MAX_UNIT_NUM, 3), dtype=np.int32)
      for i in range(MAX_UNIT_NUM):
        mask, pos, energy = mm.get_unit_info(player_id, i, t)
        is_valid_unit = (mask and energy >= 0)
        units_info[i][0] = pos[0]
        units_info[i][1] = pos[1]
        units_info[i][2] = np.int32(energy) if is_valid_unit else -1

      o[f'_units_info'] = units_info

    # Unit info
    units_total_energy = add_unit_feature('units', mm.player_id, t=0)
    add_unit_feature('units', mm.player_id, t=1)
    add_unit_feature('enemy', mm.enemy_id, t=0)
    add_unit_feature('enemy', mm.enemy_id, t=1)
    add_unit_info(mm.player_id, t=0)

    o['units_total_energy'] = scalar(units_total_energy,
                                     MAX_UNIT_ENERGY * MAX_UNIT_NUM)

    o['_baseline_extras'] = extract_baseline_extras(mm, final_state)

    # if not skip_check:
    # assert len(o) == len(OB), f"len(o)={len(o)}, len(OB)={len(OB)}"

    # expand all feature map with dummy dim 1
    o = {k: np.expand_dims(v, 0) for k, v in o.items()}

    assert 'enemy_loc_t0' in o
    return o

  def observation(self, raw_obs, final_state):
    assert len(
        raw_obs
    ) == 2, f"len(raw_obs)={len(raw_obs)}, self.total_agent_controls={self.total_agent_controls}"
    if self.use_single_player:
      return [
          self._convert_observation(raw_obs[PLAYER0], self.mms[0], final_state)
      ]  # single player
    else:
      return [
          self._convert_observation(raw_obs[PLAYER0], self.mms[0],
                                    final_state),
          self._convert_observation(raw_obs[PLAYER1], self.mms[1], final_state)
      ]

  def _convert_shaping_reward(self, raw_obs, env_state):
    wt = self.reward_shaping_params

    def _convert(mm, ob):
      MIN_WARMUP_MATCH_STEP = 1

      r = 0

      # reward for open unobserved cells
      r_explore = 0
      if mm.match_step > MIN_WARMUP_MATCH_STEP:
        r_explore += (mm.step_observe_corner_cells_num *
                      wt['new_observed_corners'])  # per game
        r_explore += (mm.step_observe_anti_main_diag_area *
                      wt['new_observed_main_diag'])  # per game
        r_explore += (mm.step_observe_anti_diag_down_tri *
                      wt['new_observed_down_tri'])  # per match
        r_explore += (mm.units_position_energy_sum / 100 *
                      wt['units_position_energy_sum'])  # per step

      # reward for visit relic neighbour node s
      r_visit_relic_nb = 0
      r_visit_relic_nb = mm.step_new_visited_relic_nb_num * wt['relic_nb']

      # reward for units sit on hidden relic node.
      r_units_on_relic = (mm.step_units_on_relic_num * wt['units_on_relic'])

      r_team_point = max(
          ob['team_points'][mm.player_id] -
          self.prev_raw_obs[mm.player]['team_points'][mm.player_id], 0)
      r_team_point *= wt['team_point']

      team_wins = raw_obs[mm.player]['team_wins']

      # game end reward
      r_game = 0
      if self.is_game_done(raw_obs, mm.player):
        if team_wins[mm.player_id] > team_wins[mm.enemy_id]:
          r_game = wt['game_result']
        elif team_wins[mm.player_id] < team_wins[mm.enemy_id]:
          r_game = -wt['game_result']

      # match end reward
      r_match = 0
      prev_team_wins = self.prev_raw_obs[mm.player]['team_wins']
      diff = team_wins - prev_team_wins
      if diff[mm.player_id] > 0:
        r_match = wt['match_result']
      elif diff[mm.enemy_id] > 0:
        r_match = -wt['match_result']

      r_dead = 0
      r_dead += mm.units_dead_count * wt['dead_uints']
      r_frozen = 0
      r_frozen += mm.step_units_frozen_count * wt['first_frozen_uints']
      r_frozen += mm.units_frozen_count * wt['stay_frozen_units']

      r_kill = self.mms[mm.enemy_id].units_dead_count * wt['kill_units']

      r = (r_explore + +r_visit_relic_nb + r_game + r_match + r_team_point +
           r_units_on_relic + r_dead + r_frozen + r_kill)

      # if r_kill != 0:
      # print(
      # f'step={mm.game_step} match-step={mm.match_step}, r={r:.5f} explore={r_explore:.4f} '
      # f' r_visit_relic_nb={r_visit_relic_nb}, r_team_point={r_team_point}, r_dead={r_dead}'
      # f' r_frozen={r_frozen}, r_kill={r_kill}')
      return r

    return [
        _convert(self.mms[i], raw_obs[p])
        for i, p in enumerate([PLAYER0, PLAYER1])
    ]

  def _convert_match_win_loss(self, raw_obs, env_state):
    wt = self.reward_shaping_params

    def _convert(mm, ob):
      team_wins = raw_obs[mm.player]['team_wins']

      # match end reward
      r_match = 0
      prev_team_wins = self.prev_raw_obs[mm.player]['team_wins']
      diff = team_wins - prev_team_wins
      if diff[mm.player_id] > 0:
        r_match = wt['match_result']
      elif diff[mm.enemy_id] > 0:
        r_match = -wt['match_result']
      return r_match

    return [
        _convert(self.mms[i], raw_obs[p])
        for i, p in enumerate([PLAYER0, PLAYER1])
    ]

  def _convert_match_explore_win_loss(self, raw_obs, env_state):
    wt = self.reward_shaping_params

    def _convert(mm, ob):
      team_wins = raw_obs[mm.player]['team_wins']
      mm2 = self.mms[mm.enemy_id]

      r_match = 0
      r_match_observed = 0

      r = 0
      if mm.match_step == MAX_MATCH_STEPS:
        team_points = raw_obs[mm.player]['team_points'][mm.player_id]
        enemy_points = raw_obs[mm.player]['team_points'][mm.enemy_id]
        net_win_points = team_points - enemy_points
        # r_match = max(min(net_win_points / 100, 1), -1)
        if team_points > enemy_points:
          r_match = wt['match_win']
        elif team_points < enemy_points:
          r_match = -wt['match_win']

        r = r_match
      elif mm.match_step == HALF_MATCH_STEPS and mm.game_step <= 300:
        team_observed_num = mm.match_observed_num
        enemy_observed_num = mm2.match_observed_num
        # net_win_ob = team_observed_num - enemy_observed_num
        # r_match_observed = max(min(net_win_ob / 100, 1), -1)
        if team_observed_num > enemy_observed_num:
          r_match_observed = wt['match_observed']
        elif team_observed_num < enemy_observed_num:
          r_match_observed = -wt['match_observed']

        r = r_match_observed

      # r += (mm.units_energy_cost_change * wt['energy_cost_change'])
      # print(
      # f'step={mm.game_step} match-step={mm.match_step}, r_match={r_match} r_match_observed={r_match_observed}'
      # )
      return r

    return [
        _convert(self.mms[i], raw_obs[p])
        for i, p in enumerate([PLAYER0, PLAYER1])
    ]

  def _convert_shaping_reward_v2(self, raw_obs, env_state):
    wt = self.reward_shaping_params

    def _convert(mm, ob):
      r_match = 0
      r = 0
      if mm.match_step == MAX_MATCH_STEPS:
        relic_num = (count_relic_score_nodes_num(env_state) / 2 + 0.1)

        team_points = raw_obs[mm.player]['team_points'][mm.player_id]
        r_match = team_points / MAX_MATCH_STEPS / relic_num * 5
        r = r_match

        # print(
        # f'step={mm.game_step} match-step={mm.match_step}, r_match={r_match}'
        # )
      return r

    return [
        _convert(self.mms[i], raw_obs[p])
        for i, p in enumerate([PLAYER0, PLAYER1])
    ]

  def _convert_win_loss_reward2(self, raw_obs, env_state):

    def _convert(mm, ob):
      if mm.match_step == MAX_MATCH_STEPS:
        team_points = raw_obs[mm.player]['team_points'][mm.player_id]
        enemy_points = raw_obs[mm.player]['team_points'][mm.enemy_id]
        if team_points > enemy_points:
          mm.match_wins += 1
        elif team_points < enemy_points:
          mm.match_wins -= 1
        # print(f'team_points={team_points} enemy_points={enemy_points}')

      # game end reward
      r_game = 0
      if self.is_game_done(raw_obs, mm.player):
        if mm.match_wins > 0:
          r_game = 1
        elif mm.match_wins < 0:
          r_game = -1

      # print(
      # f'step={mm.game_step} match-step={mm.match_step}, r_game={r_game}, match_wins={mm.match_wins}'
      # )
      return r_game

    return [
        _convert(self.mms[i], raw_obs[p])
        for i, p in enumerate([PLAYER0, PLAYER1])
    ]

  def _convert_reward(self, raw_obs, env_state):
    """Use the match win-loss reward for now."""
    assert self.reward_schema in ('shaping', 'game_win_loss2',
                                  'match_win_loss', 'match_explore_win_loss',
                                  'shaping_v2')
    if self.reward_schema == 'game_win_loss2':
      reward = self._convert_win_loss_reward2(raw_obs, env_state)

    if self.reward_schema == 'shaping':
      reward = self._convert_shaping_reward(raw_obs, env_state)

    if self.reward_schema == 'match_win_loss':
      reward = self._convert_match_win_loss(raw_obs, env_state)

    if self.reward_schema == 'match_explore_win_loss':
      reward = self._convert_match_explore_win_loss(raw_obs, env_state)

    if self.reward_schema == 'shaping_v2':
      reward = self._convert_shaping_reward_v2(raw_obs, env_state)

    if self.use_single_player:
      return [reward[0]]  # single player
    else:
      return reward

  def _get_available_action_mask(self, mm):
    """Mask for unit action: compute available action based on unit position"""
    actions_mask = np.zeros(EXT_ACTION_SHAPE, dtype=bool)

    units = []
    action_centered_positions = set()
    for i in range(MAX_UNIT_NUM):
      unit_mask, pos, energy = mm.get_unit_info(mm.player_id, i, t=0)

      # If units or is not exists, it can't do anything.
      if not unit_mask:
        continue

      # Empty energy unit must stay
      pos = (int(pos[0]), int(pos[1]))
      if energy == 0:
        action_centered_positions.add(pos)

      if energy >= 0:
        # Always can stay
        actions_mask[i][ACTION_CENTER] = 1

      # Unit runs out of energy
      if energy < mm.unit_move_cost:
        continue

      units.append((energy, i, pos))

    def update_move_action_mask(i, pos, energy):
      # has enough energy to move
      for k in range(1, MAX_MOVE_ACTION_IDX + 1):
        nx, ny = (pos[0] + DIRECTIONS[k][0], pos[1] + DIRECTIONS[k][1])
        if not is_pos_on_map((nx, ny)):
          continue
        if mm.cell_type[nx][ny] == CELL_ASTERIOD:
          continue
        actions_mask[i][k] = 1

    sap_hit_map = mm.get_global_sap_hit_map()

    def update_sap_action_mask(i, pos, energy):
      if energy < mm.unit_sap_cost:
        return

      # TODO: how to coodinate sap to not sap on same cell, maybe coodinate it post-execution?
      unit_sap_mask = gen_sap_range(pos, mm.unit_sap_range)
      unit_sap_mask = (unit_sap_mask & sap_hit_map)

      sap_dxdy = np.argwhere(unit_sap_mask) - np.array(pos, dtype=np.int32)
      for dx, dy in sap_dxdy:
        sap_id = mm.sap_indexer.position_to_idx[(dx, dy)]
        sap_id += MOVE_ACTION_NUM
        actions_mask[i][sap_id] = 1

    # sort units by energy
    units.sort()
    action_centered_positions = set()
    for energy, i, pos in units:
      update_move_action_mask(i, pos, energy)
      update_sap_action_mask(i, pos, energy)

    return {UNITS_ACTION: actions_mask}

  def get_actions_taken_mask(self, model_action, mm):
    """Should ignore all the actions that can not be performed. Compute this
    before env.step() to make use of mm from prev step."""
    mask = np.zeros(EXT_ACTION_SHAPE, dtype=bool)

    available_action_mask = self._get_available_action_mask(mm)[UNITS_ACTION]
    units_action = model_action[UNITS_ACTION]
    for i, a in enumerate(units_action):
      if np.any(available_action_mask[i]):
        mask[i][a] = 1
    return {UNITS_ACTION: mask}

  def get_info(self,
               model_action,
               raw_obs,
               reward,
               done=False,
               env_state=None):

    def count_actions(info, action, taken_masks, mm):
      action_count = {a: 0 for a in ACTION_ID_TO_NAME.values()}
      action = action[UNITS_ACTION]
      for i in range(MAX_UNIT_NUM):
        a = action[i][0]
        if taken_masks[i][a] > 0:
          aid = a
          if aid >= MOVE_ACTION_NUM:
            aid = ACTION_SAP
          name = ACTION_ID_TO_NAME[aid]
          action_count[name] += 1

      # if mm.player_id == 1:
      # print(
      # f"step={mm.game_step}, player-{mm.player_id}, actions={action_count}"
      # )
      # for i in range(MAX_UNIT_NUM):
      # mask, pos, energy = mm.get_unit_info(mm.player_id, i, t=0)
      # mask_, pos_, energy_ = mm.get_unit_info(mm.player_id, i, t=1)
      # if mask and mask_:
      # a = action[i][0]
      # if aid >= MOVE_ACTION_NUM:
      # continue
      # print(
      # f"unit[{i}], mask={mask}, pos={pos}/pos_={pos_}, energy={energy}/energy_={energy_}, aid={aid}, a={ACTION_ID_TO_NAME[aid]}, d={DIRECTIONS[aid]}"
      # )

      # append '_' for each action name
      info.update([('_' + a.lower(), c) for a, c in action_count.items()])

    def add_unit_total_energy(info, mm):
      total_energy = 0
      for i in range(MAX_UNIT_NUM):
        mask, _, energy = mm.get_unit_info(mm.player_id, i, t=0)
        if mask:
          total_energy += energy

      info['_unit_total_energy'] = total_energy

    def _get_info(agent_action, raw_obs1, prev_obs1, agent_reward, mm,
                  env_state):
      info = {}

      info['actions_taken_mask'] = self._actions_taken_mask[mm.player_id]
      if self.use_agent:
        # match with actions mask
        units_action = self.agent_action_do_model_action(
            self.agent_actions[mm.player_id], mm)
        info['agent_action'] = units_action

      # action mask for current state, (for sample action)
      info['available_action_mask'] = self._get_available_action_mask(mm)

      step_reward = reward[mm.player_id]
      info['player'] = (int(self._seed) %
                        1000) * 100 + mm.player_id  # for testing
      info['_step_reward'] = step_reward
      info['_step_new_found_relic_node_num'] = mm.step_new_found_relic_node_num
      info['_step_new_visited_relic_nb_num'] = mm.step_new_visited_relic_nb_num
      info['_step_actionable_unit_num'] = self._actions_taken_mask[
          mm.player_id][UNITS_ACTION].any(axis=-1, keepdims=True).sum()

      # Team points stats
      tp0 = raw_obs1['team_points'][mm.player_id]
      tp1 = prev_obs1['team_points'][mm.enemy_id]

      team_win = raw_obs1['team_wins'][mm.player_id]
      enemy_win = raw_obs1['team_wins'][mm.enemy_id]
      # print(
      # f"step={raw_obs[PLAYER0]['steps']}, match_steps={mm.match_step} done={done}, player_id={mm.player_id} team_point={tp0}"
      # )
      info['_step_team_points'] = max(tp0 - tp1, 0)

      info['_match_observed_node_num'] = 0
      info['_match_visited_node_num'] = 0

      info['_match_team_points'] = 0
      info['_winner_match_team_points'] = 0
      info['_match_played'] = 0
      info['_winner'] = 0
      info['_match_dead_units'] = 0
      info['_match_frozen_units'] = 0
      match_step = raw_obs[PLAYER0]['match_steps']

      # if mm.prev_units_on_relic_num != mm.units_on_relic_num:
      # print(
      # f"step={raw_obs[PLAYER0]['steps']}, match_steps={mm.match_step} done={done}, player_id={mm.player_id} team_point={tp0}, "
      # f" units_on_relic_num={mm.units_on_relic_num}, pret_on_relic_num={mm.prev_units_on_relic_num}"
      # )

      if match_step == MAX_MATCH_STEPS:
        info['_match_observed_node_num'] = mm.match_observed.sum()
        info['_match_visited_node_num'] = mm.match_visited.sum()

        mm.total_team_points += tp0
        info['_match_team_points'] = tp0
        info['_winner_match_team_points'] = max(tp0, tp1)

        info['_match_played'] = 1
        if team_win > enemy_win:
          info['_winner'] = mm.player_id
        else:
          info['_winner'] = mm.enemy_id

        info['_match_dead_units'] = mm.total_units_dead_count
        info['_match_frozen_units'] = mm.total_units_frozen_count

      info['_game_total_match_points'] = 0
      info['_game_observed_node_num'] = 0
      info['_game_visited_node_num'] = 0

      info['_game_total_hidden_relic_nodes_num'] = 0
      info['_game_total_found_relic_nodes_num'] = 0

      info['_game_visited_relic_nb_nodes_num'] = 0
      info['_game_total_relic_nb_nodes_num'] = 0
      if done:
        info['_game_total_match_points'] = mm.total_team_points

        info['_game_visited_relic_nb_nodes_num'] = (
            mm.get_game_visited_relic_nb_num())
        info['_game_total_relic_nb_nodes_num'] = ((mm.is_relic_neighbour
                                                   > 0).sum())

        info['_game_observed_node_num'] = mm.game_observed.sum()
        info['_game_visited_node_num'] = mm.game_visited.sum()

      info['_game_total_hidden_relic_nodes_num'] = count_relic_score_nodes_num(
          env_state)
      info['_game_total_found_relic_nodes_num'] = (mm.team_point_mass
                                                   >= 0.5).sum()

      step = raw_obs[PLAYER0]['steps']
      # print(f"step={step} match_step={match_step}, step_reward={step_reward}")
      count_actions(info, agent_action,
                    self._actions_taken_mask[mm.player_id][UNITS_ACTION], mm)
      add_unit_total_energy(info, mm)

      if self.use_agent:
        agent_action = self.agents[mm.player_id].act(mm.game_step,
                                                     raw_obs[mm.player])
        self.agent_actions[mm.player_id] = agent_action
      return info

    if model_action is None:
      model_action = [None, None]

    if self.use_single_player:
      return [
          _get_info(model_action[0], raw_obs[PLAYER0],
                    self.prev_raw_obs[PLAYER0], reward[0], self.mms[0],
                    env_state)
      ]  # single player
    else:
      return [
          _get_info(model_action[i], raw_obs[player],
                    self.prev_raw_obs[player], reward[i], self.mms[i],
                    env_state) for i, player in enumerate([PLAYER0, PLAYER1])
      ]

from collections import OrderedDict, deque, defaultdict, Counter
from functools import cached_property
import random
import sys

import gym

gym.logger.set_level(40)

import numpy as np
from gym import spaces
from luxai_s3.wrappers import LuxAIS3GymEnv
from scipy.ndimage import maximum_filter

from random import randint

from .const import *

EXT_ACTION_SHAPE = (MAX_UNIT_NUM, ALL_ACTION_NUM)

# Let's use move action only first
ACTION_SPACE = spaces.Dict({
    UNITS_ACTION:
    spaces.MultiDiscrete(np.zeros(MAX_UNIT_NUM, dtype=int) + ALL_ACTION_NUM)
})

MAP_SHAPE = (1, MAP_WIDTH, MAP_HEIGHT)
MAP_SHAPE2 = (MAP_WIDTH, MAP_HEIGHT)

OB = OrderedDict([
    # Game params
    ('unit_move_cost', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('unit_sensor_range', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('unit_sap_cost', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('unit_sap_range', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('nebula_energy_reduction', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),

    # Time & Match
    ('game_step', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('match_step', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_team_points', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_team_points_delta', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_wins', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('enemy_team_points', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('enemy_team_points_delta', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('enemy_wins', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),

    # Map info
    ('cell_type', spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + N_CELL_TYPES)),
    ('nebula_tile_vision_reduction', spaces.Box(low=0, high=1,
                                                shape=MAP_SHAPE)),
    ('vision_map', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('visible', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    # ('game_observed', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('match_observed', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('game_visited', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('match_visited', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('is_relic_node', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),

    # use this to indicate nodes of unvisited relic cells (and its neighbour)
    ('game_unvisited_relic_nb', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    # use this to indicate the hidden place of relc nodes.
    ('team_point_prob', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    #
    ('cell_energy', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),

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

OBSERVATION_SPACE = spaces.Dict(OB)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def game_win_loss(scores, player_id):
  if scores[0] == scores[1]:
    return 0
  return np.argmax(scores) == player_id and 1 or -1


def manhatten_distance(p1, p2):
  return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def generate_manhattan_mask(mat, center, range_limit):
  rows, cols = mat.shape
  x_center, y_center = center
  x, y = np.ogrid[:rows, :cols]
  manhattan_distance = np.abs(x - x_center) + np.abs(y - y_center)
  mask = manhattan_distance <= range_limit
  return mask


def anti_diag_sym(A):
  A = np.flipud(A)
  A = np.fliplr(A)
  A = A.T
  return A


def anti_diag_sym_i(v):
  i, j = v
  if i == -1 or j == -1:
    return -1, -1
  return MAP_WIDTH - 1 - j, MAP_HEIGHT - 1 - i


def unit_move(p, a):
  d = DIRECTIONS[a]
  return (p[0] + d[0], p[1] + d[1])


def is_pos_on_map(tmp):
  return 0 <= tmp[0] < MAP_WIDTH and 0 <= tmp[1] < MAP_HEIGHT


class NebulaEnergyReduction:

  VALID_VALUES = (0, 10, 25)

  def __init__(self):
    self.counter = Counter()

  def is_valid(self, val):
    return val in self.VALID_VALUES

  def add(self, v):
    if self.is_valid(v):
      self.counter[v] += 1

  def best_guess(self):
    if len(self.counter) <= 0:
      return 0
    return self.counter.most_common(1)[0][0]


class VisionMap:

  def __init__(self, unit_sensor_range):
    self.vision = None
    self.unit_sensor_range = unit_sensor_range

  def _add_vision(self, p, unit_sensor_range, wt=1):
    i, j = p
    v = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=np.int32)
    for r in range(self.unit_sensor_range + 1):
      d = (self.unit_sensor_range - r)
      x0 = max(0, (i - d))
      x1 = min(MAP_WIDTH, (i + d + 1))
      y0 = max(0, (j - d))
      y1 = min(MAP_HEIGHT, (j + d + 1))
      v[x0:x1, y0:y1] = (r + 1) * wt
    return v

  def update(self, mask, position, energy):
    self.vision = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=np.int32)

    unit_positions = defaultdict(int)
    for i in range(MAX_UNIT_NUM):
      m, p, e = mask[i], position[i], energy[i]
      if m and e >= 0:
        unit_positions[(p[0], p[1])] += 1

    for p, v in unit_positions.items():
      self.vision += self._add_vision(p, self.unit_sensor_range, v)


def seed_to_transpose(s):
  return bool(s & 1), bool((s // 2) & 1)


class SapIndexer:

  def __init__(self):
    self.position_to_idx = {}
    self.idx_to_position = {}

    mat = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=np.bool)
    center = (12, 12)
    self.mask = generate_manhattan_mask(mat, center, MAX_SAP_RANGE)

    delta_positions = np.argwhere(self.mask > 0) - np.array(center,
                                                            dtype=np.int32)
    for i, (x, y), in enumerate(delta_positions):
      self.position_to_idx[(x, y)] = i
      self.idx_to_position[i] = (x, y)


class MapManager:

  MAX_PAST_OB_NUM = 3

  def __init__(self, player, env_cfg, transpose=False, sap_indexer=None):
    self.player_id = int(player[-1])
    self.player = player
    self.env_cfg = env_cfg
    self.transpose = transpose
    self.cell_type = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.visible = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.match_observed = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    self.game_observed = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    self.match_visited = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    self.game_visited = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    self.is_relic_node = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.last_relic_node_num = 0
    self.is_relic_neighbour = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.last_game_visited_relic_nb_num = 0

    self.prev_team_point = 0
    self.team_point_mass = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.float32)

    self.cell_energy = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.game_step = 0
    self.match_step = 0

    self.past_obs = deque([])

    # Use idx for model feature encoding and id for action encoding
    # self.unit_idx_to_id = list(range(MAX_UNIT_NUM))
    # random.shuffle(self.unit_idx_to_id)

    self._nebula_energy_reduction = NebulaEnergyReduction()
    self.nebula_vision_reduction = 0
    self.vision_map = VisionMap(self.unit_sensor_range)
    self.sap_indexer = sap_indexer or SapIndexer()

    self.total_team_points = 0

    self.units_on_relic_num = 0
    self.prev_units_on_relic_num = 0

    self.units_frozen_count = 0
    self.prev_units_frozen_count = 0
    self.total_units_frozen_count = 0

    self.units_dead_count = 0
    self.prev_units_dead_count = 0
    self.total_units_dead_count = 0

  @property
  def nebula_energy_reduction(self):
    return self._nebula_energy_reduction.best_guess()

  @property
  def enemy_id(self):
    return 1 - self.player_id

  @property
  def unit_move_cost(self):
    return self.env_cfg['unit_move_cost']

  @property
  def unit_sensor_range(self):
    return self.env_cfg['unit_sensor_range']

  @property
  def unit_sap_cost(self):
    return self.env_cfg['unit_sap_cost']

  @property
  def unit_sap_range(self):
    return self.env_cfg['unit_sap_range']

  def update_cell_type(self, ob):
    # adding 1 to start cell type from 0
    cells = ob['map_features']['tile_type'] + 1

    # Update map cell type
    c = cells > CELL_UNKONWN
    self.cell_type[c] = cells[c]

    # also update the symmetrical positions
    cells_sym = anti_diag_sym(cells)
    ct = cells_sym > CELL_UNKONWN
    self.cell_type[ct] = cells_sym[ct]

  def update_visible_and_observed(self, ob):
    self.visible = ob['sensor_mask'].astype(bool)
    self.prev_match_observed = self.match_observed.copy()
    self.prev_game_observed = self.game_observed.copy()
    self.match_observed |= self.visible
    self.game_observed |= self.visible

  @cached_property
  def anti_main_diag_area(self):
    x = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    for i in range(MAP_WIDTH):
      x[i, MAP_WIDTH - i - 1] = True
    return maximum_filter(x, size=5)

  @property
  def step_observe_anti_main_diag_area(self):
    new_ob_mask = (~self.prev_game_observed) & self.game_observed
    return (new_ob_mask & self.anti_main_diag_area).sum()

  @cached_property
  def anti_diag_down_tri(self):
    mp = np.tri(MAP_WIDTH, MAP_HEIGHT, k=0, dtype=bool)
    mp = np.flip(mp, axis=1)
    return mp

  @property
  def step_observe_anti_diag_down_tri(self):
    new_ob_mask = (~self.prev_match_observed) & self.match_observed
    return (new_ob_mask & self.anti_diag_down_tri).sum()

  @property
  def step_observe_corner_cells_num(self):
    sz = 8
    new_ob_mask = (self.prev_game_observed <= 0) & (self.match_observed > 0)
    return (new_ob_mask[0:sz + 1, (MAP_WIDTH - sz):23 + 1].sum() +
            new_ob_mask[(MAP_WIDTH - sz):23 + 1, 0:sz + 1].sum() +
            new_ob_mask[9:9, 14 + 1:14 + 1].sum())

  def update_counters(self):
    self.last_game_visited_relic_nb_num = self.get_game_visited_relic_nb_num()
    self.last_relic_node_num = self.is_relic_node.sum()

  def mirror(self, ob, kind='M'):
    # print(f'{self.game_step}, kind={kind}')
    position_trans = anti_diag_sym_i
    mat_trans = anti_diag_sym
    if kind == 'T':
      position_trans = lambda x: (x[1], x[0])
      mat_trans = lambda x: x.T

    def update_positions(positions):
      for i, v in enumerate(positions):
        positions[i] = position_trans(v)

    update_positions(ob['units']['position'][0])
    update_positions(ob['units']['position'][1])
    update_positions(ob['relic_nodes'])

    ob['sensor_mask'] = mat_trans(ob['sensor_mask'])

    mf = ob['map_features']
    mf['energy'] = mat_trans(mf['energy'])
    mf['tile_type'] = mat_trans(mf['tile_type'])

  def infer_units_info(self, ob, model_action):
    if (model_action is None
        or len(self.past_obs) == 0) or ob['match_steps'] <= 1:
      return

    # other sensor range will not be blocked by nebula
    # if self.unit_sensor_range > 2:
    # return
    # assert self.unit_sensor_range == 2

    units_action = model_action[UNITS_ACTION]

    def make_action(p, e, action):
      if a == ACTION_CENTER:
        return p, e

      # TODO: check move target
      if a in MOVE_ACTIONS_NO_CENTER:
        if e > self.unit_move_cost:
          tmp = unit_move(p, a)
          if (0 <= tmp[0] < MAP_WIDTH and 0 <= tmp[1] < MAP_HEIGHT
              and self.cell_type[tmp[0], tmp[1]] != CELL_ASTERIOD):
            p = tmp

        e -= self.unit_move_cost
        e = max(e, 0)
        return p, e

      assert a >= ACTION_SAP
      e -= self.unit_sap_cost
      return p, e

    pid = self.player_id
    for i in range(MAX_UNIT_NUM):
      # Get info from last step and apply action
      m0, p0, e0 = self.get_unit_info(self.player_id, i, t=0)

      # Skip since we do not have info for the unit in last step.
      if not m0:
        continue

      # Unit info from current step
      mask = ob['units_mask'][pid][i]
      position = ob['units']['position'][pid][i]
      energy = ob['units']['energy'][pid][i]

      a = units_action[i][0]
      p1, e1 = make_action(p0, e0, a)

      if (self.unit_sensor_range == MIN_UNIT_SENSOR_RANGE and not mask
          and is_pos_on_map(p1)
          and self.cell_type[p1[0]][p1[1]] in (CELL_NEBULA, CELL_UNKONWN)):
        ob['units']['position'][pid][i] = p1
        ob['units']['energy'][pid][i] = e1
        ob['units_mask'][pid][i] = True
        # we can also konw that this cell is nebula here, but better done with vision map

        # print(
        # f'gstep={ob["steps"]}, mstep={ob["match_steps"]} pid={pid}, unit[{i}] p0={p0}, e0={e0} to p1={p1} e1={e1} by a={ACTION_ID_TO_NAME[a]}'
        # )

      if (m0 and mask and e0 > 0 and energy > 0 and e1 > 0
          and self.cell_type[position[0], position[1]] == CELL_NEBULA):
        reduction = e1 - energy
        self._nebula_energy_reduction.add(reduction)
        # print(
        # f'gstep={ob["steps"]}, mstep={ob["match_steps"]}, nebula_energy_reduction={self._nebula_energy_reduction.best_guess()}'
        # )
  @property
  def step_units_on_relic_num(self):
    return self.units_on_relic_num - self.prev_units_on_relic_num

  def update_frozen_or_dead_units(self, env_state):
    self.prev_units_on_relic_num = self.units_on_relic_num
    self.prev_units_frozen_count = self.units_frozen_count
    self.prev_units_dead_count = self.units_dead_count

    if self.match_step <= 1 or len(self.past_obs) < 2:
      return

    if env_state is not None:
      self.units_on_relic_num = self.count_on_relic_nodes_units(env_state)

    self.units_frozen_count = 0
    self.units_dead_count = 0
    self.units_position_energy_sum = 0

    n_units = 0
    for i in range(MAX_UNIT_NUM):
      is_dead, is_frozen = False, False
      mask, p0, e0 = self.get_unit_info(self.player_id, i, t=0)
      mask1, p1, e1 = self.get_unit_info(self.player_id, i, t=1)
      if mask and mask1:
        if e0 < 0 and e1 >= 0:
          is_dead = True

        if (e0 == 0 and e1 >= 0
            and (not self.team_point_mass[p0[0], p0[1]] >= MIN_TP_VAL)):
          is_frozen = True

      if mask and (not self.team_point_mass[p0[0], p0[1]] >= MIN_TP_VAL):
        self.units_position_energy_sum += self.cell_energy[p0[0], p0[1]]
        n_units += 1

      # if is_dead:
      # print(
      # f'gstep={self.game_step}, mstep={self.match_step} pid={self.player_id}, unit[{i}] p0={p0}, e0={e0} to p1={p1} e1={e1} is_dead={is_dead}, is_frozen={is_frozen}'
      # )

      self.units_frozen_count += int(is_frozen)
      self.units_dead_count += int(is_dead)

    self.total_units_dead_count += self.units_dead_count
    self.total_units_frozen_count += self.units_frozen_count

  @property
  def step_units_frozen_count(self):
    return max(self.units_frozen_count - self.prev_units_frozen_count, 0)

    # print(
    # f'gstep={self.game_step}, mstep={self.match_step} pid={self.player_id}, energy_sum={self.units_position_energy_sum}, n_units={n_units}'
    # )

  def update(self, ob, model_action=None, env_state=None):
    # Match restarted and reset some of the unit states
    if ob['match_steps'] == 0:
      self.prev_team_point = 0
      self.past_obs.clear()
      self.total_units_dead_count = 0
      self.total_units_frozen_count = 0
      self.match_visited[:, :] = 0
      self.match_observed[:, :] = 0
      self.prev_units_on_relic_num = self.units_on_relic_num = 0
      self.prev_units_dead_count = self.units_dead_count = 0
      self.prev_units_frozen_count = self.units_frozen_count = 0

    # Mirror should go first before everything else.
    if self.player_id == 1:
      self.mirror(ob)

    if self.transpose:
      self.mirror(ob, 'T')

    # use non-infered units position
    self.vision_map.update(ob['units_mask'][self.player_id],
                           ob['units']['position'][self.player_id],
                           ob['units']['energy'][self.player_id])

    self.infer_units_info(ob, model_action)
    self.game_step = ob['steps']
    self.match_step = ob['match_steps']
    self.update_counters()

    self.update_visible_and_observed(ob)
    self.update_cell_type(ob)

    unit_masks = ob['units_mask'][self.player_id]
    unit_positions = ob['units']['position'][self.player_id][unit_masks]
    self.update_visited_node(unit_positions, ob)

    self.update_relic_node(ob)

    self.update_team_point_mass(ob, unit_positions)
    self.prev_team_point = ob['team_points'][self.player_id]

    self.update_cell_energy(ob)

    self.past_obs.appendleft(ob)

    # append some default observation to make get_unit_info easier
    while len(self.past_obs) < self.MAX_PAST_OB_NUM:
      self.past_obs.appendleft(ob)

    if len(self.past_obs) > self.MAX_PAST_OB_NUM:
      self.past_obs.pop()

    self.update_frozen_or_dead_units(env_state)
    self.update_vision_map()
    self.update_sap_position_by_enemy_position()

    # print(
    # f'step={self.game_step}, step_ob_corner: {self.step_observe_corner_cells_num}'
    # )
  def update_sap_position_by_enemy_position(self):
    """Enemy position for sap action"""
    self.enemy_position_mask = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.get_unit_info(self.enemy_id, i, t=0)
      if mask and energy >= 0:
        self.enemy_position_mask[pos[0], pos[1]] = True

    self.enemy_position_mask = maximum_filter(self.enemy_position_mask, size=3)

  def update_vision_map(self):
    nebula_cell_mask = (self.visible <= 0) & (self.vision_map.vision > 0)

    # Update cell map with guessed nebula
    self.cell_type[nebula_cell_mask] = CELL_NEBULA
    self.cell_type[anti_diag_sym(nebula_cell_mask)] = CELL_NEBULA

    vision_reduction = self.vision_map.vision[nebula_cell_mask]
    if vision_reduction.size > 0:
      v = vision_reduction.max()
      if 0 < v <= MAX_VISION_REDUCTION:
        self.nebula_vision_reduction = max(self.nebula_vision_reduction, v)
      # print(
      # f'gstep={self.game_step}, mstep={self.match_step}, nebula_vision_reduction={self.nebula_vision_reduction}, energy_reduction={self._nebula_energy_reduction.best_guess()}'
      # )

    # self.vision_reduction_cells = nebula_cell_mask

  def get_unit_info(self, pid, i, t):
    # Remap uint-i to another unit.
    # i = self.unit_idx_to_id[i]
    ob = self.past_obs[t]
    mask = ob['units_mask'][pid][i]
    position = ob['units']['position'][pid][i]
    energy = ob['units']['energy'][pid][i]

    # if mask and energy < 0:
    # print(
    # f"gs={self.game_step}, ms={self.match_step}, player={pid}, unit[{i}], mask={mask}, position={position}, energy={energy}"
    # )

    # TODO: use last unit position, maybe not important?
    if not mask:
      position = (0, 0) if self.player_id == 0 else (MAP_WIDTH - 1,
                                                     MAP_HEIGHT - 1)

    return mask, position, energy

  def get_game_visited_relic_nb_num(self):
    return ((self.game_visited) & (self.is_relic_neighbour > 0)).sum()

  @property
  def step_new_visited_relic_nb_num(self):
    return (self.get_game_visited_relic_nb_num() -
            self.last_game_visited_relic_nb_num)

  def update_visited_node(self, unit_positions, ob):
    self.unit_positions = np.zeros((MAP_SHAPE2), dtype=bool)
    self.unit_positions[unit_positions[:, 0], unit_positions[:, 1]] = True
    self.match_visited[self.unit_positions] = 1
    self.game_visited |= self.match_visited

  def count_on_relic_nodes_units(self, env_state):
    return ((env_state.relic_nodes_map_weights > 0) &
            (self.unit_positions)).sum()

  @property
  def step_new_found_relic_node_num(self):
    return self.is_relic_node.sum() - self.last_relic_node_num

  def update_relic_node(self, ob):
    relic_nodes_mask = ob['relic_nodes_mask']
    relic_nodes_positions = ob['relic_nodes'][relic_nodes_mask]
    self.is_relic_node[relic_nodes_positions[:, 0],
                       relic_nodes_positions[:, 1]] = 1
    self.is_relic_node |= anti_diag_sym(self.is_relic_node)
    self.is_relic_neighbour = maximum_filter(
        (self.is_relic_node == 1).astype(np.int32), size=RELIC_NB_SIZE)

  def update_cell_energy(self, ob):
    energy = ob['map_features']['energy']
    is_visible = (self.visible > 0)
    self.cell_energy[is_visible] = energy[is_visible]

    energy_tr = anti_diag_sym(energy)
    is_visible_tr = anti_diag_sym(is_visible)
    self.cell_energy[is_visible_tr] = energy_tr[is_visible_tr]

  def get_team_point(self):
    return ob['team_points'][self.player_id]

  def update_team_point_mass(self, ob, unit_positions):
    """Update team point confidence"""
    team_point = ob['team_points'][self.player_id]
    # print(ob['steps'], ob['match_steps'], 'team_point=', team_point, 'prev_point', self.prev_team_point)
    if ob['match_steps'] == 0 or len(unit_positions) == 0:
      return

    unit_pos_mask = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    unit_pos_mask[unit_positions[:, 0], unit_positions[:, 1]] = True
    unit_nearby_relic = (self.is_relic_neighbour > 0) & (unit_pos_mask)
    if unit_nearby_relic.sum() == 0:
      return

    delta = team_point - self.prev_team_point
    assert delta >= 0

    # because team points is non-decreasing, a delta of 0 means all unit positions
    # nearby relic are not team point positions.
    if delta == 0:
      self.team_point_mass[unit_nearby_relic] = NON_TEAM_POINT_MASS
      self.team_point_mass[anti_diag_sym(
          unit_nearby_relic)] = NON_TEAM_POINT_MASS
      return

    change = 50
    # when delta > 0
    must_be_team_point = (self.team_point_mass
                          >= TEAM_POINT_MASS) & (unit_nearby_relic)

    # exclude the cell that must be team points, whatever remains is new team points position
    delta -= must_be_team_point.sum()

    # Means something wrong with the curr team point mass
    if delta < 0:
      self.team_point_mass[must_be_team_point] -= change
      self.team_point_mass[anti_diag_sym(must_be_team_point)] -= change
      return

    assert delta >= 0
    non_team_point = (self.team_point_mass
                      <= NON_TEAM_POINT_MASS) & (unit_nearby_relic)
    team_point_candidate = unit_nearby_relic & (~must_be_team_point) & (
        ~non_team_point)
    num = team_point_candidate.sum()
    if num == 0:
      return

    if delta == 0:
      # No new team points
      self.team_point_mass[team_point_candidate] -= change
      self.team_point_mass[anti_diag_sym(team_point_candidate)] -= change
    elif delta >= num:
      # Every candidate position is a team point position
      self.team_point_mass[team_point_candidate] += change
      self.team_point_mass[anti_diag_sym(team_point_candidate)] += change
    else:
      # num < delta, some of the point is team point
      assert delta < num
      # print('>>>>>>>>>>>>>>', ob['steps'], delta, num, must_be_team_point.sum(), non_team_point.sum())
      self.team_point_mass[team_point_candidate] += (delta / num)
      self.team_point_mass[anti_diag_sym(team_point_candidate)] += (delta /
                                                                    num)

  def get_relic_nb_nodes_to_visit(self):
    to_visit = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    to_visit[(~self.game_visited) & (self.is_relic_neighbour > 0)] = 1
    return to_visit

  def get_must_be_relic_nodes(self):
    relic_nodes = -np.ones((MAP_WIDTH, MAP_HEIGHT), np.int32)
    relic_nodes[(self.team_point_mass >= MIN_TP_VAL)] = 1
    return relic_nodes

  def get_sap_mask(self, pos, sap_range_limit):
    mask = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    mask = generate_manhattan_mask(mask, pos, sap_range_limit)
    return mask & self.enemy_position_mask


class LuxS3Env(gym.Env):

  def __init__(self,
               reward_schema=None,
               game_env=None,
               reward_shaping_params=None):
    self.reward_schema = reward_schema
    self.game = game_env or LuxAIS3GymEnv(numpy_output=True)
    self.reward_shaping_params = reward_shaping_params
    self.mms = None
    self.prev_raw_obs = None
    self._seed = None

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
    self._seed = seed

  def _update_mms(self, obs, model_actions, env_state=None):
    a0, a1 = None, None
    if model_actions is not None:
      a0, a1 = model_actions
    self.mms[0].update(obs[PLAYER0], a0, env_state)
    self.mms[1].update(obs[PLAYER1], a1, env_state)

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
    # t1, t2 = seed_to_transpose(self._seed)

    sap_indexer = SapIndexer()

    t1 = (self._seed % 2 == 0)
    t2 = bool(1 - int(t1))
    # t1, t2 = True, True
    t1, t2 = False, False
    self.mms = [
        MapManager(PLAYER0, env_cfg, transpose=t1, sap_indexer=sap_indexer),
        MapManager(PLAYER1, env_cfg, transpose=t2, sap_indexer=sap_indexer)
    ]
    self._update_mms(raw_obs, model_actions=None, env_state=final_state)
    self.sap_indexer = sap_indexer

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

    return self.observation(raw_obs, final_state), reward, done, info

  def _encode_action(self, action, mm, action_taken_mask):
    """Translate the model action into game env action.

    TODO: to encode SAP action, prev observation is required.
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

      if mm.player_id == 1:
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
    if SINGLE_PLAER:
      model_action = [
          model_action,
          {
              UNITS_ACTION: np.zeros(
                  (MAX_UNIT_NUM, 1), np.int32
              )  # use dummy action for the other player is not optimal
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
    raw_obs, step_reward, terminated, truncated, info = self.game.step(action)
    final_state = info['final_state']
    self._update_mms(raw_obs,
                     model_actions=model_action,
                     env_state=final_state)

    done = self.is_game_done(raw_obs, PLAYER0)

    obs = self.observation(raw_obs, final_state)
    reward = self._convert_reward(raw_obs, final_state)
    info = self.get_info(model_action, raw_obs, reward, done, final_state)

    self.prev_raw_obs = raw_obs
    return obs, reward, done, info

  def _convert_observation(self, ob, mm, final_state=None, skip_check=False):
    """Construct all features using MAP_SHAPE2."""
    o = {}

    def scalar(v, maxv):
      return np.zeros(MAP_SHAPE2) + (v / maxv)

    def get_units_total_energy(env_state, pid):
      return env_state.units.energy[pid].sum()

    def extract_baseline_extras(mm, env_state):
      extras = np.zeros(N_BASELINE_EXTRA_DIM)
      if env_state is None:
        return extras
      extras[0] = mm.game_step / MAX_GAME_STEPS  # step
      extras[1] = (mm.game_step //
                   (MAX_MATCH_STEPS + 1)) / TEAM_WIN_NORM  # match
      extras[2] = mm.match_step / MAX_MATCH_STEPS  # match step

      team_win = env_state.team_wins[mm.player_id]
      enemy_win = env_state.team_wins[mm.enemy_id]
      extras[3] = team_win / TEAM_WIN_NORM
      extras[4] = (team_win - enemy_win) / TEAM_WIN_NORM

      hidden_relics_num = (env_state.relic_nodes_map_weights > 0).sum()
      if hidden_relics_num == 0:
        hidden_relics_num = 1
      team_points = env_state.team_points[mm.player_id]
      enemy_points = env_state.team_points[mm.enemy_id]
      extras[5] = team_points / TEAM_POINTS_NORM / hidden_relics_num

      extras[6] = (team_points - enemy_points) / TEAM_POINTS_NORM
      extras[6] = max(min(extras[6], 1), -1)

      team_energy = get_units_total_energy(env_state, mm.player_id)
      enemy_energy = get_units_total_energy(env_state, mm.enemy_id)
      extras[7] = team_energy / 1000
      extras[8] = (team_energy - enemy_energy) / 1000

      extras[9] = hidden_relics_num / MAX_HIDDEN_RELICS_NUM

      relic_num = env_state.relic_nodes_mask.sum()
      extras[10] = relic_num / MAX_RELIC_NODE_NUM

      nodes = env_state.relic_nodes[env_state.relic_nodes_mask]
      if len(nodes) > 0:
        extras[11] = nodes.sum(axis=-1).min() / MAP_WIDTH

      extras[12] = mm.units_dead_count / MAX_UNIT_NUM
      extras[13] = mm.step_units_frozen_count / MAX_UNIT_NUM
      extras[14] = mm.units_frozen_count / MAX_UNIT_NUM

      extras[15] = mm.unit_sensor_range / MAX_SENSOR_RANGE
      extras[16] = mm.unit_move_cost / MAX_MOVE_COST
      extras[17] = mm.unit_sap_cost / MAX_SAP_COST
      extras[18] = mm.unit_sap_range / MAX_SAP_RANGE
      extras[19] = mm.nebula_vision_reduction / MAX_VISION_REDUCTION
      extras[20] = mm.nebula_energy_reduction / MAX_ENERGY_REDUCTION

      # print(nodes)
      # print(mm.game_step, self._seed, nodes.sum(axis=-1).min())
      # print(
      # f"step={mm.game_step}, match={mm.game_step // (MAX_MATCH_STEPS + 1)}, "
      # f"match_step={mm.match_step}, team_win={team_win}, diff_win={(team_win - enemy_win)}, "
      # f"tp={team_points}, diff_p={team_points - enemy_points}, hidden_relics_num={hidden_relics_num}, relic_num={relic_num}, "
      # f"energy={team_energy}, diff_e={(team_energy - enemy_energy)}")
      # print(extras)
      return extras

    # Game params
    o['unit_move_cost'] = scalar(mm.unit_move_cost, MAX_MOVE_COST)
    o['unit_sensor_range'] = scalar(mm.unit_sensor_range, MAX_SENSOR_RANGE)
    o['unit_sap_cost'] = scalar(mm.unit_sap_cost, MAX_SAP_COST)
    o['unit_sap_range'] = scalar(mm.unit_sap_range, MAX_SAP_RANGE)
    o['nebula_energy_reduction'] = scalar(mm.nebula_energy_reduction,
                                          MAX_ENERGY_REDUCTION)

    # Time & Match
    o['game_step'] = scalar(mm.game_step, MAX_GAME_STEPS)
    o['match_step'] = scalar(mm.match_step, MAX_MATCH_STEPS)

    team_points = ob['team_points']
    units_points = team_points[mm.player_id]
    enemy_points = team_points[mm.enemy_id]
    o['units_team_points'] = scalar(units_points, TEAM_POINTS_NORM)
    o['enemy_team_points'] = scalar(
        max(min(units_points - enemy_points, TEAM_POINTS_NORM),
            -TEAM_POINTS_NORM), TEAM_POINTS_NORM)

    prev_team_points = self.prev_raw_obs[mm.player]['team_points']
    o['units_team_points_delta'] = scalar(
        max(units_points - prev_team_points[mm.player_id], 0), MAX_UNIT_NUM)
    o['enemy_team_points_delta'] = scalar(
        max(enemy_points - prev_team_points[mm.enemy_id], 0), MAX_UNIT_NUM)

    team_wins = ob['team_wins']
    units_wins = team_wins[mm.player_id]
    enemy_wins = team_wins[mm.enemy_id]
    o['units_wins'] = scalar(units_wins, TEAM_WIN_NORM)
    o['enemy_wins'] = scalar(units_wins - enemy_wins, TEAM_WIN_NORM)

    # Map info
    o['cell_type'] = mm.cell_type.copy()

    v = np.zeros(MAP_SHAPE2)
    v[mm.cell_type == CELL_NEBULA] = -(mm.nebula_vision_reduction /
                                       MAX_VISION_REDUCTION)
    o['nebula_tile_vision_reduction'] = v

    norm = (mm.unit_sensor_range + 1) * MAX_UNIT_NUM
    o['vision_map'] = (mm.vision_map.vision).astype(np.float32) / norm

    o['visible'] = mm.visible.astype(np.float32)
    o['match_observed'] = mm.match_observed.astype(np.float32)
    # o['game_observed'] = mm.game_observed.astype(np.float32)
    o['match_visited'] = mm.match_visited.astype(np.float32)
    o['game_visited'] = mm.match_visited.astype(np.float32)
    o['is_relic_node'] = mm.is_relic_node.astype(np.float32)

    # cells need a visit
    o['game_unvisited_relic_nb'] = mm.get_relic_nb_nodes_to_visit()

    # places need unit stay
    o['team_point_prob'] = mm.get_must_be_relic_nodes()

    # energy_map = np.zeros(MAP_SHAPE2)
    energy_map = mm.cell_energy.copy()
    energy_map[mm.cell_type == CELL_NEBULA] -= mm.nebula_energy_reduction
    o['cell_energy'] = energy_map / MAX_ENERTY_PER_TILE

    # print(
    # f"nebula_energy_reduction={mm.nebula_energy_reduction}, vision_reduction={mm.nebula_vision_reduction}"
    # )

    def add_unit_feature(prefix, player_id, t):
      unit_pos = np.zeros(MAP_SHAPE2)
      unit_energy = np.zeros(MAP_SHAPE2)
      for i in range(MAX_UNIT_NUM):
        mask, pos, energy = mm.get_unit_info(player_id, i, t)
        if mask:
          # assert MAP_WIDTH > pos[
          # 0] >= 0, f"pid={player_id}, pos={pos[0]}, {pos[1]}; energy={energy}, mask={mask}"
          # assert MAP_HEIGHT > pos[
          # 1] >= 0, f"pid={player_id}, pos={pos[0]}, {pos[1]}; energy={energy}, mask={mask}"
          # assert energy >= 0, f"step={mm.game_step}, pid={player_id}, pos={pos[0]}, {pos[1]}; energy={energy}, mask={mask}"
          # Why energy is negative
          unit_energy[pos[0]][pos[1]] += (energy / MAX_UNIT_ENERGY /
                                          MAX_UNIT_NUM)

        # Assume position is always present
        unit_pos[pos[0]][pos[1]] += (1 / MAX_UNIT_NUM)

      o[f'{prefix}_loc_t{t}'] = unit_pos
      o[f'{prefix}_energy_t{t}'] = unit_energy

    def add_unit_info(player_id, t):
      units_info = np.zeros((MAX_UNIT_NUM, 3), dtype=np.int32)
      for i in range(MAX_UNIT_NUM):
        mask, pos, energy = mm.get_unit_info(player_id, i, t)
        # add units info
        units_info[i][0] = pos[0]
        units_info[i][1] = pos[1]
        units_info[i][2] = np.int32(energy) if mask else 0
      o[f'_units_info'] = units_info

    # Unit info
    add_unit_feature('units', mm.player_id, t=0)
    add_unit_feature('units', mm.player_id, t=1)
    add_unit_feature('enemy', mm.enemy_id, t=0)
    add_unit_feature('enemy', mm.enemy_id, t=1)
    add_unit_info(mm.player_id, t=0)

    o['_baseline_extras'] = extract_baseline_extras(mm, final_state)

    if not skip_check:
      assert len(o) == len(OB), f"len(o)={len(o)}, len(OB)={len(OB)}"

    # expand all feature map with dummy dim 1
    o = {k: np.expand_dims(v, 0) for k, v in o.items()}
    return o

  def observation(self, raw_obs, final_state):
    assert len(
        raw_obs
    ) == 2, f"len(raw_obs)={len(raw_obs)}, self.total_agent_controls={self.total_agent_controls}"
    if SINGLE_PLAER:
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

  def _convert_win_loss_reward2(self, raw_obs, env_state):

    def _convert(mm, ob):
      # game end reward
      r_game = 0
      team_wins = raw_obs[mm.player]['team_wins']
      if self.is_game_done(raw_obs, mm.player):
        if team_wins[mm.player_id] > team_wins[mm.enemy_id]:
          r_game = 1
        elif team_wins[mm.player_id] < team_wins[mm.enemy_id]:
          r_game = -1

      # print(f'step={mm.game_step} match-step={mm.match_step}, r_game={r_game}')
      return r_game

    return [
        _convert(self.mms[i], raw_obs[p])
        for i, p in enumerate([PLAYER0, PLAYER1])
    ]

  def _convert_reward(self, raw_obs, env_state):
    """Use the match win-loss reward for now."""
    assert self.reward_schema in ('shaping', 'game_win_loss2',
                                  'match_win_loss')
    if self.reward_schema == 'game_win_loss2':
      reward = self._convert_win_loss_reward2(raw_obs, env_state)

    if self.reward_schema == 'shaping':
      reward = self._convert_shaping_reward(raw_obs, env_state)

    if self.reward_schema == 'match_win_loss':
      reward = self._convert_match_win_loss(raw_obs, env_state)

    if SINGLE_PLAER:
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

      # Unit runs out of energy
      if energy < mm.unit_move_cost:
        continue

      units.append((energy, i, pos))

    def update_move_action_mask(i, pos, energy):
      # has enough energy to move
      for k in range(1, MAX_MOVE_ACTION_IDX + 1):
        nx, ny = (pos[0] + DIRECTIONS[k][0], pos[1] + DIRECTIONS[k][1])
        if nx < 0 or nx >= MAP_WIDTH:
          continue
        if ny < 0 or ny >= MAP_HEIGHT:
          continue
        if mm.cell_type[nx][ny] == CELL_ASTERIOD:
          continue
        actions_mask[i][k] = 1

      # Can only stay on hidden relic node
      if mm.team_point_mass[pos[0]][pos[1]] >= MIN_TP_VAL:

        # Only one units can stay and force it to stay there
        if pos not in action_centered_positions:
          # actions_mask[i][:MOVE_ACTION_NUM] = 0
          actions_mask[i][ACTION_CENTER] = 1
          action_centered_positions.add(pos)

      # Can only stay on green cell (not relic node) for more energy
      # if unit energy < 100
      if (energy < 100
          and (not mm.team_point_mass[pos[0]][pos[1]] >= MIN_TP_VAL)
          and mm.cell_energy[pos[0]][pos[1]] >= mm.unit_move_cost):
        actions_mask[i][ACTION_CENTER] = 1

    def update_sap_action_mask(i, pos, energy):
      if energy < mm.unit_sap_cost:
        return

      # TODO: how to coodinate sap to not sap on same cell
      unit_sap_mask = mm.get_sap_mask(pos, mm.unit_sap_range)
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

    def count_actions(info, action, taken_masks):
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

      info['_game_total_hidden_relic_nodes_num'] = (
          env_state.relic_nodes_map_weights > 0).sum()
      info['_game_total_found_relic_nodes_num'] = (mm.team_point_mass
                                                   >= 30).sum()

      step = raw_obs[PLAYER0]['steps']
      # print(f"step={step} match_step={match_step}, step_reward={step_reward}")
      count_actions(info, agent_action,
                    self._actions_taken_mask[mm.player_id][UNITS_ACTION])
      add_unit_total_energy(info, mm)
      return info

    if model_action is None:
      model_action = [None, None]

    if SINGLE_PLAER:
      return [
          _get_info(model_action[0], raw_obs[PLAYER0],
                    self.prev_raw_obs[PLAYER0], self.mms[0])
      ]  # single player
    else:
      return [
          _get_info(model_action[i], raw_obs[player],
                    self.prev_raw_obs[player], reward[i], self.mms[i],
                    env_state) for i, player in enumerate([PLAYER0, PLAYER1])
      ]

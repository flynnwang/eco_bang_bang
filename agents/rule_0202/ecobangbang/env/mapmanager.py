from collections import OrderedDict, deque, defaultdict, Counter
from functools import cached_property, lru_cache
import copy
import random
import sys

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from .const import *


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def manhatten_distance(p1, p2):
  return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_player_init_pos(player_id):
  target_pos = (0, 0)
  if player_id == 1:
    target_pos = (23, 23)
  return target_pos


def generate_manhattan_mask(shape, center, range_limit):
  rows, cols = shape
  x_center, y_center = center
  x, y = np.ogrid[:rows, :cols]
  manhattan_distance = np.abs(x - x_center) + np.abs(y - y_center)
  mask = manhattan_distance <= range_limit
  return mask


def gen_sap_range(pos, d, dtype=bool, val=True):
  sap_range = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=dtype)
  x0 = max(0, (pos[0] - d))
  x1 = min(MAP_WIDTH, (pos[0] + d + 1))
  y0 = max(0, (pos[1] - d))
  y1 = min(MAP_HEIGHT, (pos[1] + d + 1))
  sap_range[x0:x1, y0:y1] = val
  return sap_range


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


def pos_equal(x, y):
  return x[0] == y[0] and x[1] == y[1]


class UnitSapDropoffFactorEstimator:

  VALID_VALUES = [0.25, 0.5, 1]

  def __init__(self, mm):
    self.mm = mm
    self._counter = Counter()

  def _add_dropoff_factor(self, factor):
    for v in self.VALID_VALUES:
      if abs(factor - v) < 1e-5:
        self._counter[v] += 1

    # print(f"add dropoff={factor},  counter: {self._counter}", file=sys.stderr)
    # print(f" -- dropoff best_guess: {self.best_guess()}", file=sys.stderr)

  def index(self):
    v = self.best_guess()
    return self.VALID_VALUES.index(v)

  def best_guess(self):
    if len(self._counter) <= 0:
      return self.VALID_VALUES[1]  # default 0.5
    return self._counter.most_common(1)[0][0]

  def estimate(self, sap_locations):
    dropoff = np.zeros((MAP_WIDTH, MAP_HEIGHT))
    full_sap = np.zeros((MAP_WIDTH, MAP_HEIGHT))
    for pos in sap_locations:
      d = 1
      x0 = max(0, (pos[0] - d))
      x1 = min(MAP_WIDTH, (pos[0] + d + 1))
      y0 = max(0, (pos[1] - d))
      y1 = min(MAP_HEIGHT, (pos[1] + d + 1))
      dropoff[x0:x1, y0:y1] += 1
      dropoff[pos[0]][pos[1]] -= 1  # exclude the center

      full_sap[pos[0]][pos[1]] += 1

    for i in range(MAX_UNIT_NUM):
      m0, p0, e0 = self.mm.get_unit_info(self.mm.enemy_id, i, t=0)
      if not m0:
        continue
      m1, p1, e1 = self.mm.get_unit_info(self.mm.enemy_id, i, t=1)
      if not m1:
        continue

      dropoff_num = dropoff[p0[0]][p0[1]]
      if dropoff_num > 0:
        # TODO: consider energy void field?
        full_sap_cost = full_sap[p0[0]][p0[1]] * self.mm.unit_sap_cost
        cell_energy = self.mm.cell_energy[p0[0]][p0[1]]

        nebula_energy = 0
        if self.mm.cell_type[p0[0]][p0[1]] == CELL_NEBULA:
          nebula_energy = self.mm.nebula_energy_reduction

        move_cost = 0
        if not pos_equal(p0, p1):
          move_cost = self.mm.unit_move_cost

        energy_delta = e1 - e0
        # print(
        # f"enemy={p1} to {p0} id=[{i}] energy delta={energy_delta}, full_sap={full_sap_cost}, cell_energy={cell_energy}, nebula_energy={nebula_energy}, move_cost={move_cost}",
        # file=sys.stderr)

        energy_delta -= (full_sap_cost - cell_energy + nebula_energy +
                         move_cost)

        dropoff_factor = (energy_delta / dropoff_num) / self.mm.unit_sap_cost
        self._add_dropoff_factor(dropoff_factor)


class HiddenRelicNodeEstimator:

  def __init__(self, enable_anti_sym):
    self.priori = np.zeros((MAP_WIDTH, MAP_HEIGHT))
    self.relic_node_positions = set()
    self.enable_anti_sym = enable_anti_sym

  def check_new_relic_nodes(self, relic_node_positions):
    new_relic_node_positions = []
    for pos in relic_node_positions:
      pos = (int(pos[0]), int(pos[1]))
      if pos not in self.relic_node_positions:
        new_relic_node_positions.append(pos)

    if new_relic_node_positions:
      self.relic_node_positions.update(new_relic_node_positions)
    return np.array(new_relic_node_positions, dtype=int)

  def update(self, relic_node_positions, is_relic_neighbour, unit_positions,
             new_team_points):
    # first find the newly found relic node positions
    new_relic_node_positions = self.check_new_relic_nodes(relic_node_positions)
    # if new_relic_node_positions.size > 0:
    # print(f"new_relic_node_positions: {new_relic_node_positions}")

    # update (or reset) new relic node nb with priori
    new_relic_nb_mask = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    if new_relic_node_positions.size > 0:
      new_relic_nb_mask[new_relic_node_positions[:, 0],
                        new_relic_node_positions[:, 1]] = True
    # if self.enable_anti_sym:
    # new_relic_nb_mask = new_relic_nb_mask | (
    # anti_diag_sym(new_relic_nb_mask))

    new_relic_nb_mask = maximum_filter(new_relic_nb_mask, size=RELIC_NB_SIZE)
    # if new_relic_nb_mask.sum() > 0:
    # print(f"reset new_relic_nb_mask: {new_relic_nb_mask.sum()}")
    tmp_pri = self.priori.copy()
    tmp_pri[new_relic_nb_mask] = PRIORI
    self.priori = np.maximum(self.priori, tmp_pri)

    is_relc_nb = (is_relic_neighbour == 1)
    obs = np.argwhere(is_relc_nb & unit_positions)
    post = self.calc_posteriori_probs(obs, new_team_points)
    # print(new_team_points, post[post > 0])

    self.priori[is_relc_nb] = np.clip(post[is_relc_nb], MIN_PROB, MAX_PROB)

  def calculate_p_data(self, obs, n, p):
    if n < 0:
      return 0

    m = len(obs)

    # represent the probability of observe k team points from the first j locations.
    dp = np.zeros((m + 1, n + 1))
    dp[0][0] = 1

    for j in range(1, m + 1):
      pos = obs[j - 1]
      pj = p[pos[0]][pos[1]]
      for k in range(n + 1):
        dp[j][k] += dp[j - 1][k] * (1 - pj)
        if k > 0:
          dp[j][k] += dp[j - 1][k - 1] * pj
    return dp[m][n]

  def calc_posteriori_probs(self, observed_cells, new_team_points):
    m = len(observed_cells)
    n = min(new_team_points, m)

    p_data = self.calculate_p_data(observed_cells, n, self.priori)
    if p_data <= 0:
      return self.priori
    assert p_data > 0
    # print(f'p_data = {p_data}')

    post = self.priori.copy()
    for i in observed_cells:
      x, y = i
      ob_exclude = [c for c in observed_cells if not (c[0] == x and c[1] == y)]
      # print(observed_cells, ob_exclude, new_team_points)
      p_data_given_i = self.calculate_p_data(ob_exclude, n - 1, self.priori)

      post_prob = p_data_given_i * self.priori[x][y] / p_data
      post[x][y] = post_prob
      # post[y][x] = post_prob
      # print(
      # f"post[{x}][{y}] = {post[x][y]}, self.priori[x][y]={self.priori[x][y]}, p_data_given_i={p_data_given_i}"
      # )
    return post


class NebulaEnergyReduction:

  VALID_VALUES = [0, 1, 2, 3, 5, 25]

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

  def index(self):
    v = self.best_guess()
    return self.VALID_VALUES.index(v)


class VisionMap:

  def __init__(self, unit_sensor_range):
    self.vision = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=np.int32)
    self.unit_sensor_range = unit_sensor_range

  def _add_vision(self, p, unit_sensor_range, wt):
    i, j = p
    v = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=np.int32)
    for r in range(unit_sensor_range + 1):
      d = (unit_sensor_range - r)
      x0 = max(0, (i - d))
      x1 = min(MAP_WIDTH, (i + d + 1))
      y0 = max(0, (j - d))
      y1 = min(MAP_HEIGHT, (j + d + 1))

      vis = (r + 1) * wt
      if d == 0:
        vis = UNIT_POSITION_VISION_POWER * wt

      v[x0:x1, y0:y1] = vis
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


def relic_score_nodes_map(state):
  return ((state.relic_nodes_map_weights <= state.relic_nodes_mask.sum() // 2)
          & (state.relic_nodes_map_weights > 0))


def count_relic_score_nodes_num(state):
  return relic_score_nodes_map(state).sum()


class SapIndexer:

  def __init__(self):
    self.position_to_idx = {}
    self.idx_to_position = {}

    mat = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    center = (12, 12)
    self.mask = gen_sap_range(center, MAX_SAP_RANGE)

    delta_positions = np.argwhere(self.mask > 0) - np.array(center,
                                                            dtype=np.int32)
    for i, (x, y), in enumerate(delta_positions):
      self.position_to_idx[(x, y)] = i
      self.idx_to_position[i] = (x, y)


class MapManager:

  MAX_PAST_OB_NUM = 3

  def __init__(self,
               player,
               env_cfg,
               transpose=False,
               sap_indexer=None,
               use_mirror=False,
               use_hidden_relic_estimator=False,
               enable_anti_sym=True,
               full_params=None):
    self.player_id = int(player[-1])
    self.player = player
    self.env_cfg = env_cfg
    self.transpose = transpose
    self.cell_type = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.visible = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.match_observed = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    self.last_observed_step = np.ones(
        (MAP_WIDTH, MAP_HEIGHT), dtype=np.int32) * -100
    self.last_visited_step = np.ones(
        (MAP_WIDTH, MAP_HEIGHT), dtype=np.int32) * -100
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

    self.use_mirror = use_mirror
    self.use_hidden_relic_estimator = use_hidden_relic_estimator
    self.hidden_relic_estimator = HiddenRelicNodeEstimator(enable_anti_sym)

    self.energy_cost_map = None
    self.units_energy_cost_change = 0

    self.enable_anti_sym = enable_anti_sym
    self.full_params = full_params

    self.match_wins = 0

    self.sap_dropoff_factor_estimator = UnitSapDropoffFactorEstimator(self)

  def add_sap_locations(self, sap_locations):
    self.sap_dropoff_factor_estimator.estimate(sap_locations)

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
  def unit_sap_dropoff_factor(self):
    return self.sap_dropoff_factor_estimator.best_guess()

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
    if self.enable_anti_sym:
      cells_sym = anti_diag_sym(cells)
      ct = cells_sym > CELL_UNKONWN
      self.cell_type[ct] = cells_sym[ct]

  def update_visible_and_observed(self, ob):
    self.visible = ob['sensor_mask'].astype(bool)
    self.prev_match_observed = self.match_observed.copy()
    self.prev_game_observed = self.game_observed.copy()
    self.match_observed |= self.visible
    self.game_observed |= self.visible

    # set last observed time
    self.last_observed_step[self.visible] = self.game_step
    self.last_observed_step[anti_diag_sym(self.visible)] = self.game_step

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

    self.true_team_point_map = mat_trans(self.true_team_point_map)
    self.true_relic_map = mat_trans(self.true_relic_map)
    self.true_cell_type = mat_trans(self.true_cell_type)
    self.true_cell_energy = mat_trans(self.true_cell_energy)

  def infer_nebula_energy_reduction(self, ob, model_action):
    if (model_action is None
        or len(self.past_obs) == 0) or ob['match_steps'] <= 1:
      return

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

  def update_frozen_or_dead_units(self):
    self.prev_units_on_relic_num = self.units_on_relic_num
    self.prev_units_frozen_count = self.units_frozen_count
    self.prev_units_dead_count = self.units_dead_count

    if self.match_step <= 1 or len(self.past_obs) < 2:
      return

    # if env_state is not None:
    # self.units_on_relic_num = self.count_on_relic_nodes_units(env_state)

    self.units_frozen_count = 0
    self.units_dead_count = 0
    self.units_position_energy_sum = 0
    self.units_energy_cost_change = 0

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

        if self.energy_cost_map is not None:
          c0 = self.energy_cost_map[p0[0]][p0[1]]
          c1 = self.energy_cost_map[p1[0]][p1[1]]
          if not np.isinf(c0) and not np.isinf(c1):
            self.units_energy_cost_change += (c1 - c0)  # cost net reduced

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

  def append_ob(self, ob):
    self.past_obs.appendleft(ob)

    # append some default observation to make get_unit_info easier
    while len(self.past_obs) < self.MAX_PAST_OB_NUM:
      self.past_obs.appendleft(ob)

    if len(self.past_obs) > self.MAX_PAST_OB_NUM:
      self.past_obs.pop()

  def record_ground_truth_features(self, env_state=None):
    from luxai_s3.utils import to_numpy

    # Mirror should go first before everything else.
    self.true_team_point_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.true_relic_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.true_cell_type = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.true_cell_energy = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)

    if env_state is not None:
      self.true_team_point_map = to_numpy(
          relic_score_nodes_map(env_state)).astype(np.int32)

      relic_nodes_mask = env_state.relic_nodes_mask
      relic_nodes_positions = env_state.relic_nodes[relic_nodes_mask]
      self.true_relic_map[relic_nodes_positions[:, 0],
                          relic_nodes_positions[:, 1]] = 1

      self.true_cell_type = to_numpy(env_state.map_features.tile_type) + 1
      self.true_cell_energy = to_numpy(env_state.map_features.energy)

  def update(self, ob, model_action=None, env_state=None):
    self.record_ground_truth_features(env_state)

    if self.use_mirror:
      self.mirror(ob)

    if self.transpose:
      self.mirror(ob, 'T')

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

      # self.append_ob(ob)

      # use match_step=0 reset map manager, do not update below
      # return

    # use non-infered units position
    self.vision_map.update(ob['units_mask'][self.player_id],
                           ob['units']['position'][self.player_id],
                           ob['units']['energy'][self.player_id])

    self.infer_nebula_energy_reduction(ob, model_action)
    self.game_step = ob['steps']
    self.match_step = ob['match_steps']
    self.update_counters()

    self.update_visible_and_observed(ob)
    self.update_cell_type(ob)

    unit_masks = ob['units_mask'][self.player_id]
    unit_positions = ob['units']['position'][self.player_id][unit_masks]
    self.update_visited_node(unit_positions, ob)

    self.update_relic_node(ob)

    if not self.use_hidden_relic_estimator:
      self.update_team_point_mass(ob, unit_positions)

    self.update_cell_energy(ob)

    self.append_ob(ob)

    self.update_frozen_or_dead_units()
    self.update_vision_map()
    self.update_sap_position_by_enemy_position()

    if self.use_hidden_relic_estimator and ob['match_steps'] > 0:
      self.update_hidden_relic_estimator(ob)

    self.energy_cost_map = self.compute_energy_cost_map(
        self.cell_type, self.cell_energy, self.is_relic_node,
        self.nebula_energy_reduction)
    self.prev_team_point = ob['team_points'][self.player_id]
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

    self.enemy_max_energy = np.zeros((MAP_SHAPE2), dtype=int)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.get_unit_info(self.enemy_id, i, t=0)
      if mask:
        self.enemy_max_energy[pos[0]][pos[1]] = max(
            self.enemy_max_energy[pos[0]][pos[1]], energy)

  def update_vision_map(self):
    nebula_cell_mask = (self.visible <= 0) & (self.vision_map.vision > 0)

    # Update cell map with guessed nebula
    self.cell_type[nebula_cell_mask] = CELL_NEBULA
    if self.enable_anti_sym:
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
    if not mask or energy < 0:
      position = (0, 0) if self.player_id == 0 else (MAP_WIDTH - 1,
                                                     MAP_HEIGHT - 1)

    return mask, position, energy

  def get_game_visited_relic_nb_num(self):
    return ((self.game_visited) & (self.is_relic_neighbour > 0)).sum()

  @property
  def step_new_visited_relic_nb_num(self):
    return (self.get_game_visited_relic_nb_num() -
            self.last_game_visited_relic_nb_num)

  @property
  def cell_energy_with_nebula_energy_reduction(self):
    energy_map = self.cell_energy.copy()
    energy_map[self.cell_type == CELL_NEBULA] -= self.nebula_energy_reduction
    return energy_map

  def update_visited_node(self, unit_positions, ob):
    self.unit_positions = np.zeros((MAP_SHAPE2), dtype=bool)
    self.unit_positions[unit_positions[:, 0], unit_positions[:, 1]] = True
    self.match_visited[self.unit_positions] = 1
    self.game_visited |= self.match_visited

    self.last_visited_step[self.unit_positions] = self.game_step

    enemy_masks = ob['units_mask'][self.enemy_id]
    enemy_positions = ob['units']['position'][self.enemy_id][enemy_masks]
    self.enemy_positions = np.zeros((MAP_SHAPE2), dtype=bool)
    self.enemy_positions[enemy_positions[:, 0], enemy_positions[:, 1]] = True

  def count_on_relic_nodes_units(self):
    return ((self.team_point_mass > MIN_TP_VAL) & (self.unit_positions)).sum()

  @property
  def step_new_found_relic_node_num(self):
    return self.is_relic_node.sum() - self.last_relic_node_num

  def update_relic_node(self, ob):
    relic_nodes_mask = ob['relic_nodes_mask']
    relic_nodes_positions = ob['relic_nodes'][relic_nodes_mask]
    self.is_relic_node[relic_nodes_positions[:, 0],
                       relic_nodes_positions[:, 1]] = 1
    if self.enable_anti_sym:
      self.is_relic_node |= anti_diag_sym(self.is_relic_node)

    self.is_relic_neighbour = maximum_filter(
        (self.is_relic_node == 1).astype(np.int32), size=RELIC_NB_SIZE)

  def update_cell_energy(self, ob):
    energy = ob['map_features']['energy']
    is_visible = (self.visible > 0)
    self.cell_energy[is_visible] = energy[is_visible]

    if self.enable_anti_sym:
      energy_tr = anti_diag_sym(energy)
      is_visible_tr = anti_diag_sym(is_visible)
      self.cell_energy[is_visible_tr] = energy_tr[is_visible_tr]

  @lru_cache(maxsize=None)
  def get_player_half_mask(self, player_id):
    init_pos = get_player_init_pos(player_id)
    return generate_manhattan_mask(MAP_SHAPE2, init_pos, range_limit=MAP_WIDTH)

  def update_hidden_relic_estimator(self, ob):
    relic_nodes_mask = ob['relic_nodes_mask']
    relic_nodes_positions = ob['relic_nodes'][relic_nodes_mask]

    team_point = ob['team_points'][self.player_id]
    new_team_points = team_point - self.prev_team_point
    self.hidden_relic_estimator.update(relic_nodes_positions,
                                       self.is_relic_neighbour,
                                       self.unit_positions, new_team_points)

    p = self.hidden_relic_estimator.priori.copy()
    self.team_point_mass = p.copy()

    units_half = self.get_player_half_mask(self.player_id)

    min_val = MIN_PROB + 1e-5
    non_relic_point = (p <= min_val) & (p > 0) & units_half
    self.team_point_mass[non_relic_point | anti_diag_sym(non_relic_point)] = 0

    max_val = MAX_PROB - 1e-5
    is_relc_point = (p >= max_val) | anti_diag_sym(p >= max_val)
    self.team_point_mass[is_relc_point] = 1

    # print(f'update_hidden_relic_estimator: {self.team_point_mass}, {tmp}')

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
      if self.enable_anti_sym:
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
      if self.enable_anti_sym:
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
      if self.enable_anti_sym:
        self.team_point_mass[anti_diag_sym(team_point_candidate)] -= change
    elif delta >= num:
      # Every candidate position is a team point position
      self.team_point_mass[team_point_candidate] += change
      if self.enable_anti_sym:
        self.team_point_mass[anti_diag_sym(team_point_candidate)] += change
    else:
      # num < delta, some of the point is team point
      assert delta < num
      # print('>>>>>>>>>>>>>>', ob['steps'], delta, num, must_be_team_point.sum(), non_team_point.sum())
      self.team_point_mass[team_point_candidate] += (delta / num)
      if self.enable_anti_sym:
        self.team_point_mass[anti_diag_sym(team_point_candidate)] += (delta /
                                                                      num)

  def get_relic_nb_nodes_to_visit(self):
    to_visit = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    to_visit[(~self.game_visited) & (self.is_relic_neighbour > 0)] = 1
    return to_visit

  def get_global_sap_hit_map(self):
    hit_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.get_unit_info(self.enemy_id, i, t=0)
      if not mask or energy < 0:
        continue

      d = 1
      x0 = max(0, (pos[0] - d))
      x1 = min(MAP_WIDTH, (pos[0] + d + 1))
      y0 = max(0, (pos[1] - d))
      y1 = min(MAP_HEIGHT, (pos[1] + d + 1))
      hit_map[x0:x1, y0:y1] += True

    # Add unvisible team point positions
    if self.player_id == 1:
      assert self.transpose == False
      assert self.use_mirror == False

    init_pos = get_player_init_pos(self.enemy_id)
    enemy_half = generate_manhattan_mask(MAP_SHAPE2,
                                         init_pos,
                                         range_limit=MAP_WIDTH - 1)
    hit_map[(self.visible == 0) & (self.team_point_mass > 0.8)
            & enemy_half] = True
    return hit_map

  @property
  def game_observed_num(self):
    return (self.cell_type != CELL_UNKONWN).sum()

  @property
  def match_observed_num(self):
    # return (self.match_observed > 0).sum()
    # return (self.visible > 0).sum()
    return (self.match_visited > 0).sum()

  def compute_energy_cost_map(self, cell_type, cell_energy, is_relic_node,
                              nebula_energy_reduction):
    cost_map = np.full((MAP_WIDTH, MAP_HEIGHT), float(self.unit_move_cost))

    # nebula energy reduction adds extra cost
    cost_map[cell_type == CELL_NEBULA] += nebula_energy_reduction

    # cell energy cost change the cost map but max at 0 to prevent from loop
    cost_map -= cell_energy
    cost_map = np.maximum(cost_map, 1)

    # use a big value for asteriod
    cost_map[cell_type == CELL_ASTERIOD] = 100

    energy_cost = np.full((MAP_WIDTH, MAP_HEIGHT), np.inf, dtype=np.float64)

    seed_mask = (self.is_relic_node > 0)
    energy_cost[seed_mask] = 0

    N = MAP_WIDTH * 2
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    for _ in range(N):
      min_neighbors = minimum_filter(energy_cost,
                                     footprint=kernel,
                                     mode='constant',
                                     cval=np.inf)
      with np.errstate(invalid='ignore'):
        energy_cost = np.minimum(energy_cost, min_neighbors + cost_map)

    return energy_cost

  def get_erengy_cost_map_feature(self, energy_cost):
    # Normalize the cost values
    not_inf = ~np.isinf(energy_cost)
    non_inf_cost = energy_cost[not_inf]
    if non_inf_cost.size > 0:
      mx = non_inf_cost.max()
      energy_cost[not_inf] /= mx

    energy_cost[np.isinf(energy_cost)] = 1.1
    return energy_cost

  def to_last_sap_actions(self, model_action):
    sap_locations = []

    actions = model_action[UNITS_ACTION]
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.get_unit_info(self.player_id, i, t=0)
      if not mask or energy <= 0:
        continue

      a = actions[i][0]
      if a < MOVE_ACTION_NUM:
        continue

      if energy < self.unit_sap_cost:
        continue

      a -= MOVE_ACTION_NUM
      x, y = self.sap_indexer.idx_to_position[a]

      sap_loc = (pos[0] + x, pos[1] + y)
      if not is_pos_on_map(sap_loc):
        __import__('ipdb').set_trace()
      sap_locations.append(sap_loc)
    return sap_locations

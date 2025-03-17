from collections import OrderedDict, deque, defaultdict, Counter
from functools import cached_property, lru_cache
import copy
import random
import sys
from datetime import datetime

import torch
import chex
import jax
import jax.numpy as jnp
from jax import lax
from flax import struct

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from .const import *

SAVE_ALL_STEPS_TP_PROB = False


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def manhatten_distance(p1, p2):
  return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_player_init_pos(player_id, use_mirror):
  if not use_mirror:
    target_pos = (0, 0)
    if player_id == 1:
      target_pos = (23, 23)
  else:
    target_pos = (0, 0)
    if player_id == 0:
      target_pos = (23, 23)
  return target_pos


def generate_manhattan_dist(shape, center):
  rows, cols = shape
  x_center, y_center = center
  x, y = np.ogrid[:rows, :cols]
  manhattan_distance = np.abs(x - x_center) + np.abs(y - y_center)
  return manhattan_distance


def generate_manhattan_mask(shape, center, range_limit):
  manhattan_distance = generate_manhattan_dist(shape, center)
  mask = manhattan_distance <= range_limit
  return mask


def set_value_by_range(arr, pos, d, val):
  x0 = max(0, (pos[0] - d))
  x1 = min(MAP_WIDTH, (pos[0] + d + 1))
  y0 = max(0, (pos[1] - d))
  y1 = min(MAP_HEIGHT, (pos[1] + d + 1))
  arr[x0:x1, y0:y1] = val


def gen_sap_range(pos, d, dtype=bool, val=True):
  sap_range = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=dtype)
  set_value_by_range(sap_range, pos, d, val)
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


def is_drifted_step(steps, nebula_tile_drift_speed):
  return ((steps - 1) * abs(nebula_tile_drift_speed) %
          1) > (steps * abs(nebula_tile_drift_speed) % 1)


def shift_map(mp, n):
  mp = np.roll(mp, shift=-n, axis=0)  # Shift Up
  mp = np.roll(mp, shift=n, axis=1)  # Shift right
  return mp


def shift_map_by_sign(mp, drift_speed):
  sign = -(1 if drift_speed > 0 else -1)  # the cell is transposed from numpy
  return shift_map(mp, sign)


ENERGY_NODE_FNS = [
    lambda d, x, y, z: jnp.sin(d * x + y) * z, lambda d, x, y, z:
    (x / (d + 1) + y) * z
]

energy_node_fns = jnp.array([
    [0, 1.2, 1, 4],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    # [1, 4, 0, 2],
    [0, 1.2, 1, 4],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    # [1, 4, 0, 0]
])


@struct.dataclass
class EnvState2:
  energy_nodes: chex.Array
  energy_node_fns: chex.Array
  energy_nodes_mask: chex.Array


def compute_energy_features(state: EnvState2):
  # first compute a array of shape (map_height, map_width, num_energy_nodes) with values equal to the distance of the tile to the energy node
  mm = jnp.meshgrid(jnp.arange(MAP_WIDTH), jnp.arange(MAP_HEIGHT))
  mm = jnp.stack([mm[0], mm[1]]).T.astype(jnp.int16)  # mm[x, y] gives [x, y]
  distances_to_nodes = jax.vmap(
      lambda pos: jnp.linalg.norm(mm - pos, axis=-1))(state.energy_nodes)

  def compute_energy_field(node_fn_spec, distances_to_node, mask):
    fn_i, x, y, z = node_fn_spec
    return jnp.where(
        mask,
        lax.switch(fn_i.astype(jnp.int16), ENERGY_NODE_FNS, distances_to_node,
                   x, y, z),
        jnp.zeros_like(distances_to_node),
    )

  energy_field = jax.vmap(compute_energy_field)(state.energy_node_fns,
                                                distances_to_nodes,
                                                state.energy_nodes_mask)
  # print(energy_field)
  energy_field = jnp.where(
      energy_field.mean() < 0.25,
      energy_field + (0.25 - energy_field.mean()),
      energy_field,
  )
  energy_field = jnp.round(energy_field.sum(0)).astype(jnp.int16)
  energy_field = jnp.clip(energy_field, MIN_ENERGY_PER_TILE,
                          MAX_ENERGY_PER_TILE)
  return energy_field


@lru_cache(maxsize=None)
def compute_energy_features_cached(pos):
  if pos is not None:
    energy_nodes_mask_ = jnp.array([True, False, False, True, False, False],
                                   dtype=jnp.bool)
    pos2 = anti_diag_sym_i(pos)
    energy_nodes = jnp.array([pos, [0, 0], [0, 0], pos2, [0, 0], [0, 0]],
                             dtype=jnp.int16)
  else:
    # Case of no energy field
    energy_nodes_mask_ = jnp.array([False, False, False, False, False, False],
                                   dtype=jnp.bool)
    energy_nodes = jnp.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                             dtype=jnp.int16)

  env_state = EnvState2(energy_node_fns=energy_node_fns,
                        energy_nodes_mask=energy_nodes_mask_,
                        energy_nodes=energy_nodes)
  return compute_energy_features(env_state)


class EnergyNodeEstimator:

  ENERGY_NODE_DRIFT_SPEED_VALUSE = [0.01, 0.02, 0.03, 0.04, 0.05]

  def __init__(self):
    self.counter = Counter()
    self.energy_node = None
    self.energy_node_found = False
    self.current_energy_field = None

    positions = np.argwhere(
        generate_manhattan_mask((MAP_WIDTH, MAP_HEIGHT), (0, 0),
                                MAP_WIDTH - 1))
    self.candidate_energy_posotions = [v for v in positions] + [None]

  def is_energy_filed_match(self, pos, filtered_energy_field, step_visible):
    if pos is not None:
      pos2 = anti_diag_sym_i(pos)
      if pos2[0] < pos[0]:
        pos = pos2

      pos = (int(pos[0]), int(pos[1]))

    test_energy_field = compute_energy_features_cached(pos)
    test_energy_field2 = test_energy_field[step_visible]
    matched = (test_energy_field2 == filtered_energy_field).all()
    return matched, test_energy_field

  def find_energy_node(self, step_energy_field, step_visible):
    filtered_energy_field = step_energy_field[step_visible]
    if filtered_energy_field.size <= 0:
      return False, None

    # print(
    # f" filtered_energy_field={filtered_energy_field}, {np.argwhere(step_visible)}",
    # file=sys.stderr)

    matched_nodes = []
    for pos in self.candidate_energy_posotions:
      matched, field = self.is_energy_filed_match(pos, filtered_energy_field,
                                                  step_visible)
      # print(
      # f" check energy node at {pos}, match={matched}, field={field[step_visible]}",
      # file=sys.stderr)
      if matched:
        matched_nodes.append((pos, field))
        # print(f"***** --> found energy node at {pos}", file=sys.stderr)

    # Update the candidates: TODO: generate energy field generation only once
    self.candidate_energy_posotions = [x[0] for x in matched_nodes]

    if len(matched_nodes) == 1:
      self.energy_node = matched_nodes[0][0]
      self.current_energy_field = np.asarray(matched_nodes[0][1]).copy()
      self.energy_node_found = True

  def update_energy(self, step_energy_field, step_visible, last_energy_field,
                    last_visible, game_step):
    # print(
    # f"step={game_step} (before update), energy field found={self.energy_node_found}, node={self.energy_node}, candidates={self.candidate_energy_posotions}",
    # file=sys.stderr)
    step_visible_num = step_visible.sum()

    comm_visible = (step_visible & last_visible)
    # print(
    # f"[debug][step={game_step+1}] comm_visible.sum() = {comm_visible.sum()}",
    # file=sys.stderr)
    if comm_visible.sum() > 0:
      if (step_energy_field[comm_visible]
          != last_energy_field[comm_visible]).any():

        # in case has found the energy field before, track the position
        if self.energy_node_found:
          self.energy_node_found = False
          if self.energy_node is not None:
            mp = np.zeros(MAP_SHAPE2, dtype=bool)
            # print(f"self.energy_node={self.energy_node}, mp.shape={mp.shape}",
            # file=sys.stderr)

            mp[self.energy_node[0]][self.energy_node[1]] = True
            mp = maximum_filter(mp,
                                size=MAX_ENERGY_NODE_DRIFT_MAGNITUDE * 2 + 1)
            mp[self.energy_node[0]][self.energy_node[1]] = False

            upper = generate_manhattan_mask((MAP_WIDTH, MAP_HEIGHT), (0, 0),
                                            MAP_WIDTH - 1)
            mp[~upper] = False

            self.candidate_energy_posotions = np.argwhere(mp)
            # print(
            # f"----------small reset energy node positions: {self.candidate_energy_posotions}",
            # file=sys.stderr)
            if step_visible_num > 0 and not self.energy_node_found:
              self.find_energy_node(step_energy_field, step_visible)
            # print(
            # f"----------small reset energy node positions, found? = {self.energy_node_found}",
            # file=sys.stderr)

        # In case not found by small reset, Reset to search
        if not self.energy_node_found:
          # reset search range
          positions = np.argwhere(
              generate_manhattan_mask((MAP_WIDTH, MAP_HEIGHT), (0, 0),
                                      MAP_WIDTH - 1))
          self.candidate_energy_posotions = [v for v in positions] + [None]

          # print(f"************Reset energy node positions!", file=sys.stderr)

        # print(f"energy field changed at step = {game_step}", file=sys.stderr)

    if step_visible_num > 0 and not self.energy_node_found:
      self.find_energy_node(step_energy_field, step_visible)

    # print(
    # f"step={game_step} (after update), energy field found={self.energy_node_found}, node={self.energy_node}, candidates={self.candidate_energy_posotions}",
    # file=sys.stderr)


class NebulaDriftEstimator:

  VALID_VALUES = [-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15]

  def __init__(self):
    self.counter = Counter()
    for v in self.VALID_VALUES:
      self.counter[v] = 1
    self.drift_speed = None

  def is_map_matched(self, m1, m2):
    comm_mask = (m1 != CELL_UNKONWN) & (m2 != CELL_UNKONWN)
    if comm_mask.size == 0:
      return False, 0

    matched_cells = m1[comm_mask] == m2[comm_mask]
    return matched_cells.all(), matched_cells.sum()

  def update(self, step_map, last_cells, game_step):
    matched, n = self.is_map_matched(step_map, last_cells)
    drifted = not matched
    # print(f'step={game_step}, nebula drifted={drifted} n={n}', file=sys.stderr)
    if not drifted:
      return

    # map tile drifted
    for v in self.VALID_VALUES:
      if not is_drifted_step(game_step, v):
        self.counter[v] -= 100
        continue

      m1 = shift_map_by_sign(last_cells, v)
      m2 = step_map

      matched, n = self.is_map_matched(m1, m2)
      if matched:
        self.counter[v] += 1
      else:
        self.counter[v] -= 1

    valid_num = 0
    for v in self.VALID_VALUES:
      if self.counter[v] > 0:
        valid_num += 1

    if valid_num == 1:
      self.drift_speed = self.counter.most_common(1)[0][0]

  def index(self):
    if self.drift_speed is None:
      return N_NEBULA_DRIFT_SPPED - 1
    return self.VALID_VALUES.index(self.drift_speed)


class UnitSapDropoffFactorEstimator:

  VALID_VALUES = [0.25, 0.5, 1]

  def __init__(self, mm):
    self.mm = mm
    self._counter = Counter()
    self.unit_energy_lost_step = np.ones(MAP_SHAPE2,
                                         dtype=int) * -MAX_MATCH_STEPS

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

      # Here, we assume the enemy does not trigger sap itself, in that case it is wrong.
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

  def update_unit_energy_lost_map(self):
    self.unit_energy_lost = np.zeros(MAP_SHAPE2, dtype=int)

    for i in range(MAX_UNIT_NUM):
      m0, p0, e0 = self.mm.get_unit_info(self.mm.player_id, i, t=0)
      if not m0:
        continue
      m1, p1, e1 = self.mm.get_unit_info(self.mm.player_id, i, t=1)
      if not m1:
        continue

      energy_delta = e1 - e0
      if energy_delta > 0:
        cell_energy = self.mm.cell_energy[p0[0]][p0[1]]
        nebula_energy = 0
        if self.mm.cell_type[p0[0]][p0[1]] == CELL_NEBULA:
          nebula_energy = self.mm.nebula_energy_reduction
          # print(
          # f" -- -- -- nebula_energy_reduction={nebula_energy}: {self.mm._nebula_energy_reduction.counter}",
          # file=sys.stderr)

        move_cost = 0
        if not pos_equal(p0, p1):
          move_cost = self.mm.unit_move_cost

        sap_cost = 0
        if self.mm.unit_sapped_last_step[i]:
          sap_cost = self.mm.unit_sap_cost

        energy_delta += (cell_energy - nebula_energy - move_cost - sap_cost)
        if energy_delta >= 20:
          self.unit_energy_lost[p0[0]][p0[1]] += energy_delta
          self.unit_energy_lost_step[p0[0]][p0[1]] = self.mm.game_step
          # print(
          # f" >>> step={self.mm.game_step}, unid_id={i}, {p1}=>{p0}, e={e1}=>{e0} ce={cell_energy}, ne={nebula_energy}, mc={move_cost} sap={sap_cost} energy_delta={energy_delta} ",
          # file=sys.stderr)


class Observation:

  def __init__(self, positions, relic_num):
    self.positions = [(int(x), int(y)) for x, y in positions]
    self.relic_num = min(relic_num, len(positions))

  def is_determined(self):
    if len(self.positions) == 0:
      return True
    return self.relic_num == 0 or (self.relic_num == len(self.positions))

  def simplify(self, position_to_relic):
    next_positions = []
    num = self.relic_num
    for pos in self.positions:
      is_relic = position_to_relic.get(pos)
      if is_relic is not None:
        num -= int(is_relic)
        continue
      next_positions.append(pos)
    return Observation(next_positions, num)

  def is_valid_solution(self, pos_to_val):
    num = 0
    for pos in self.positions:
      num += pos_to_val[pos]
    return num == self.relic_num


class HiddenRelicSolver:

  MAX_OB_NUM = 5

  def __init__(self):
    self.position_to_relic = {}
    self.obs = []

  def reset_with_relic_nb(self, new_relic_nb_positions):
    for x, y in new_relic_nb_positions:
      pos = (int(x), int(y))
      is_relic = self.position_to_relic.get(pos)
      if is_relic == False:
        del self.position_to_relic[pos]

  def simplify_obs(self):
    has_determined_pos = True
    while has_determined_pos:
      unsolved_positions = set()
      has_determined_pos = False
      next_obs = []
      for ob in reversed(self.obs):
        ob = ob.simplify(self.position_to_relic)
        if ob.is_determined():
          self.add_determined_observation(ob)
          has_determined_pos = True
        else:
          next_obs.append(ob)
          unsolved_positions.update(ob.positions)
      self.obs = next_obs
    return unsolved_positions

  def add_determined_observation(self, ob):
    is_relic = True
    if ob.relic_num == 0:
      is_relic = False
    for pos in ob.positions:
      self.position_to_relic[pos] = is_relic

  def solve(self, remainingOverageTime):
    # First, simplify all observations to reduce search spaces
    unsolved_positions = self.simplify_obs()
    n = len(unsolved_positions)
    unsolved_positions = list(unsolved_positions)
    # print(f"Solving {n} positions...", file=sys.stderr)

    start_time = datetime.now()
    positions_values = defaultdict(set)
    for s in range(2**n):
      pos_to_val = {}
      tmp = s
      for pos in unsolved_positions:
        pos_to_val[pos] = tmp % 2
        tmp >>= 1

      valid = True
      for ob in self.obs:
        if not ob.is_valid_solution(pos_to_val):
          valid = False
          break

      if valid:
        for pos, val in pos_to_val.items():
          positions_values[pos].add(val)
        # print(f'valid solution: s={s}, pos_to_val={pos_to_val.items()}',
        # file=sys.stderr)
      now = datetime.now()
      waitTime = min((remainingOverageTime - 5), 3)

      if (now - start_time).total_seconds() > waitTime:
        # print(f" remainingOverageTime = {remainingOverageTime}",
        # file=sys.stderr)
        raise HiddenRelicSolverTimeout

      if remainingOverageTime < 10:
        raise HiddenRelicSolverTimeout

    # import time
    # time.sleep(5)
    solved_num = 0
    for pos, values in positions_values.items():
      if len(values) == 1:
        self.position_to_relic[pos] = bool(list(values)[0])
        solved_num += 1
      # print(f"pos={pos}, values={values}", file=sys.stderr)

    # print(
    # f"Solved position: {solved_num}; undetermined: {len(positions_values) - solved_num}",
    # file=sys.stderr)

  def observe(self, ob, remainingOverageTime):
    self.obs.append(ob)

    if remainingOverageTime > 10:
,     self.solve(remainingOverageTime)

    self.simplify_obs()
    if len(self.obs) > self.MAX_OB_NUM:
      self.obs = self.obs[-self.MAX_OB_NUM:]


class HiddenRelicSolverTimeout(Exception):
  pass


class HiddenRelicNodeEstimator:

  def __init__(self, mm, enable_anti_sym):
    self.mm = mm
    self.priori = np.zeros((MAP_WIDTH, MAP_HEIGHT))
    self.priori_ = np.zeros((MAP_WIDTH, MAP_HEIGHT))
    self.relic_node_positions = set()
    self.enable_anti_sym = enable_anti_sym
    self.solver = HiddenRelicSolver()

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

    # update (or reset) new relic node nb with priori
    new_relic_nb_mask = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    if new_relic_node_positions.size > 0:
      new_relic_nb_mask[new_relic_node_positions[:, 0],
                        new_relic_node_positions[:, 1]] = True

    # for newly found relic node, reset previous solved ones.
    if new_relic_nb_mask.sum():
      new_relic_nb_mask = maximum_filter(new_relic_nb_mask, size=RELIC_NB_SIZE)
      new_relic_nb_positions = np.argwhere(new_relic_nb_mask)
      self.solver.reset_with_relic_nb(new_relic_nb_positions)

    is_relic_nb = (is_relic_neighbour == 1)
    if new_team_points > 0:
      relic_positions = np.argwhere(is_relic_nb & unit_positions)
    else:
      relic_positions = np.argwhere(unit_positions)

    # Estimation method
    new_relic_nb_mask = maximum_filter(new_relic_nb_mask, size=RELIC_NB_SIZE)
    tmp_pri = self.priori_.copy()
    tmp_pri[new_relic_nb_mask] = PRIORI
    self.priori_ = np.maximum(self.priori_, tmp_pri)

    post = self.calc_posteriori_probs(relic_positions, new_team_points)
    self.priori_[is_relic_nb] = np.clip(post[is_relic_nb], MIN_PROB, MAX_PROB)

    ob = Observation(relic_positions, new_team_points)
    try:
      self.solver.observe(ob, self.mm.remainingOverageTime)

      # Use solver states when possible, and random guess for unsolved ones.
      self.priori = np.zeros(MAP_SHAPE2)
      relic_nb_positions = np.argwhere(is_relic_nb)
      for x, y in relic_nb_positions:
        pos = (int(x), int(y))
        is_relic = self.solver.position_to_relic.get(pos)

        p = 0
        if is_relic is None:
          # p = random.random() * 0.5 + 0.25
          p = self.priori_[pos[0]][pos[1]]
        else:
          p = 1.0 if is_relic else 0.0
        self.priori[pos[0]][pos[1]] = p
        # print(
        # f" self.priori[{pos[0]}][{pos[1]}]={self.priori[pos[0]][pos[1]]}",
        # file=sys.stderr)
    except HiddenRelicSolverTimeout:
      self.priori = self.priori_

      units_half = self.mm.get_player_half_mask(self.mm.player_id)

      p = self.priori_
      min_val = MIN_PROB + 1e-5
      non_relic_point = (p <= min_val) & (p > 0) & units_half
      for x, y in np.argwhere(non_relic_point):
        self.solver.position_to_relic[(int(x), int(y))] = False

      max_val = MAX_PROB - 1e-5
      is_relc_point = (p >= max_val) | anti_diag_sym(p >= max_val)
      for x, y in np.argwhere(non_relic_point):
        self.solver.position_to_relic[(int(x), int(y))] = True

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

    p_data = self.calculate_p_data(observed_cells, n, self.priori_)
    if p_data <= 0:
      return self.priori_
    assert p_data > 0
    # print(f'p_data = {p_data}')

    post = self.priori_.copy()
    for i in observed_cells:
      x, y = i
      ob_exclude = [c for c in observed_cells if not (c[0] == x and c[1] == y)]
      # print(observed_cells, ob_exclude, new_team_points)
      p_data_given_i = self.calculate_p_data(ob_exclude, n - 1, self.priori_)

      post_prob = p_data_given_i * self.priori_[x][y] / p_data
      post[x][y] = post_prob
      # post[y][x] = post_prob
      # print(
      # f"post[{x}][{y}] = {post[x][y]}, self.priori_[x][y]={self.priori_[x][y]}, p_data_given_i={p_data_given_i}"
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


class EnergyVoidFieldFactorEstimator:

  VALID_VALUES = [0.0625, 0.125, 0.25, 0.375]

  def __init__(self, mm):
    self.mm = mm
    self.passby_counter = 0
    self._counter = Counter()

  def best_guess(self):
    if len(self._counter) <= 0:
      return self.VALID_VALUES[1]  # default 0.125
    return self._counter.most_common(1)[0][0]

  def index(self):
    v = self.best_guess()
    return self.VALID_VALUES.index(v)

  def estimate(self):
    """Estimate energy void field factor based on known nebula energy reduction
    and ignore enemy sap.

    Note: Furthermore, each unit generates an "energy void" field around itself
    that affects all cardinally (up, right, down left) adjacent opposition units.
    """
    # at t=0, infer from t=1

    enemy_energy_sum = np.zeros(MAP_SHAPE2, dtype=int)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.enemy_id, i, t=0)
      if not mask or energy < 0:
        # print(
        # f'skip for enemy[{i}]={pos} because of mask0={mask}, energy0={energy}',
        # file=sys.stderr)
        continue

      mask1, pos1, energy1 = self.mm.get_unit_info(self.mm.enemy_id, i, t=1)
      if not mask1 or energy1 < 0:
        # print(
        # f'skip for enemy[{i}]={pos1} because of mask1={mask1}, energy1={energy1}',
        # file=sys.stderr)
        continue

      # Use position from current step, but energy from last step
      for k in range(4):
        nx, ny = (pos[0] + DIRECTIONS[k][0], pos[1] + DIRECTIONS[k][1])
        if not is_pos_on_map((nx, ny)):
          continue
        enemy_energy_sum[nx][ny] += energy1

    unit_pos_num = np.zeros(MAP_SHAPE2, dtype=int)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.player_id, i, t=0)
      if mask and energy > 0:
        unit_pos_num[pos[0]][pos[1]] += 1

    for i in range(MAX_UNIT_NUM):
      mask0, pos0, energy0 = self.mm.get_unit_info(self.mm.player_id, i, t=0)
      if not mask0 or energy0 < 0:
        # print(
        # f'skip for unit[{i}]={pos0} because of mask0={mask0}, energy0={energy0}',
        # file=sys.stderr)
        continue

      mask1, pos1, energy1 = self.mm.get_unit_info(self.mm.player_id, i, t=1)
      if not mask1 or energy1 < 0:
        # print(
        # f'skip for unit[{i}]={pos1} because of mask1={mask1}, energy1={energy1}',
        # file=sys.stderr)
        continue

      # print(
      # f"unit[{i}] mask0={mask0}, pos0={pos0} energy0={energy0}, mask1={mask1}, pos1={pos1} e1={energy1}",
      # file=sys.stderr)

      nearby_enemy_energy = enemy_energy_sum[pos0[0]][pos0[1]]
      if nearby_enemy_energy <= 0:
        # print(
        # f'skip for unit[{i}]={pos0} e={energy0} because of nearby_enemy_energy == 0',
        # file=sys.stderr)
        continue

      e = 0
      # Remove unit move cost
      move_cost = 0
      if not pos_equal(pos0, pos1):
        move_cost = -self.mm.unit_move_cost
      e += move_cost

      # Remove unit sap cost
      unit_sap = 0
      p = (int(pos0[0]), int(pos0[1]))
      if p in self.mm.last_sap_locations:
        unit_sap = -self.mm.unit_sap_cost
      e -= unit_sap

      # Add cell energy
      cell_energy = self.mm.cell_energy[pos0[0]][pos0[1]]
      e += cell_energy

      nebula = 0
      if self.mm.cell_type[pos0[0], pos0[1]] == CELL_NEBULA:
        nebula = -self.mm.nebula_energy_reduction
      e += nebula

      delta = energy0 - (energy1 + e)
      uc = unit_pos_num[pos0[0]][pos0[1]]
      v = self._add_guess(delta, nearby_enemy_energy, uc)
      # print((
      # f"s={self.mm.game_step} void_factor_test={v} use_sap={use_sap_cost}, delta={delta} unit[{i}] delta={delta} e[{pos1}]={energy1}, e[{pos0}]={energy0} unit_sap={unit_sap}, "
      # f"unit_move={move_cost}, cell_energy={cell_energy} nebula={nebula}, enemy_nearby={nearby_enemy_energy} "
      # f"uc={uc}, unit_sap_cost={self.mm.unit_sap_cost}"),
      # file=sys.stderr)
    # print(
    # f" -- energy void factor best_guess: {self.best_guess()}, counter: {self._counter}",
    # file=sys.stderr)

  def _add_guess(self, delta, nearby_enemy_energy, nc):
    delta = abs(delta)
    for v in self.VALID_VALUES:
      if delta * nc == int(nearby_enemy_energy * v):
        self._counter[v] += 1
    return None


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
    self.visible = np.zeros((MAP_WIDTH, MAP_HEIGHT), bool)
    self.match_observed = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    self.match_relic_hints = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
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
    self.enemy_vision_map = VisionMap(self.unit_sensor_range)
    self.sap_indexer = sap_indexer or SapIndexer()

    self.total_team_points = 0

    self.units_on_relic_num = 0
    self.prev_units_on_relic_num = 0

    self.units_frozen_count = 0
    self.prev_units_frozen_count = 0
    self.total_units_frozen_count = 0

    self.match_unit_sap_count = 0

    self.units_dead_count = 0
    self.prev_units_dead_count = 0
    self.total_units_dead_count = 0
    self.last_match_relic_cell_num = -1
    self.last_match_found_relic = True

    self.match_units_sap_dead_count = 0
    self.match_units_collision_dead_count = 0

    self.use_mirror = use_mirror
    self.use_hidden_relic_estimator = use_hidden_relic_estimator
    self.hidden_relic_estimator = HiddenRelicNodeEstimator(
        self, enable_anti_sym)

    self.energy_cost_map = None
    self.units_energy_cost_change = 0

    self.enable_anti_sym = enable_anti_sym
    self.full_params = full_params

    self.match_wins = 0

    self.sap_dropoff_factor_estimator = UnitSapDropoffFactorEstimator(self)
    self.nebula_drift_estimator = NebulaDriftEstimator()
    self.energy_node_estimator = EnergyNodeEstimator()

    self.has_reset_cell_type = False
    self.remainingOverageTime = 10

    if SAVE_ALL_STEPS_TP_PROB:
      self.team_point_probs = []

    self.last_sap_locations = set()
    self.enemy_sap_mask = None
    self.energy_void_field_factor_estimator = EnergyVoidFieldFactorEstimator(
        self)

    self.enemy_max_energy = np.zeros((MAP_SHAPE2), dtype=int)

  def get_valid_unit_mask(self, pid, t):
    ob = self.past_obs[t]
    unit_mask = ob['units_mask'][pid]
    unit_energy = ob['units']['energy'][pid]
    return unit_mask & (unit_energy >= 0)

  def step_unit_killed_num(self):
    m0 = self.get_valid_unit_mask(self.player_id, t=0)
    m1 = self.get_valid_unit_mask(self.player_id, t=1)
    return (m1 & ~m0).sum()

  def player_energy_sum(self, pid):
    energy_sum = np.zeros(MAP_SHAPE2, dtype=int)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.get_unit_info(pid, i, t=0)
      if not mask or energy < 0:
        continue
      for k in range(4):
        nx, ny = (pos[0] + DIRECTIONS[k][0], pos[1] + DIRECTIONS[k][1])
        if not is_pos_on_map((nx, ny)):
          continue
        energy_sum[nx][ny] += energy

    f = self.energy_void_field_factor_estimator.best_guess()
    return energy_sum.astype(float) * f

  def cell_net_energy(self, pos):
    e = self.cell_energy[pos[0]][pos[1]]
    if self.cell_type[pos[0]][pos[1]] == CELL_NEBULA:
      e -= self.nebula_energy_reduction
    return e

  def has_found_relic_in_match(self):
    return self.is_relic_node.sum() > self.last_match_relic_cell_num

  def add_enemy_sap_locations(self, enemy_sap_locations, mirror):
    self.enemy_sap_mask = np.zeros(MAP_SHAPE2, dtype=bool)
    for p in enemy_sap_locations:
      if mirror:
        p = anti_diag_sym_i(p)
      self.enemy_sap_mask[p[0]][p[1]] = True
    self.enemy_sap_mask = maximum_filter(self.enemy_sap_mask, size=3)
    # print(
    # f"pid={self.player_id} enemy_sap_mask.sum={self.enemy_sap_mask.sum()}")

  def add_sap_locations(self, sap_locations):
    self.last_sap_locations = {(int(p[0]), int(p[1])) for p in sap_locations}
    self.sap_dropoff_factor_estimator.estimate(sap_locations)
    self.match_unit_sap_count += len(sap_locations)

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

  def last_step_team_points(self, player_id, t):
    return self.past_obs[t]['team_points'][player_id]

  def update_cell_type(self, ob):
    # TODO: may not needed
    # if (not self.has_reset_cell_type and ob['steps'] > 82
    # and self.nebula_drift_estimator.drift_speed is not None):
    # # clear map cell to clear accumuated cell type error
    # self.cell_type[:, :] = 0

    # adding 1 to start cell type from 0
    cells = ob['map_features']['tile_type'] + 1
    change_step = ob['steps'] - 1
    if len(self.past_obs) and self.game_step < MAX_MATCH_STEPS:
      last_cells = self.past_obs[0]['map_features']['tile_type'] + 1
      self.nebula_drift_estimator.update(cells, last_cells, change_step)

    drift_speed = self.nebula_drift_estimator.drift_speed
    if drift_speed is not None and is_drifted_step(change_step, drift_speed):
      # print(
      # f'step={self.game_step}, apply nebula drift = {drift_speed}, {self.nebula_drift_estimator.counter}',
      # file=sys.stderr)
      self.cell_type = shift_map_by_sign(self.cell_type, drift_speed)

    # Update map cell type
    c = cells > CELL_UNKONWN
    self.cell_type[c] = cells[c]

    # also update the symmetrical positions
    if self.enable_anti_sym:
      cells_sym = anti_diag_sym(cells)
      ct = cells_sym > CELL_UNKONWN
      self.cell_type[ct] = cells_sym[ct]

  @property
  def exploration_required(self):
    relic_not_found = (self.is_relic_node.sum() == 0)
    relic_spawn_cond = (self.game_step <= 303 and self.last_match_found_relic
                        and not self.has_found_relic_in_match())
    return relic_not_found or relic_spawn_cond

  def update_visible_and_observed(self, ob):
    self.visible = ob['sensor_mask'].astype(bool)
    self.prev_match_observed = self.match_observed.copy()
    self.prev_game_observed = self.game_observed.copy()
    self.match_observed |= self.visible
    self.game_observed |= self.visible

    # set last observed time
    self.last_observed_step[self.visible] = self.game_step
    self.last_observed_step[anti_diag_sym(self.visible)] = self.game_step

    # Activate relic hint layer if no relic node found in first 50 matches
    if self.match_step == 50 and self.exploration_required:
      self.match_relic_hints[:, :] = 1
      # print(f'relic hint activated', file=sys.stderr)

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
    self.unit_sapped_last_step = np.zeros(MAX_UNIT_NUM, dtype=bool)
    if (model_action is None
        or len(self.past_obs) == 0) or ob['match_steps'] <= 1:
      return

    units_action = model_action[UNITS_ACTION]

    def make_action(p, e, action, unit_id):
      if a == ACTION_CENTER:
        return p, e

      # TODO: check move target
      if a in MOVE_ACTIONS_NO_CENTER:
        if e >= self.unit_move_cost:
          e -= self.unit_move_cost
          tmp = unit_move(p, a)
          if (0 <= tmp[0] < MAP_WIDTH and 0 <= tmp[1] < MAP_HEIGHT
              and self.cell_type[tmp[0], tmp[1]] != CELL_ASTERIOD):
            p = tmp

        e += self.cell_energy[p[0]][p[1]]
        e = max(e, 0)
        return p, e

      assert a >= ACTION_SAP
      if e >= self.unit_sap_cost:
        e -= self.unit_sap_cost
        self.unit_sapped_last_step[unit_id] = True
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
      p1, e1 = make_action(p0, e0, a, i)
      if (m0 and mask and e0 > 0 and energy > 0 and e1 > 0
          and self.cell_type[position[0], position[1]] == CELL_NEBULA):
        reduction = e1 - energy
        self._nebula_energy_reduction.add(reduction)
        # print(
        # f'gstep={ob["steps"]}, mstep={ob["match_steps"]}, nebula_energy_reduction={self._nebula_energy_reduction.best_guess()}',
        # file=sys.stderr)
        # print(f' [debug] unit={i}, {p0}=>{position} e=({e0}={energy}) a={a}',
        # file=sys.stderr)

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

    sap_death_num = 0

    n_units = 0
    for i in range(MAX_UNIT_NUM):
      is_dead, is_frozen = False, False
      mask, p0, e0 = self.get_unit_info(self.player_id, i, t=0)
      mask1, p1, e1 = self.get_unit_info(self.player_id, i, t=1)
      if mask1 and e0 < 0 and e1 >= 0:
        is_dead = True
        is_sap = False

        # use position1, since e0 < 0, it's position is (-1, -1)
        if (self.enemy_sap_mask is not None
            and self.enemy_sap_mask[p1[0]][p1[1]]):
          sap_death_num += 1
          is_sap = True
        # print(
        # f">>>>  step={self.game_step} dead={is_dead} is_sap={int(is_sap)}, p1={p1}, p0={p0}"
        # )

      if mask and mask1:
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
      # f'gstep={self.game_step}, mstep={self.match_step} pid={self.player_id}, unit[{i}] p0={p0}, e0={e0} m0={mask} to p1={p1} e1={e1} m1={mask1} is_dead={is_dead}, is_frozen={is_frozen}',
      # file=sys.stderr)

      self.units_frozen_count += int(is_frozen)
      self.units_dead_count += int(is_dead)

    self.total_units_dead_count += self.units_dead_count
    self.total_units_frozen_count += self.units_frozen_count

    self.match_units_sap_dead_count += sap_death_num
    self.match_units_collision_dead_count += (self.units_dead_count -
                                              sap_death_num)

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
      self.blindshot_enemy_trace = np.zeros(MAP_SHAPE2, dtype=float)
      self.enemy_energy_trace = np.zeros(MAP_SHAPE2, dtype=float)

      self.prev_team_point = 0
      # self.past_obs.clear()
      self.match_units_sap_dead_count = 0
      self.match_units_collision_dead_count = 0

      self.total_units_dead_count = 0
      self.total_units_frozen_count = 0
      self.match_visited[:, :] = 0
      self.match_observed[:, :] = 0
      self.match_relic_hints[:, :] = 0
      self.prev_units_on_relic_num = self.units_on_relic_num = 0
      self.prev_units_dead_count = self.units_dead_count = 0
      self.prev_units_frozen_count = self.units_frozen_count = 0

      self.last_match_found_relic = self.has_found_relic_in_match()
      self.last_match_relic_cell_num = self.is_relic_node.sum()

      self.hidden_relic_estimator.solver.obs = []

      self.match_unit_sap_count = 0

      # self.append_ob(ob)

      # use match_step=0 reset map manager, do not update below
      # return

    # use non-infered units position
    self.vision_map.update(ob['units_mask'][self.player_id],
                           ob['units']['position'][self.player_id],
                           ob['units']['energy'][self.player_id])
    self.enemy_vision_map.update(ob['units_mask'][self.enemy_id],
                                 ob['units']['position'][self.enemy_id],
                                 ob['units']['energy'][self.enemy_id])

    self.game_step = ob['steps']
    self.match_step = ob['match_steps']
    self.update_counters()

    # Update map info first
    self.update_visible_and_observed(ob)
    self.update_cell_type(ob)
    self.update_cell_energy(ob)

    self.infer_nebula_energy_reduction(ob, model_action)

    unit_positions = self.get_unit_positions(ob, self.player_id)
    self.update_visited_node(unit_positions, ob)

    self.update_relic_node(ob)

    if not self.use_hidden_relic_estimator:
      self.update_team_point_mass(ob, unit_positions)

    self.append_ob(ob)

    self.update_frozen_or_dead_units()
    self.update_vision_map()

    self.update_blind_shot_trace()
    self.update_sap_position_by_enemy_position()

    if self.use_hidden_relic_estimator and ob['match_steps'] > 0:
      self.update_hidden_relic_estimator(ob)
      if SAVE_ALL_STEPS_TP_PROB:
        self.team_point_probs.append(self.hidden_relic_estimator.priori.copy())
        print(f'xxxxxxxxxxx  {len(self.team_point_probs)}')

    self.energy_cost_map = self.compute_energy_cost_map(
        self.cell_type, self.cell_energy, self.is_relic_node,
        self.nebula_energy_reduction)
    self.energy_void_field_factor_estimator.estimate()
    self.prev_team_point = ob['team_points'][self.player_id]

  def update_sap_position_by_enemy_position(self):
    """Enemy position for sap action"""
    self.enemy_position_mask = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
    self.enemy_position_mask_can_negtive = np.zeros((MAP_WIDTH, MAP_HEIGHT),
                                                    dtype=bool)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.get_unit_info(self.enemy_id, i, t=0)
      if mask and energy >= 0:
        self.enemy_position_mask[pos[0], pos[1]] = True

      if mask:
        ob = self.past_obs[0]
        pos = ob['units']['position'][self.enemy_id][i]
        self.enemy_position_mask_can_negtive[pos[0], pos[1]] = True

    self.enemy_position_mask = maximum_filter(self.enemy_position_mask, size=3)

    self.enemy_max_energy = np.zeros((MAP_SHAPE2), dtype=int)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.get_unit_info(self.enemy_id, i, t=0)
      if mask:
        self.enemy_max_energy[pos[0]][pos[1]] = max(
            self.enemy_max_energy[pos[0]][pos[1]], energy)

    self.unit_count = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=int)
    self.unit_min_energy = np.ones(
        (MAP_WIDTH, MAP_HEIGHT), dtype=int) * MAX_UNIT_ENERGY
    self.min_energy_unit_id = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=int)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.get_unit_info(self.player_id, i, t=0)
      if mask and energy >= 0:
        self.unit_count[pos[0]][pos[1]] += 1
        if energy < self.unit_min_energy[pos[0]][pos[1]]:
          self.unit_min_energy[pos[0]][pos[1]] = energy
          self.min_energy_unit_id[pos[0]][pos[1]] = i

    # Update enemy energy trace
    self.enemy_energy_trace *= TRACE_DECAY
    for i in range(MAX_UNIT_NUM):
      mask1, pos1, energy1 = self.get_unit_info(self.enemy_id, i, t=1)
      if (mask1 and energy1 >= 0 and not self.visible[pos1[0]][pos1[1]]):
        self.enemy_energy_trace[pos1[0]][pos1[1]] += energy1
    self.enemy_energy_trace[self.visible] = 0

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
    if not mask:
      position = (0, 0) if self.player_id == 0 else (MAP_WIDTH - 1,
                                                     MAP_HEIGHT - 1)

    # do not use previous step info for match reset
    if t > 0 and self.match_step < ob['match_steps']:
      mask = False
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

  def get_unit_positions(self, ob, pid, drop_died=True):
    unit_masks = ob['units_mask'][pid]
    unit_energy = ob['units']['energy'][pid]

    mask = unit_masks
    if drop_died:
      mask &= (unit_energy >= 0)

    unit_positions = ob['units']['position'][pid][mask]
    return unit_positions

  def update_visited_node(self, unit_positions, ob):
    self.unit_positions = np.zeros((MAP_SHAPE2), dtype=bool)
    self.unit_positions[unit_positions[:, 0], unit_positions[:, 1]] = True

    self.match_visited[self.unit_positions] = 1
    self.game_visited |= self.match_visited
    self.last_visited_step[self.unit_positions] = self.game_step

    self.enemy_positions = np.zeros((MAP_SHAPE2), dtype=bool)
    enemy_positions = self.get_unit_positions(ob, self.enemy_id)
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

    if self.match_step >= 50 and self.game_step <= 303:
      # From 50 match steps onwards, if relic node found, reset relic hint layers
      if self.has_found_relic_in_match():
        self.match_relic_hints[:, :] = 0
      else:
        match_observed = self.visible | anti_diag_sym(self.visible)
        self.match_relic_hints[match_observed] = 0
        # print(f" s={self.game_step}, hints={self.match_relic_hints.sum() / 2}",
        # file=sys.stderr)

  def update_cell_energy(self, ob):
    energy = ob['map_features']['energy']
    sensor_mask = ob['sensor_mask']

    change_step = ob['steps'] - 1
    if len(self.past_obs) > 0:
      last_energy = self.past_obs[0]['map_features']['energy']
      last_visible = self.past_obs[0]['sensor_mask']
      self.energy_node_estimator.update_energy(energy, sensor_mask,
                                               last_energy, last_visible,
                                               change_step)

    if self.energy_node_estimator.energy_node_found:
      self.cell_energy = self.energy_node_estimator.current_energy_field
      # print(
      # f"--> using energy node at {self.energy_node_estimator.energy_node}",
      # file=sys.stderr)

    # In case energy node is not found, always update energy field map
    self.cell_energy[sensor_mask] = energy[sensor_mask]
    if self.enable_anti_sym:
      energy_tr = anti_diag_sym(energy)
      is_visible_tr = anti_diag_sym(sensor_mask)
      self.cell_energy[is_visible_tr] = energy_tr[is_visible_tr]

  @lru_cache(maxsize=None)
  def get_player_half_mask(self, player_id):
    init_pos = get_player_init_pos(player_id, self.use_mirror)
    return generate_manhattan_mask(MAP_SHAPE2,
                                   init_pos,
                                   range_limit=MAP_WIDTH - 1)

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

      hit_map[pos[0]][pos[1]] = True
      for k in range(4):
        next_pos = (pos[0] + DIRECTIONS[k][0], pos[1] + DIRECTIONS[k][1])
        if not is_pos_on_map(next_pos):
          continue
        hit_map[next_pos[0]][next_pos[1]] = True

    # Add enemy last step position
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.get_unit_info(self.enemy_id, i, t=1)
      if not mask or energy < 0 or self.visible[pos[0]][pos[1]]:
        continue

      # Only add one position for last step enemy position
      hit_map[pos[0]][pos[1]] = True

    # Add unvisible team point positions
    init_pos = get_player_init_pos(self.enemy_id, self.use_mirror)
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
    extra_step_cost = 5
    cost_map += extra_step_cost
    cost_map -= cell_energy
    cost_map = np.maximum(cost_map, 1)

    # use a big value for asteriod
    cost_map[cell_type == CELL_ASTERIOD] += 25 * 20

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

  def to_last_sap_actions(self, model_action, action_taken_mask=None):
    sap_locations = []

    actions = model_action[UNITS_ACTION]
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.get_unit_info(self.player_id, i, t=0)
      if not mask or energy <= 0:
        continue

      a = actions[i][0]
      if isinstance(a, torch.Tensor):
        a = a.item()

      if action_taken_mask is not None and not action_taken_mask[i][a]:
        # print(f"skip not taken action by action_taken_mask", file=sys.stderr)
        continue

      # print(f"type(a)={type(a)}, is_tensor={isinstance(a, torch.tensor)}")

      if a < MOVE_ACTION_NUM:
        continue

      if energy < self.unit_sap_cost:
        continue

      a -= MOVE_ACTION_NUM
      x, y = self.sap_indexer.idx_to_position[a]

      sap_loc = (pos[0] + x, pos[1] + y)
      if not is_pos_on_map(sap_loc):
        print(f"pos={pos} x={x}, y={y} sap_loc={sap_loc}", file=sys.stderr)
      # __import__('ipdb').set_trace()
      sap_locations.append(sap_loc)
    return sap_locations

  def update_blind_shot_trace(self):
    candidate_loc = np.zeros(MAP_SHAPE2, dtype=float)
    self.unit_energy_diff = np.zeros(MAP_SHAPE2, dtype=float)

    # If there is no enemy in visible area (from enemy energy from last step)
    if self.enemy_max_energy.sum() == 0:
      for i in range(MAX_UNIT_NUM):
        m1, p1, e1 = self.get_unit_info(self.player_id, i, t=1)
        if not m1 or e1 < 0:
          continue

        # If unit was valid in last step but not current step, it's dead
        m0, p0, e0 = self.get_unit_info(self.player_id, i, t=0)
        if not m0:
          e0 = 0

        self.unit_energy_diff[p1[0]][p1[1]] += (e0 - e1)

        pos = p1
        if m0 and e0 >= 0:
          pos = p0

        # unit on relic point
        is_unit_on_relic = False
        if self.team_point_mass[pos[0]][pos[1]] > 0.8:
          is_unit_on_relic = True

        cell_energy = self.cell_energy[p0[0]][p0[1]]

        nebula_energy = 0
        if self.cell_type[p0[0]][p0[1]] == CELL_NEBULA:
          nebula_energy = self.nebula_energy_reduction

        move_cost = 0
        if not pos_equal(p0, p1):
          move_cost = self.unit_move_cost

        sap_cost = 0
        if self.unit_sapped_last_step[i]:
          sap_cost = self.unit_sap_cost

        action_cost = (move_cost + sap_cost)
        energy_gain = cell_energy - nebula_energy
        e1 += (-action_cost + energy_gain)

        is_unit_sapped = False
        if e1 - e0 >= self.unit_sap_cost:
          is_unit_sapped = True
          # print(
          # f"step={self.game_step}, e1-e0={e1-e0} m1={m1}, p1={p1}, e1={e1}, m0={m0}, p0={p0}, e0={e0} "
          # f"move_cost={move_cost}, sap_cost={sap_cost} cell_energy={cell_energy} nebula_energy={nebula_energy} "
          # )

        # enemy blind shot at unit
        if is_unit_on_relic and is_unit_sapped:
          sap_range = gen_sap_range(pos, self.unit_sap_range)

          enemy_init_pos = get_player_init_pos(self.enemy_id, self.use_mirror)
          dist = manhatten_distance(pos, enemy_init_pos)
          x, y = np.ogrid[:MAP_WIDTH, :MAP_HEIGHT]
          enemy_relic_dist = (np.abs(x - enemy_init_pos[0]) +
                              np.abs(y - enemy_init_pos[1]))
          enemy_candidate_range = (enemy_relic_dist < (dist + 1)) & sap_range

          candidate_loc[enemy_candidate_range] = 1

    # First clear the visible area
    self.blindshot_enemy_trace *= TRACE_DECAY
    self.blindshot_enemy_trace[candidate_loc > 0] = 1
    self.blindshot_enemy_trace[self.visible] = 0

    # if self.blindshot_enemy_trace.sum() > 0:
    # print(
    # f's={self.game_step}, trace.sum={self.blindshot_enemy_trace.sum()} cnt={(self.blindshot_enemy_trace > 0).sum()}'
    # )

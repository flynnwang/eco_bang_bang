from typing import Callable, Dict, Optional, Tuple, Union, NamedTuple, Any, List
from argparse import Namespace
import os
import sys
import functools

import numpy as np
import torch
import scipy.optimize
from scipy.ndimage import minimum_filter, maximum_filter

# hello

from .env.const import *
from .env.mapmanager import (
    MapManager,
    SapIndexer,
    anti_diag_sym,
    anti_diag_sym_i,
    EXT_ACTION_SHAPE,
    manhatten_distance,
    is_pos_on_map,
    pos_equal,
    generate_manhattan_mask,
    generate_manhattan_dist,
    gen_sap_range,
    is_drifted_step,
    shift_map_by_sign,
)

# SUBMIT_AGENT = False
SUBMIT_AGENT = True

DO_SAMPLE = True
USE_MIRROR_TRANS = False

USE_RANDOM = True

DEVICE = 'cpu'
if not SUBMIT_AGENT:
  # import random
  # DEVICE = random.sample(['cuda:0', 'cuda:1'], k=1)[0]
  DEVICE = 'cuda:0'

N_CELLS = MAP_WIDTH * MAP_HEIGHT

LOG3 = np.log(4)


@functools.lru_cache(maxsize=1024, typed=False)
def dd(dist, r=1.1):
  dist = min(dist, MAP_WIDTH * 2)
  return r**dist


def cell_idx_to_pos(idx):
  return int(idx % MAP_WIDTH), int(idx // MAP_WIDTH)


def can_attack(energy, mm, margin=3):
  return energy >= mm.unit_sap_cost + mm.unit_move_cost * margin


def cant_move_to(upos, cpos, mm):
  return (mm.cell_type[cpos[0]][cpos[1]] == CELL_ASTERIOD
          and not pos_equal(cpos, upos))


def is_within_sap_range(upos, cpos, unit_sap_range):
  return ((abs(upos[0] - cpos[0]) <= unit_sap_range)
          and (abs(upos[1] - cpos[1]) <= unit_sap_range))


def get_player_init_pos(player_id):
  target_pos = (0, 0)
  if player_id == 1:
    target_pos = (23, 23)
  return target_pos


def on_enemy_side(cpos, player_id):
  target_pos = get_player_init_pos(player_id)
  mdist = manhatten_distance(target_pos, cpos)
  return mdist > MAP_WIDTH


def right_tailed_exp(energy, val, m, v=20):
  if energy >= m:
    alpha = np.exp(-(energy - m)**2 / (v**2 * 2.0))
    val *= alpha
  return val


def left_tailed_exp(energy, val, m, v=20):
  if energy < m:
    alpha = np.exp(-(energy - m)**2 / (v**2 * 2.0))
    val *= alpha
  return val


RELIC_SCORE = 30
RELIC_NB_SCORE = 15

EXPLORE_CELL_SCORE = 3
MAX_EXPLORE_SCORE = 10

MIN_OPEN_RELIC_NB_PROB = 0.01
IS_RELIC_CELL_PROB = 0.8

BOOST_SAP_ENERGY_THRESHOOD = 180


class Agent:

  def __init__(self, player: str, env_cfg) -> None:
    self.player = player
    self.env_cfg = env_cfg
    # np.random.seed(0)

    obs_space_kwargs = {
        'use_energy_cost_map': True,
        'use_single_player': False
    }

    self.mm = MapManager(player,
                         env_cfg,
                         transpose=False,
                         sap_indexer=SapIndexer(),
                         use_mirror=False,
                         use_hidden_relic_estimator=True)
    self.prev_model_action = None
    self.last_sap_locations = []

  def get_enemy_sap_range(self):
    # TODO: exclude the position enemy can not see
    enemy_sap_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=float)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.enemy_id, i, t=0)
      if not mask or energy < self.mm.unit_sap_cost:
        continue
      sap_range = gen_sap_range(pos,
                                self.mm.unit_sap_range + 1)  # add 1 for safety
      enemy_sap_map[sap_range] = True
    return enemy_sap_map

  def get_sap_hit_map(self, factor):

    def predict_next_move(enemy_unit_id, pos0, unit_energy):
      mask1, pos1, _ = self.mm.get_unit_info(self.mm.enemy_id,
                                             enemy_unit_id,
                                             t=1)
      if not mask1 or pos_equal(pos0, pos1):
        return pos0

      # Assume units will stay on relic
      if self.mm.team_point_mass[pos0[0]][pos0[1]] > IS_RELIC_CELL_PROB:
        return pos0

      # Move is the first thing in resolusion order
      pos = pos0
      if unit_energy >= self.mm.unit_move_cost:
        dx = pos0[0] - pos1[0]
        dy = pos0[1] - pos1[1]
        tmp_pos = (pos0[0] + dx, pos0[1] + dy)
        if is_pos_on_map(tmp_pos):
          pos = tmp_pos

      # print(
      # f"$$$$$$$$$$$$$$ predict enemy move from pos[t-1]={pos1}, pos[t]={pos0} to pos[t+1]={pos}",
      # file=sys.stderr)
      return pos

    hit_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=float)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.enemy_id, i, t=0)
      if not mask or energy < 0:
        continue

      # For on-relic enemy, add extra score
      if self.mm.enemy_positions[pos[0]][pos[1]]:
        p = self.mm.team_point_mass[pos[0]][pos[1]]
        if p > IS_RELIC_CELL_PROB:
          hit_map[pos[0]][pos[1]] += RELIC_SCORE

      # For positions around current position, add cost with dropoff
      # (intended not using surrandings from predicted move)
      for d in [1, 0]:
        x0 = max(0, (pos[0] - d))
        x1 = min(MAP_WIDTH, (pos[0] + d + 1))
        y0 = max(0, (pos[1] - d))
        y1 = min(MAP_HEIGHT, (pos[1] + d + 1))

        h = self.mm.unit_sap_cost
        if d == 1:
          h *= self.mm.unit_sap_dropoff_factor
        hit_map[x0:x1, y0:y1] += h

      # For predicted next move, set hit_map to unit_sap_cost
      # next_pos = pos
      # if self.mm.unit_sap_dropoff_factor < 1:
      # next_pos = predict_next_move(i, pos, energy)
      # hit_map[next_pos[0]][next_pos[1]] += self.mm.unit_sap_cost

      # TODO: Keep the current posotion
      # hit_map[pos[0]][pos[1]] = self.mm.unit_sap_cost

      # slightly favour cell for enemy next move
      x, y = np.ogrid[:MAP_WIDTH, :MAP_HEIGHT]
      enemy_dist = np.abs(x - pos[0]) + np.abs(y - pos[1])
      init_pos = get_player_init_pos(self.mm.player_id)
      enemy_init_pos_dist = manhatten_distance(pos, init_pos)
      init_pos_dist = np.abs(x - init_pos[0]) + np.abs(y - init_pos[0])
      mask = (enemy_dist == 1) & (init_pos_dist < enemy_init_pos_dist)
      hit_map[mask] += 1

    return hit_map

  def gen_fire_zone(self, extend_dist=2, backoff_steps=5):
    """Fire zone is the positive energy cells that could either attack enemy
    relic position or protect my reilc points."""
    mm = self.mm
    team_point_mask = (mm.team_point_mass > IS_RELIC_CELL_PROB)
    fire_zone_range = mm.unit_sap_range * 2 + 1 + extend_dist
    fire_zone = maximum_filter(team_point_mask, fire_zone_range)

    # energy_lost_step = self.mm.sap_dropoff_factor_estimator.unit_energy_lost_step
    # energy_lost_mask = (mm.game_step - energy_lost_step) <= backoff_steps
    # energy_lost_mask = maximum_filter(energy_lost_mask, 3)

    # return fire_zone, energy_lost_mask
    return fire_zone

  def compute_unit_to_cell(self):
    mm = self.mm
    is_explore_step = (mm.match_step <= 50 and mm.game_step < 303)

    match_observed = mm.match_observed + anti_diag_sym(mm.match_observed)
    energy_threshold = 60 + mm.match_step

    # TODO: test drop it
    if mm.match_step >= 70:
      energy_threshold = 60

    has_found_relic = mm.has_found_relic_in_match()
    n = (MAP_WIDTH * MAP_HEIGHT)
    n_explore = n - match_observed.sum()
    expore_score = n * EXPLORE_CELL_SCORE / (n_explore + 1)
    print(
        f' +++ step={mm.game_step} to-exp-cell-num={n_explore} exp_score={expore_score} has_found_relic={has_found_relic}, last_match_found_relic={mm.last_match_found_relic}',
        file=sys.stderr)

    def get_explore_weight(upos, energy, cpos):
      if mm.game_step >= (3 * MAX_MATCH_STEPS):
        return 0

      if not mm.last_match_found_relic:
        return 0

      if has_found_relic:
        return 0

      if match_observed[cpos[0]][cpos[1]]:
        return 0

      return min(expore_score, MAX_EXPLORE_SCORE)

    energy_map = mm.cell_energy.copy()
    energy_map[mm.cell_energy != CELL_UNKONWN] -= mm.unit_move_cost
    energy_map[mm.cell_type == CELL_NEBULA] -= mm.nebula_energy_reduction
    if not SUBMIT_AGENT:
      print(
          f'>>>>>>>>>>>>>>> nebula_energy_reduction={mm.nebula_energy_reduction}',
          file=sys.stderr)

    fire_zone = self.gen_fire_zone()

    player_init_pos = get_player_init_pos(mm.player_id)
    d1 = generate_manhattan_dist(MAP_SHAPE2,
                                 player_init_pos).astype(np.float32)
    # d1[d1 > MAP_WIDTH] = MAP_WIDTH
    d1 /= MAP_WIDTH

    def get_fuel_energy(upos, energy, cpos):
      e = fuel = energy_map[cpos[0]][cpos[1]]
      fuel = right_tailed_exp(energy, fuel, energy_threshold)

      # Boost more net energy position without energy thresholding
      if e > 0 and fire_zone[cpos[0]][cpos[1]]:
        fuel += (e * d1[cpos[0]][cpos[1]])

      return fuel

    enemy_sap_range = self.get_enemy_sap_range()

    def get_open_relic_nb(upos, energy, cpos):
      """First visit on relic neighbour"""
      if not mm.is_relic_neighbour[cpos[0]][cpos[1]]:
        return 0

      # if not mm.match_visited[cpos[0]][cpos[1]]:
      # return RELIC_NB_SCORE

      p = mm.team_point_mass[cpos[0]][cpos[1]]
      if p < MIN_OPEN_RELIC_NB_PROB:
        return 0

      v = RELIC_NB_SCORE
      # Lower relic nb on enemy side
      if on_enemy_side(cpos, mm.player_id):
        # v = mm.unit_sap_cost / 10 * p
        v = mm.unit_sap_cost / 10

      # If enemy may sap it, lower its weight
      if enemy_sap_range[cpos[0]][cpos[1]]:
        v = mm.unit_sap_cost / 10 * 0.5

      last_visited_step = mm.last_visited_step[cpos[0]][cpos[1]]
      t = mm.game_step - last_visited_step
      alpha = np.log(t + 1) / LOG3
      w = min(alpha, 1) * v

      # cpos_nb_mask = gen_sap_range(cpos, self.mm.unit_sap_range)
      # if (mm.enemy_max_energy[cpos_nb_mask] > energy).sum() > 0:
      # w = -1
      return w

    init_pos = get_player_init_pos(mm.enemy_id)
    enemy_half = generate_manhattan_mask(MAP_SHAPE2,
                                         init_pos,
                                         range_limit=MAP_WIDTH - 1)
    blind_shot_targets = ((~mm.visible) &
                          (mm.team_point_mass > IS_RELIC_CELL_PROB)
                          & enemy_half)
    # blind_shot_targets = np.zeros(MAP_SHAPE2, dtype=bool)  # disable blind shot
    self.blind_shot_targets = blind_shot_targets

    def stay_on_relic(upos, energy, cpos):
      # If the relic node has been occupied by unit but not this one, lower its score
      if (mm.unit_positions[cpos[0]][cpos[1]] and not pos_equal(upos, cpos)):
        return 0

      # # Relic unit do not change relic position
      # p_unit = mm.team_point_mass[upos[0]][upos[1]]
      # if (p_unit > IS_RELIC_CELL_PROB and not pos_equal(upos, cpos)):
      # return 0

      # Will score only for relic p > 0.8
      v = RELIC_SCORE
      p = mm.team_point_mass[cpos[0]][cpos[1]]

      # Do not goto enemy side if energy below threshold
      if on_enemy_side(cpos, mm.player_id):
        v = mm.unit_sap_cost / 10

      # If enemy may sap it, lower its weight
      if not pos_equal(cpos, upos) and enemy_sap_range[cpos[0]][cpos[1]]:
        v = mm.unit_sap_cost / 10 * 0.5

      w = 0
      if p > IS_RELIC_CELL_PROB:
        w += v * p

      return w

    hit_factor = 10
    enemy_hit_map = self.get_sap_hit_map(hit_factor)
    enemy_hit_map[
        blind_shot_targets] += mm.unit_sap_cost  # roughly adding 3 points
    self.enemy_hit_map = enemy_hit_map

    def get_sap_enemy_score(upos, energy, cpos):
      """Max sap damage that could be hit from the `cpos`."""
      if not can_attack(energy, mm):
        return 0

      # Do not attack from negtive energy position
      # fuel = energy_map[cpos[0]][cpos[1]]
      # if fuel <= 0:
      # return 0

      sap_range = gen_sap_range(cpos, self.mm.unit_sap_range)

      h = enemy_hit_map[sap_range].max()
      h /= hit_factor

      # Boost unit with extra energy for SAP
      h *= min(energy / BOOST_SAP_ENERGY_THRESHOOD, 1)

      # sap if energy is large (and unit not on relic)
      # if self.mm.team_point_mass[pos[0]][pos[1]] < 0.6:
      # h *= max((energy / energy_threshold), 1)
      return h

    score_debug = {}

    def get_unit_cell_wt(upos, energy, cpos, unit_cost_map):
      if cant_move_to(upos, cpos, mm):
        return -9999

      # if energy < unit_cost_map[cpos[0]][cpos[1]]:
      # if not SUBMIT_AGENT:
      # print(f'game_step={mm.game_step}: skip due to inf at {cpos}',
      # file=sys.stderr)
      # return -9999

      # Do not target cell with enemy energy > unit energy
      if energy < mm.enemy_max_energy[cpos[0]][cpos[1]]:
        return -9999

      # mdist = manhatten_distance(upos, cpos) + 7
      mdist = dd(manhatten_distance(upos, cpos) + 1)
      wt = 0.0001

      energy_ratio = energy / energy_threshold

      expore_wt = get_explore_weight(upos, energy, cpos)

      fuel_wt = get_fuel_energy(upos, energy, cpos)

      relic_nb_wt = get_open_relic_nb(upos, energy, cpos)

      on_relic_wt = stay_on_relic(upos, energy, cpos)

      sap_wt = get_sap_enemy_score(upos, energy, cpos)

      # If enemy hit me at relic position, skip them?
      cpos_nb_mask = gen_sap_range(cpos, self.mm.unit_sap_range + 1)
      # any_enemy_nearby = mm.enemy_position_mask_can_negtive[cpos_nb_mask].sum(
      # ) > 0
      # if not any_enemy_nearby and energy_lost_mask[cpos[0]][cpos[1]]:
      # on_relic_wt *= 0.2
      # relic_nb_wt *= 0.2

      wt += (expore_wt + fuel_wt + relic_nb_wt + on_relic_wt + sap_wt) / mdist

      is_relic_nb = mm.is_relic_neighbour[cpos[0]][cpos[1]]
      # has enemy nearby, dangerous, go away
      has_enemy_nearby = (mm.enemy_max_energy[cpos_nb_mask] > energy).sum() > 0
      if (has_enemy_nearby
          and (not is_relic_nb or on_enemy_side(cpos, mm.player_id))):
        wt -= self.mm.unit_sap_cost / 10

      score_debug[(tuple(upos), tuple(cpos))] = {
          'explore_wt': expore_wt,
          'fuel_wt': fuel_wt,
          'relic_nb_wt': relic_nb_wt,
          'on_relic_wt': on_relic_wt,
          'sap_wt': sap_wt,
          'energy_ratio': energy_ratio,
          'wt': wt,
          'mdist': mdist,
      }

      # if USE_RANDOM:
      # wt += np.random.rand() / 1000
      return wt

    weights = np.ones((MAX_UNIT_NUM, N_CELLS)) * -9999
    cell_index = list(range(N_CELLS))
    if USE_RANDOM:
      np.random.shuffle(cell_index)

    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.player_id, i, t=0)
      if not mask:
        continue

      for j in cell_index:
        target_cell_pos = cell_idx_to_pos(j)
        weights[i, j] = get_unit_cell_wt(pos,
                                         energy,
                                         target_cell_pos,
                                         unit_cost_map=None)

    unit_to_cell = {}
    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for unit_id, target_id in zip(rows, cols):
      wt = weights[unit_id, target_id]
      if wt < 1e-6:  # TODO: use 0?
        # if wt < -1:  # TODO: use 0?
        continue
      cpos = cell_idx_to_pos(target_id)
      unit_to_cell[unit_id] = cpos
      if not SUBMIT_AGENT:
        _, upos, _ = self.mm.get_unit_info(self.mm.player_id, unit_id, t=0)
        wts = score_debug[(tuple(upos), tuple(cpos))]
        print(f" unit[{unit_id}]={upos} assgined to cell={cpos}, wts={wts}",
              file=sys.stderr)

    return unit_to_cell

  def compute_energy_cost_map(self,
                              target_pos,
                              asteriod_cost=400,
                              N=MAP_WIDTH * 2,
                              extra_step_cost=10):
    """Using `extra_step_cost` to control the balance between cost and path length."""
    mm = self.mm

    static_cost_map = np.full((MAP_WIDTH, MAP_HEIGHT),
                              float(mm.unit_move_cost))
    static_cost_map -= mm.cell_energy

    cell_type = mm.cell_type

    energy_cost = np.full((MAP_WIDTH, MAP_HEIGHT), np.inf, dtype=np.float64)
    energy_cost[target_pos[0]][target_pos[1]] = 0

    is_map_drifted = True
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    for n in range(N):
      if is_map_drifted:
        cost_map = static_cost_map.copy()
        # nebula energy reduction adds extra cost
        cost_map[cell_type == CELL_NEBULA] += mm.nebula_energy_reduction
        # use a large value for asteriod
        cost_map[cell_type == CELL_ASTERIOD] += asteriod_cost
        # cell energy cost change the cost map but max at 0 to prevent from loop
        cost_map = np.maximum(cost_map, 0)
        # Add extra step cost to favour shorter path
        cost_map += extra_step_cost

      min_neighbors = minimum_filter(energy_cost,
                                     footprint=kernel,
                                     mode='constant',
                                     cval=np.inf)
      with np.errstate(invalid='ignore'):
        energy_cost = np.minimum(energy_cost, min_neighbors + cost_map)

      is_map_drifted = False

      # Do not shift map in first 50 steps because map info is not populated yet
      if mm.game_step > MAX_MATCH_STEPS:
        game_step = mm.game_step + n
        drift_speed = mm.nebula_drift_estimator.drift_speed
        if drift_speed is not None and is_drifted_step(game_step, drift_speed):
          is_map_drifted = True
          cell_type = shift_map_by_sign(cell_type, drift_speed)

    return energy_cost

  def encode_unit_actions(self, unit_to_cell):
    mm = self.mm
    unit_actions = np.zeros((MAX_UNIT_NUM, 3), dtype=np.int32)

    energy_map = mm.cell_energy_with_nebula_energy_reduction

    move_actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
    if USE_RANDOM:
      np.random.shuffle(move_actions)

    def select_move_action(unit_id, unit_pos, unit_energy, energy_cost):
      action = ACTION_CENTER
      if not np.isinf(energy_cost[unit_pos[0]][unit_pos[1]]):
        actions = []
        # for k in range(1, MAX_MOVE_ACTION_IDX + 1):
        for k in move_actions:
          nx, ny = (unit_pos[0] + DIRECTIONS[k][0],
                    unit_pos[1] + DIRECTIONS[k][1])
          if not is_pos_on_map((nx, ny)):
            continue

          if mm.cell_type[nx][ny] == CELL_ASTERIOD:
            continue

          # Do not move to cell with enemy energy > unit energy
          if unit_energy < mm.enemy_max_energy[nx][ny]:
            continue

          cost = energy_cost[nx][ny]
          if np.isinf(cost):
            continue

          r = -energy_map[nx][ny]  # try to move with more energy
          a = (cost, r, (nx, ny), k, DIRECTIONS_TO_ACTION[k])
          actions.append(a)
          # if not SUBMIT_AGENT and self.player == PLAYER1:

          # if self.player == PLAYER1:
          # print(
          # f"game_step={mm.game_step}, unit={unit_id} action={ACTION_ID_TO_NAME[k]}, from={unit_pos} to {(nx, ny)} dir={DIRECTIONS[k]} cost={cost}",
          # file=sys.stderr)

      if len(actions):
        actions.sort()
        action = actions[0][-1]
        next_pos = actions[0][2]
        kd = actions[0][3]

        # if self.player == PLAYER1:
        # print((
        # f"game_step={mm.game_step}, unit={unit_id} action[{action}]={ACTION_ID_TO_NAME[action]}, from={unit_pos} to "
        # f"{next_pos} dir={DIRECTIONS[kd]} {DIRECTIONS_TO_ACTION[kd]} cost={cost}"
        # ),
        # file=sys.stderr)
      return action

    for i in range(MAX_UNIT_NUM):
      if i not in unit_to_cell:
        unit_actions[i][0] = ACTION_CENTER
        continue

      cell_pos = unit_to_cell[i]
      mask, unit_pos, unit_energy = mm.get_unit_info(mm.player_id, i, t=0)
      if pos_equal(unit_pos, cell_pos):
        unit_actions[i][0] = ACTION_CENTER
        continue

      if not SUBMIT_AGENT:
        # if self.player == PLAYER1:
        print(
            f"pid=[{self.mm.player}] game_step={mm.game_step} sending unit={i} pos={unit_pos} to cell={cell_pos}",
            file=sys.stderr)

      energy_cost = self.compute_energy_cost_map(cell_pos)
      unit_actions[i][0] = select_move_action(i, unit_pos, unit_energy,
                                              energy_cost)

    return unit_actions

  def attack(self, unit_actions, unit_to_cell):
    mm = self.mm

    attackers = []
    for unit_id in range(MAX_UNIT_NUM):
      mask, unit_pos, unit_energy = mm.get_unit_info(mm.player_id,
                                                     unit_id,
                                                     t=0)
      if not mask or not can_attack(unit_energy, mm):
        continue

      if unit_actions[unit_id][0] != ACTION_CENTER:
        continue

      attackers.append((unit_id, unit_pos, unit_energy))

      if not SUBMIT_AGENT:
        print(
            f"attack candidates unit={unit_id} pos={unit_pos} e={unit_energy}",
            file=sys.stderr)

    if not attackers:
      return

    attack_positions = np.argwhere(self.enemy_hit_map)
    if len(attack_positions) == 0:
      if not SUBMIT_AGENT:
        print(f"no attack_positions found, return", file=sys.stderr)
      return

    def get_sap_damage(upos, unit_energy, cpos):
      if not is_within_sap_range(upos, cpos, self.mm.unit_sap_range):
        return -1

      h = self.enemy_hit_map[cpos[0]][cpos[1]]

      if (self.mm.team_point_mass[cpos[0]][cpos[1]] < 0.6
          and self.blind_shot_targets[cpos[0]][cpos[1]]):
        h += self.mm.unit_sap_cost * self.mm.unit_sap_dropoff_factor  # lower the priority of blind shot

      h *= min(unit_energy / BOOST_SAP_ENERGY_THRESHOOD, 1)
      return h

    weights = np.ones((len(attackers), len(attack_positions))) * -9999
    for i, (unit_id, unit_pos, unit_energy) in enumerate(attackers):
      for j, cpos in enumerate(attack_positions):
        weights[i, j] = get_sap_damage(unit_pos, unit_energy, cpos)

    attack_actions = []
    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for i, j in zip(rows, cols):
      wt = weights[i, j]
      if wt <= 0:
        continue

      attack_actions.append((attackers[i], attack_positions[j], wt))

      unit_id, unit_pos, unit_energy = attackers[i]
      if not SUBMIT_AGENT:
        print(f"found attacker unit={unit_id} pos={unit_pos} e={unit_energy}",
              file=sys.stderr)

      if not SUBMIT_AGENT:
        atk_pos = attack_positions[j]
        if self.blind_shot_targets[atk_pos[0]][atk_pos[1]]:
          print(f"blind shot from unit[{unit_id}]={unit_pos} at pos={atk_pos}",
                file=sys.stderr)

    # use attack with larger energy
    attack_actions.sort(key=lambda a:
                        (-a[2], -a[0][-1], a[0][0]))  # (unit_energy, unit_id)
    enemy_energy = mm.enemy_max_energy.copy()
    self.last_sap_locations.clear()
    self.last_sap_units_info = []
    for (unit_id, unit_pos, unit_energy), cpos, _ in attack_actions:
      sap_mask = gen_sap_range(cpos, d=1)
      is_blind_shot = self.blind_shot_targets[cpos[0]][cpos[1]]
      if (enemy_energy[sap_mask & (enemy_energy > 0)].sum() <= 0
          and not is_blind_shot):
        if not SUBMIT_AGENT:
          print(f'step={mm.game_step}, unit[{unit_pos}] sap saved',
                file=sys.stderr)
        continue

      dropoff_sap_cost = mm.unit_sap_cost * mm.unit_sap_dropoff_factor
      sap_cost = gen_sap_range(cpos, d=1, dtype=int, val=dropoff_sap_cost)
      sap_cost[cpos[0]][cpos[1]] = mm.unit_sap_cost
      enemy_energy -= sap_cost

      unit_actions[unit_id][0] = ACTION_SAP
      unit_actions[unit_id][1] = cpos[0] - unit_pos[0]
      unit_actions[unit_id][2] = cpos[1] - unit_pos[1]
      self.last_sap_locations.append(cpos)
      # self.last_sap_units_info.append(
      # (unit_id, unit_pos, unit_energy, self.env._seed, self.player))
      if not SUBMIT_AGENT:
        print(
            f'step={mm.game_step}, unit[{unit_pos}] sap at {cpos} with damage={wt}',
            file=sys.stderr)

  def act(self, step: int, raw_obs, remainingOverageTime: int = 60):
    """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
    if not SUBMIT_AGENT:
      print(f"============ game step {self.mm.game_step + 1} ========== ",
            file=sys.stderr)
    self.mm.update(raw_obs, self.prev_model_action)
    self.mm.add_sap_locations(self.last_sap_locations)

    unit_to_cell = self.compute_unit_to_cell()
    unit_actions = self.encode_unit_actions(unit_to_cell)
    self.attack(unit_actions, unit_to_cell)

    self.prev_model_action = {UNITS_ACTION: unit_actions}
    return unit_actions

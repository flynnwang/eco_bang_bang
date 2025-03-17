from typing import Callable, Dict, Optional, Tuple, Union, NamedTuple, Any, List
from argparse import Namespace
import os
import sys
import functools

import numpy as np
import torch
import scipy.optimize

from .env.const import *
from .env.luxenv import (
    MapManager,
    LuxS3Env,
    SapIndexer,
    anti_diag_sym,
    anti_diag_sym_i,
    EXT_ACTION_SHAPE,
    manhatten_distance,
    min_cost_bellman_ford,
    is_pos_on_map,
    minimum_filter,
    pos_equal,
)
from .model import create_model

SUBMIT_AGENT = False
# SUBMIT_AGENT = True

DO_SAMPLE = True
USE_MIRROR_TRANS = False

DEVICE = 'cpu'
if not SUBMIT_AGENT:
  # import random
  # DEVICE = random.sample(['cuda:0', 'cuda:1'], k=1)[0]
  DEVICE = 'cuda:0'

N_CELLS = MAP_WIDTH * MAP_HEIGHT

LOG3 = np.log(2)


@functools.lru_cache(maxsize=1024, typed=False)
def dd(dist, r=1.2):
  dist = min(dist, MAP_WIDTH * 2)
  return r**dist


def cell_idx_to_pos(idx):
  return int(idx % MAP_WIDTH), int(idx // MAP_WIDTH)


def can_attack(energy, mm, margin=3):
  return energy >= mm.unit_sap_cost + mm.unit_move_cost * margin


def cant_move_to(upos, cpos, mm):
  return (mm.cell_type[cpos[0]][cpos[1]] == CELL_ASTERIOD
          and not pos_equal(cpos, upos))


def gen_sap_range(pos, d, dtype=bool, val=True):
  sap_range = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=dtype)
  x0 = max(0, (pos[0] - d))
  x1 = min(MAP_WIDTH, (pos[0] + d + 1))
  y0 = max(0, (pos[1] - d))
  y1 = min(MAP_HEIGHT, (pos[1] + d + 1))
  sap_range[x0:x1, y0:y1] = val
  return sap_range


def is_within_sap_range(upos, cpos, unit_sap_range):
  return ((abs(upos[0] - cpos[0]) <= unit_sap_range)
          and (abs(upos[1] - cpos[1]) <= unit_sap_range))


def on_enemy_side(cpos, player_id):
  target_pos = (0, 0)
  if player_id == 1:
    target_pos = (23, 23)

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
    self.env = LuxS3Env("", obs_space_kwargs=obs_space_kwargs,
                        game_env=1)  # for calling _convert_observation
    self.env.sap_indexer = self.mm.sap_indexer
    assert self.env.sap_indexer is not None

    self.prev_model_action = None
    self.last_sap_locations = []

  def get_sap_hit_map(self, factor):
    hit_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=float)

    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.enemy_id, i, t=0)
      if not mask:
        continue

      if energy < 0:
        # print(f'Enemy unit [{i}] at {pos} with energy={energy}',
        # file=sys.stderr)
        continue

      if self.mm.enemy_positions[pos[0]][pos[1]]:
        p = self.mm.team_point_mass[pos[0]][pos[1]]
        hit_map[pos[0]][pos[1]] += (RELIC_SCORE * (p**2))

      for d in [1, 0]:
        x0 = max(0, (pos[0] - d))
        x1 = min(MAP_WIDTH, (pos[0] + d + 1))
        y0 = max(0, (pos[1] - d))
        y1 = min(MAP_HEIGHT, (pos[1] + d + 1))

        h = self.mm.unit_sap_cost
        if d == 1:
          h *= self.mm.unit_sap_dropoff_factor
        hit_map[x0:x1, y0:y1] += h
    return hit_map

  def compute_unit_to_cell(self):
    mm = self.mm
    is_explore_step = (mm.match_step <= 50 and mm.game_step < 303)

    match_observed = mm.match_observed + anti_diag_sym(mm.match_observed)
    energy_threshold = 60 + self.mm.match_step
    if mm.match_step >= 70:
      energy_threshold = 60

    def get_explore_weight(upos, energy, cpos):
      alpha = 1

      # last_ob_time = mm.last_observed_step[cpos[0]][cpos[1]]
      # t = mm.game_step - last_ob_time
      # alpha = np.log(t + 1) / LOGX

      if match_observed[cpos[0]][cpos[1]]:
        return 0

      wt = 3
      if not is_explore_step:
        wt /= 5

      return wt * alpha

    energy_map = mm.cell_energy.copy()
    energy_map[mm.cell_energy != CELL_UNKONWN] -= mm.unit_move_cost
    energy_map[mm.cell_type == CELL_NEBULA] -= mm.nebula_energy_reduction

    def get_fuel_energy(upos, energy, cpos):
      fuel = energy_map[cpos[0]][cpos[1]]
      fuel = right_tailed_exp(energy, fuel, energy_threshold)
      return fuel

    def get_open_relic_nb(upos, energy, cpos):
      """First visit on relic neighbour"""
      if not mm.is_relic_neighbour[cpos[0]][cpos[1]]:
        return 0

      # if not mm.match_visited[cpos[0]][cpos[1]]:
      # return RELIC_NB_SCORE

      p = mm.team_point_mass[cpos[0]][cpos[1]]
      if p < 0.1:
        return 0

      last_visited_step = mm.last_visited_step[cpos[0]][cpos[1]]
      t = mm.game_step - last_visited_step
      alpha = np.log(t + 1) / LOG3
      w = min(alpha, 1) * RELIC_NB_SCORE

      # if on_enemy_side(cpos, mm.player_id):
      # w = left_tailed_exp(energy, w, energy_threshold)

      cpos_nb_mask = gen_sap_range(cpos, self.mm.unit_sap_range)
      if (mm.enemy_max_energy[cpos_nb_mask] > energy).sum() > 0:
        w = left_tailed_exp(energy, w, energy_threshold)
        # w /= 2

      return w

    def stay_on_relic(upos, energy, cpos):
      p = mm.team_point_mass[cpos[0]][cpos[1]]
      w = 0
      if p > 0.8:
        w += RELIC_SCORE * p

      # if on_enemy_side(cpos, mm.player_id):

      cpos_nb_mask = gen_sap_range(cpos, self.mm.unit_sap_range)
      if (mm.enemy_max_energy[cpos_nb_mask] > energy).sum() > 0:
        w = left_tailed_exp(energy, w, energy_threshold)
      return w

    hit_factor = 10
    enemy_hit_map = self.get_sap_hit_map(hit_factor)
    self.enemy_hit_map = enemy_hit_map

    def get_sap_enemy_score(upos, energy, cpos):
      """Max sap damage that could be hit from the `cpos`."""
      if not can_attack(energy, mm):
        return 0

      # sap_range = np.ones((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
      # sap_range = generate_manhattan_mask(sap_range, cpos,
      # self.mm.unit_sap_range)
      sap_range = gen_sap_range(cpos, self.mm.unit_sap_range)

      h = enemy_hit_map[sap_range].max()
      h /= hit_factor

      # h = left_tailed_exp(energy, h, energy_threshold)
      # h *= (energy / 200)

      # sap if energy is large (and unit not on relic)
      if self.mm.team_point_mass[pos[0]][pos[1]] < 0.6:
        h *= max((energy / energy_threshold), 1)
      return h

    score_debug = {}

    def get_unit_cell_wt(upos, energy, cpos, unit_cost_map):
      if cant_move_to(upos, cpos, mm):
        return -9999

      if energy < unit_cost_map[cpos[0]][cpos[1]]:
        # if not SUBMIT_AGENT:
        # print(f'game_step={mm.game_step}: skip due to inf at {cpos}',
        # file=sys.stderr)
        return -9999

      # Do not target cell with enemy energy > unit energy
      if energy < mm.enemy_max_energy[cpos[0]][cpos[1]]:
        return -9999

      # mdist = manhatten_distance(upos, cpos) + 7
      mdist = dd(manhatten_distance(upos, cpos) + 1)
      wt = 0.0001

      energy_ratio = energy / energy_threshold

      expore_wt = 0
      if energy >= 50:
        expore_wt = get_explore_weight(upos, energy, cpos)

      fuel_wt = get_fuel_energy(upos, energy, cpos)

      relic_nb_wt = get_open_relic_nb(upos, energy, cpos)

      on_relic_wt = stay_on_relic(upos, energy, cpos)

      sap_wt = get_sap_enemy_score(upos, energy, cpos)

      wt += (expore_wt + fuel_wt + relic_nb_wt + on_relic_wt + sap_wt) / mdist

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
      return wt

    weights = np.ones((MAX_UNIT_NUM, N_CELLS)) * -9999
    cell_index = list(range(N_CELLS))
    # np.random.shuffle(cell_index)

    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.player_id, i, t=0)
      if not mask:
        continue

      unit_cost_map = self.compute_energy_cost_map(pos, asteriod_cost=75)
      for j in cell_index:
        r, c = cell_idx_to_pos(j)
        # if energy < mm.unit_move_cost:
        # if pos_equal(pos, (r, c)):
        # weights[i, j] = 100
        # # TODO: This will block attack position
        # else:
        weights[i, j] = get_unit_cell_wt(pos, energy, (r, c), unit_cost_map)

    unit_to_cell = {}
    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for unit_id, target_id in zip(rows, cols):
      wt = weights[unit_id, target_id]
      # if wt < 1e-6:  # TODO: use 0?
      if wt < -1:  # TODO: use 0?
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
                              asteriod_cost=100,
                              N=MAP_WIDTH * 2):
    mm = self.mm
    cost_map = np.full((MAP_WIDTH, MAP_HEIGHT), float(mm.unit_move_cost))

    # nebula energy reduction adds extra cost
    cost_map[mm.cell_type == CELL_NEBULA] += mm.nebula_energy_reduction

    # cell energy cost change the cost map but max at 0 to prevent from loop
    cost_map -= mm.cell_energy
    cost_map = np.maximum(cost_map, 1)

    # use a big value for asteriod
    cost_map[mm.cell_type == CELL_ASTERIOD] = asteriod_cost

    energy_cost = np.full((MAP_WIDTH, MAP_HEIGHT), np.inf, dtype=np.float64)
    energy_cost[target_pos[0]][target_pos[1]] = 0

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    for _ in range(N):
      min_neighbors = minimum_filter(energy_cost,
                                     footprint=kernel,
                                     mode='constant',
                                     cval=np.inf)
      with np.errstate(invalid='ignore'):
        energy_cost = np.minimum(energy_cost, min_neighbors + cost_map)

    return energy_cost

  def encode_unit_actions(self, unit_to_cell):
    mm = self.mm
    unit_actions = np.zeros((MAX_UNIT_NUM, 3), dtype=np.int32)

    energy_map = mm.cell_energy_with_nebula_energy_reduction

    def select_move_action(unit_id, unit_pos, unit_energy, energy_cost):
      action = ACTION_CENTER
      if not np.isinf(energy_cost[unit_pos[0]][unit_pos[1]]):
        actions = []
        for k in range(1, MAX_MOVE_ACTION_IDX + 1):
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
          a = (cost, r, DIRECTIONS_TO_ACTION[k])
          actions.append(a)
          if not SUBMIT_AGENT:
            print(
                f"game_step={mm.game_step}, unit={unit_id} action={ACTION_ID_TO_NAME[k]}, cost={cost}",
                file=sys.stderr)

      if len(actions):
        actions.sort()
        action = actions[0][-1]
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
        print(
            f"game_step={mm.game_step} sending unit={i} pos={unit_pos} to cell={cell_pos}",
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

    def get_sap_damage(upos, cpos):
      if not is_within_sap_range(upos, cpos, self.mm.unit_sap_range):
        return -1

      return self.enemy_hit_map[cpos[0]][cpos[1]]

    weights = np.ones((len(attackers), len(attack_positions))) * -9999
    for i, (unit_id, unit_pos, _) in enumerate(attackers):
      for j, cpos in enumerate(attack_positions):
        weights[i, j] = get_sap_damage(unit_pos, cpos)

    attack_actions = []
    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for i, j in zip(rows, cols):
      wt = weights[i, j]
      if wt <= 0:
        continue

      attack_actions.append((attackers[i], attack_positions[j], wt))

      if not SUBMIT_AGENT:
        unit_id, unit_pos, unit_energy = attackers[i]
        print(f"found attacker unit={unit_id} pos={unit_pos} e={unit_energy}",
              file=sys.stderr)

    # use attack with larger energy
    attack_actions.sort(key=lambda a:
                        (-a[2], -a[0][-1], a[0][0]))  # (unit_energy, unit_id)
    enemy_energy = mm.enemy_max_energy.copy()
    self.last_sap_locations.clear()
    for (unit_id, unit_pos, unit_energy), cpos, _ in attack_actions:
      sap_mask = gen_sap_range(cpos, d=1)
      if enemy_energy[sap_mask & (enemy_energy > 0)].sum() <= 0:
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
      if not SUBMIT_AGENT:
        print(
            f'step={mm.game_step}, unit[{unit_pos}] sap at {cpos} with damage={wt}',
            file=sys.stderr)

  def act(self, step: int, raw_obs, remainingOverageTime: int = 60):
    """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
    print(f"============ game step {self.mm.game_step + 1} ========== ",
          file=sys.stderr)
    self.mm.update(raw_obs, self.prev_model_action)
    self.mm.add_sap_locations(self.last_sap_locations)

    if not self.env.prev_raw_obs:
      self.env.prev_raw_obs = {self.mm.player: raw_obs}

    unit_to_cell = self.compute_unit_to_cell()
    unit_actions = self.encode_unit_actions(unit_to_cell)
    self.attack(unit_actions, unit_to_cell)
    return unit_actions

from typing import Callable, Dict, Optional, Tuple, Union, NamedTuple, Any, List
from argparse import Namespace
import os
import sys

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
    generate_manhattan_mask,
)
from .model import create_model

# SUBMIT_AGENT = False
SUBMIT_AGENT = True

DO_SAMPLE = True
USE_MIRROR_TRANS = False

DEVICE = 'cpu'
if not SUBMIT_AGENT:
  # import random
  # DEVICE = random.sample(['cuda:0', 'cuda:1'], k=1)[0]
  DEVICE = 'cuda:0'

N_CELLS = MAP_WIDTH * MAP_HEIGHT

LOGX = np.log(5)


def cell_idx_to_pos(idx):
  return int(idx % MAP_WIDTH), int(idx // MAP_WIDTH)


def pos_equal(x, y):
  return x[0] == y[0] and x[1] == y[1]


def can_attack(energy, mm, margin=3):
  return energy >= mm.unit_sap_cost + mm.unit_move_cost * margin


def cant_move_to(upos, cpos, mm):
  return (mm.cell_type[cpos[0]][cpos[1]] == CELL_ASTERIOD
          and not pos_equal(cpos, upos))


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

  def get_sap_hit_map(self, factor, stay_on_relic):
    hit_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=float)

    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.enemy_id, i, t=0)
      if not mask:
        continue

      if self.mm.enemy_positions[pos[0]][pos[1]]:
        hit_map[pos[0]][pos[1]] += stay_on_relic(None, energy, pos) * factor

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

    def get_explore_weight(upos, energy, cpos):
      alpha = 1

      # last_ob_time = mm.last_observed_step[cpos[0]][cpos[1]]
      # t = mm.game_step - last_ob_time
      # alpha = np.log(t + 1) / LOGX

      if match_observed[cpos[0]][cpos[1]]:
        return 0

      wt = 1
      if not is_explore_step:
        wt /= 5

      return wt * alpha

    energy_map = mm.cell_energy.copy() - mm.unit_move_cost
    energy_map[mm.cell_type == CELL_NEBULA] -= mm.nebula_energy_reduction

    def get_fuel_energy(upos, energy, cpos):
      cell_energy = energy_map[cpos[0]][cpos[1]]
      if energy >= 75:
        v = 20
        m = 60
        alpha = np.exp(-(cell_energy - m)**2 / (v**2 * 2.0))
        cell_energy *= alpha
      return cell_energy

    def get_open_relic_nb(upos, energy, cpos):
      """First visit on relic neighbour"""
      if not mm.is_relic_neighbour[cpos[0]][cpos[1]]:
        return 0

      if mm.match_visited[cpos[0]][cpos[1]]:
        return 0

      return 5

    def stay_on_relic(upos, energy, cpos):
      p = mm.team_point_mass[cpos[0]][cpos[1]]

      w = 50 * p
      if p < 0.8:
        w = 30 * p

      # has_enemy = False
      # if mm.enemy_positions[cpos[0]][cpos[1]]:
      # has_enemy = True
      # if has_enemy:
      # w *= 0.1

      return w

    hit_factor = 10
    enemy_hit_map = self.get_sap_hit_map(hit_factor, stay_on_relic)
    self.enemy_hit_map = enemy_hit_map

    def get_sap_enemy_score(upos, energy, cpos):
      """Max sap damage that could be hit from the `cpos`."""
      if not can_attack(energy, mm):
        return 0

      sap_range = np.ones((MAP_WIDTH, MAP_HEIGHT), dtype=bool)
      sap_range = generate_manhattan_mask(sap_range, cpos,
                                          self.mm.unit_sap_range)

      h = enemy_hit_map[sap_range].max()
      h /= hit_factor
      return h

    def get_unit_cell_wt(upos, energy, cpos, unit_cost_map):
      if cant_move_to(upos, cpos, mm):
        return -9999

      if energy < unit_cost_map[cpos[0]][cpos[1]]:
        # if not SUBMIT_AGENT:
        # print(f'game_step={mm.game_step}: skip due to inf at {cpos}',
        # file=sys.stderr)
        return -9999

      mdist = manhatten_distance(upos, cpos) + 7
      wt = 0.0001

      if energy >= 50:
        wt += get_explore_weight(upos, energy, cpos) / mdist

      wt += get_fuel_energy(upos, energy, cpos) / mdist

      wt += get_open_relic_nb(upos, energy, cpos) / mdist

      wt += stay_on_relic(upos, energy, cpos) / mdist

      wt += get_sap_enemy_score(upos, energy, cpos) / mdist
      return wt

    weights = np.ones((MAX_UNIT_NUM, N_CELLS)) * -9999
    cell_index = list(range(N_CELLS))
    np.random.shuffle(cell_index)

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
      if wt < 1e-6:  # TODO: use 0?
        continue
      unit_to_cell[unit_id] = cell_idx_to_pos(target_id)

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

    def select_move_action(unit_id, unit_pos, energy_cost):
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
      unit_actions[i][0] = select_move_action(i, unit_pos, energy_cost)

    return unit_actions

  def attack(self, unit_actions, unit_to_cell):
    mm = self.mm

    attackers = []
    for unit_id, cpos in unit_to_cell.items():
      mask, unit_pos, unit_energy = mm.get_unit_info(mm.player_id,
                                                     unit_id,
                                                     t=0)
      if not mask or not can_attack(unit_energy, mm):
        continue

      if pos_equal(unit_pos, cpos):
        attackers.append((unit_id, unit_pos))

    if not attackers:
      return

    attack_positions = np.argwhere(self.enemy_hit_map)
    if len(attack_positions) == 0:
      return

    def get_sap_damage(upos, cpos):
      if manhatten_distance(upos, cpos) > self.mm.unit_sap_range:
        return -1

      return self.enemy_hit_map[cpos[0]][cpos[1]]

    weights = np.ones((len(attackers), len(attack_positions))) * -9999
    for i, (unit_id, unit_pos) in enumerate(attackers):
      for j, cpos in enumerate(attack_positions):
        weights[i, j] = get_sap_damage(unit_pos, cpos)

    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for i, j in zip(rows, cols):
      wt = weights[i, j]
      if wt <= 0:
        continue

      unit_id, unit_pos = attackers[i]
      cpos = attack_positions[j]

      unit_actions[unit_id][0] = ACTION_SAP
      unit_actions[unit_id][1] = cpos[0] - unit_pos[0]
      unit_actions[unit_id][2] = cpos[1] - unit_pos[1]
      if not SUBMIT_AGENT:
        print(
            f'step={mm.game_step}, unit[{unit_pos}] sap at {cpos} with damage={wt}',
            file=sys.stderr)

  def act(self, step: int, raw_obs, remainingOverageTime: int = 60):
    """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
    self.mm.update(raw_obs, self.prev_model_action)
    if not self.env.prev_raw_obs:
      self.env.prev_raw_obs = {self.mm.player: raw_obs}

    unit_to_cell = self.compute_unit_to_cell()
    unit_actions = self.encode_unit_actions(unit_to_cell)
    self.attack(unit_actions, unit_to_cell)
    return unit_actions

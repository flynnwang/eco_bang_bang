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
)
from .model import create_model

SUBMIT_AGENT = False

DO_SAMPLE = True
USE_MIRROR_TRANS = False

DEVICE = 'cpu'
if not SUBMIT_AGENT:
  # import random
  # DEVICE = random.sample(['cuda:0', 'cuda:1'], k=1)[0]
  DEVICE = 'cuda:0'

N_CELLS = MAP_WIDTH * MAP_HEIGHT


def cell_idx_to_pos(idx):
  return int(idx % MAP_WIDTH), int(idx // MAP_WIDTH)


def pos_equal(x, y):
  return x[0] == y[0] and x[1] == y[1]


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

  def compute_unit_to_cell(self):
    mm = self.mm

    def get_explore_weight(upos, energy, cpos):
      if mm.game_step > 303:
        return 0

      # if mm.match_visited[cpos[0]][cpos[1]]:
      if mm.match_observed[cpos[0]][cpos[1]]:
        return 0

      if mm.cell_type[cpos[0]][cpos[1]] == CELL_ASTERIOD:
        return 0

      wt = 1
      target_pos = (23, 23)
      if self.mm.player_id == 0:
        target_pos = (0, 0)

      if manhatten_distance(cpos, target_pos) > 24:
        wt = 0.5

      return 1

    energy_map = mm.cell_energy.copy() - mm.unit_move_cost
    energy_map[mm.cell_type == CELL_NEBULA] -= mm.nebula_energy_reduction

    def get_fuel_energy(upos, energy, cpos):
      if mm.cell_type[cpos[0]][cpos[1]] == CELL_ASTERIOD:
        return 0

      cell_energy = energy_map[cpos[0]][cpos[1]]
      if energy >= 75:
        v = 20
        m = 60
        alpha = np.exp(-(cell_energy - m)**2 / (v**2 * 2.0))
        cell_energy *= alpha

      return cell_energy

    def get_open_relic_nb(upos, energy, cpos):
      if mm.cell_type[cpos[0]][cpos[1]] == CELL_ASTERIOD:
        return 0

      if not mm.is_relic_neighbour[cpos[0]][cpos[1]]:
        return 0

      if mm.match_visited[cpos[0]][cpos[1]]:
        return 0

      return 5

    def stay_on_relic(upos, energy, cpos):
      if mm.cell_type[cpos[0]][cpos[1]] == CELL_ASTERIOD:
        return 0

      p = mm.team_point_mass[cpos[0]][cpos[1]]
      return p * 50

    def get_unit_cell_wt(upos, energy, cpos):
      mdist = manhatten_distance(upos, cpos) + 7
      wt = 0

      if energy >= 50:
        wt += get_explore_weight(upos, energy, cpos) / mdist

      wt += get_fuel_energy(upos, energy, cpos) / mdist

      wt += get_open_relic_nb(upos, energy, cpos) / mdist

      wt += stay_on_relic(upos, energy, cpos) / mdist
      return wt

    weights = np.ones((MAX_UNIT_NUM, N_CELLS)) * -9999
    cell_index = list(range(N_CELLS))
    np.random.shuffle(cell_index)

    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.player_id, i, t=0)
      if not mask:
        continue

      for j in cell_index:
        r, c = cell_idx_to_pos(j)
        if energy < mm.unit_move_cost:
          if pos_equal(pos, (r, c)):
            weights[i, j] = 100
        else:
          weights[i, j] = get_unit_cell_wt(pos, energy, (r, c))

    unit_to_cell = {}
    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for unit_id, target_id in zip(rows, cols):
      wt = weights[unit_id, target_id]
      if wt < 1e-5:
        continue
      unit_to_cell[unit_id] = cell_idx_to_pos(target_id)

    return unit_to_cell

  def compute_energy_cost_map(self, unit_pos, target_pos):
    mm = self.mm
    cost_map = np.full((MAP_WIDTH, MAP_HEIGHT), float(mm.unit_move_cost))

    # nebula energy reduction adds extra cost
    cost_map[mm.cell_type == CELL_NEBULA] += mm.nebula_energy_reduction

    # cell energy cost change the cost map but max at 0 to prevent from loop
    cost_map -= mm.cell_energy
    cost_map = np.maximum(cost_map, 1)

    # use a big value for asteriod
    cost_map[mm.cell_type == CELL_ASTERIOD] = 100

    energy_cost = np.full((MAP_WIDTH, MAP_HEIGHT), np.inf, dtype=np.float64)
    energy_cost[target_pos[0]][target_pos[1]] = 0

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    N = MAP_WIDTH * 2
    for _ in range(N):
      min_neighbors = minimum_filter(energy_cost,
                                     footprint=kernel,
                                     mode='constant',
                                     cval=np.inf)
      with np.errstate(invalid='ignore'):
        energy_cost = np.minimum(energy_cost, min_neighbors + cost_map)

      # print(f"energy_cost.shape={energy_cost.shape}, unit_pos={unit_pos}",
      # file=sys.stderr)
      # if not np.isinf(energy_cost[unit_pos[0]][unit_pos[1]]):
      # break

    return energy_cost

  def encode_unit_actions(self, unit_to_cell):
    mm = self.mm
    unit_actions = np.zeros((MAX_UNIT_NUM, 3), dtype=np.int32)

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

          if unit_id % 2 == 0:
            r = k
          else:
            r = 5 - k
          a = (cost, r, DIRECTIONS_TO_ACTION[k])
          actions.append(a)
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

      print(
          f"game_step={mm.game_step} sending unit={i} pos={unit_pos} to cell={cell_pos}",
          file=sys.stderr)

      energy_cost = self.compute_energy_cost_map(unit_pos, cell_pos)
      unit_actions[i][0] = select_move_action(i, unit_pos, energy_cost)

    return unit_actions

  def act(self, step: int, raw_obs, remainingOverageTime: int = 60):
    """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
    self.mm.update(raw_obs, self.prev_model_action)
    if not self.env.prev_raw_obs:
      self.env.prev_raw_obs = {self.mm.player: raw_obs}

    unit_to_cell = self.compute_unit_to_cell()
    action = self.encode_unit_actions(unit_to_cell)
    return action

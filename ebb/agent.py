from typing import Callable, Dict, Optional, Tuple, Union, NamedTuple, Any, List
from collections import OrderedDict, deque, defaultdict, Counter
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
    get_player_init_pos,
    set_value_by_range,
)

# SUBMIT_AGENT = False
SUBMIT_AGENT = True

DO_SAMPLE = True
USE_MIRROR_TRANS = False

USE_RANDOM = True
# USE_RANDOM = False

DEVICE = 'cpu'
if not SUBMIT_AGENT:
  # import random
  # DEVICE = random.sample(['cuda:0', 'cuda:1'], k=1)[0]
  DEVICE = 'cuda:0'

N_CELLS = MAP_WIDTH * MAP_HEIGHT

LOG3 = np.log(4)


def draw_line(mp, pos):
  pos2 = anti_diag_sym_i(pos)
  min_x = min(pos[0], pos2[0])
  max_x = max(pos[0], pos2[0])
  min_y = min(pos[1], pos2[1])
  max_y = max(pos[1], pos2[1])
  for i, x in enumerate(range(min_x, max_x + 1)):
    y = min_y + i
    mp[x][y] = 1
  return mp


@functools.lru_cache(maxsize=1024, typed=False)
def dd(dist, r=1.1):
  dist = min(dist, MAP_WIDTH * 2)
  return r**dist


def cell_idx_to_pos(idx):
  return int(idx % MAP_WIDTH), int(idx // MAP_WIDTH)


def pos_to_cell_idx(pos):
  return pos[0] + pos[1] * MAP_WIDTH


def can_attack(energy, mm, margin=3):
  return energy >= mm.unit_sap_cost + mm.unit_move_cost * margin


def cant_move_to(upos, cpos, mm):
  return (mm.cell_type[cpos[0]][cpos[1]] == CELL_ASTERIOD
          and not pos_equal(cpos, upos))


def is_within_sap_range(upos, cpos, unit_sap_range):
  return ((abs(upos[0] - cpos[0]) <= unit_sap_range)
          and (abs(upos[1] - cpos[1]) <= unit_sap_range))


def on_enemy_side(cpos, player_id, use_mirror):
  target_pos = get_player_init_pos(player_id, use_mirror)
  mdist = manhatten_distance(target_pos, cpos)
  return mdist > MAP_WIDTH


def on_team_side(cpos, player_id, use_mirror):
  return not on_enemy_side(cpos, player_id, use_mirror)


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
RELIC_NB_SCORE = 20

EXPLORE_CELL_SCORE = 3
MAX_EXPLORE_SCORE = 10

MIN_OPEN_RELIC_NB_PROB = 0.01
IS_RELIC_CELL_PROB = 0.8

BOOST_SAP_ENERGY_THRESHOOD = 120


class Agent:

  def __init__(self, player: str, env_cfg, use_mirror=False) -> None:
    self.player = player
    self.env_cfg = env_cfg
    # np.random.seed(0)

    # For testing
    # if player == PLAYER1:
    # use_mirror = True

    obs_space_kwargs = {
        'use_energy_cost_map': True,
        'use_single_player': False
    }

    self.mm = MapManager(player,
                         env_cfg,
                         transpose=False,
                         sap_indexer=SapIndexer(),
                         use_mirror=use_mirror,
                         use_hidden_relic_estimator=True)
    self.prev_model_action = None
    self.last_sap_locations = []
    self.unit_to_cell = None
    self.unit_score = defaultdict(int)

  def get_enemy_sap_cost_map(self, unit_energy=None, extra_range=0):
    # TODO: exclude the position enemy can not see
    enemy_sap_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=float)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.enemy_id, i, t=0)
      if not mask or energy < self.mm.unit_sap_cost:
        continue
      if unit_energy is not None and energy < unit_energy:
        continue
      # add 1 for dropoff and extra_range for margin
      sap_range = gen_sap_range(pos, self.mm.unit_sap_range + 1 + extra_range)
      enemy_sap_map[sap_range] += self.mm.unit_sap_cost
    return enemy_sap_map

  def get_enemy_max_energy_level(self, unit_energy, max_dist=3):
    enemy_max_energy = self.mm.enemy_max_energy.copy()
    enemy_positions = (enemy_max_energy >= unit_energy)
    for d in range(max_dist):
      dist = max_dist - d - 1
      cost = (d / (max_dist - 1) + 1)
      position_mask = maximum_filter(enemy_positions, size=2 * dist + 1)
      enemy_max_energy[position_mask] = cost
    return enemy_max_energy

  def get_sap_hit_map(self, factor):
    mm = self.mm

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

    def get_nearest_relic_position(upos):
      min_dist = 99999
      min_pos = None
      # print(
      # f' >>> relic node positions = {self.mm.hidden_relic_estimator.relic_node_positions}',
      # file=sys.stderr)
      for pos in self.mm.hidden_relic_estimator.relic_node_positions:
        if not on_team_side(pos, self.mm.player_id, self.mm.use_mirror):
          # print(f' >>> skip relic pos= {pos}', file=sys.stderr)
          continue

        d = manhatten_distance(upos, pos)
        if d < min_dist:
          min_pos = pos
          min_dist = d
      return min_pos

    hit_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=float)

    def add_pos(pos, dx, dy, v):
      np = (pos[0] + dx, pos[1] + dy)
      if is_pos_on_map(np):
        hit_map[np[0]][np[1]] += v

    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.enemy_id, i, t=0)
      if not mask or energy < 0:
        continue

      is_enemy_on_relic = False
      if self.mm.team_point_mass[pos[0]][pos[1]] > IS_RELIC_CELL_PROB:
        is_enemy_on_relic = True

      # for current enemy position
      hit_map[pos[0]][pos[1]] += self.mm.unit_sap_cost
      # print(
      # f' >>> enemy[{i}] at {pos} attack current pos = {pos} = {self.mm.unit_sap_cost}, on_relic={is_enemy_on_relic}',
      # file=sys.stderr)

      # For on-relic enemy, add extra score
      if is_enemy_on_relic:
        p = self.mm.team_point_mass[pos[0]][pos[1]]
        if p > IS_RELIC_CELL_PROB:
          hit_map[pos[0]][pos[1]] += RELIC_SCORE
      else:
        # For enemy next move (guessed: dist to nearest relic position)
        x, y = np.ogrid[:MAP_WIDTH, :MAP_HEIGHT]

      # For enemy next move (guessed: dist to nearest relic position)
      relic_pos = get_nearest_relic_position(pos)
      # print(f' >>> nearest relic = {relic_pos}', file=sys.stderr)
      if relic_pos is not None:
        rx, ry = relic_pos
        px, py = pos
        dx = 0
        if rx != px:
          dx = +1 if px < rx else -1

        dy = 0
        if ry != py:
          dy = +1 if py < ry else -1

        sap1 = mm.unit_sap_cost + 1
        sap0 = mm.unit_sap_cost * mm.unit_sap_dropoff_factor
        if dx != 0 and dy != 0:
          add_pos(pos, dx, 0, sap1)
          add_pos(pos, 0, dy, sap1)
          add_pos(pos, dx, dy, sap0)

        if (dx == 0 and dy != 0):
          add_pos(pos, -1, 0, sap0)
          add_pos(pos, +1, 0, sap0)
          add_pos(pos, -1, dy, sap0)
          add_pos(pos, 0, dy, sap1)
          add_pos(pos, +1, dy, sap0)

        if (dx != 0 and dy == 0):
          add_pos(pos, 0, -1, sap0)
          add_pos(pos, 0, +1, sap0)
          add_pos(pos, dx, -1, sap0)
          add_pos(pos, dx, 0, sap1)
          add_pos(pos, dx, +1, sap0)

    return hit_map

  def gen_fire_zone(self, dist_to_init_pos):
    """Fire zone is the positive energy cells that could either attack enemy
    relic position or protect my reilc points."""
    mm = self.mm

    team_point_mask = (mm.team_point_mass > IS_RELIC_CELL_PROB)

    attack_path_mask = mm.is_relic_node.copy()
    defense_zone = np.zeros(MAP_SHAPE2, dtype=bool)

    defense_zone_range = 7 * 2 + 1
    enemy_init_pos = get_player_init_pos(mm.enemy_id, mm.use_mirror)
    relic_node_positions = mm.hidden_relic_estimator.relic_node_positions
    for pos in relic_node_positions:
      if not on_team_side(pos, mm.player_id, mm.use_mirror):
        continue

      draw_line(attack_path_mask, pos)

      relic_defense_zone = gen_sap_range(pos, 2) & team_point_mask
      relic_defense_zone = maximum_filter(relic_defense_zone,
                                          size=defense_zone_range)

      dist = manhatten_distance(pos, enemy_init_pos)
      x, y = np.ogrid[:MAP_WIDTH, :MAP_HEIGHT]
      enemy_relic_dist = (np.abs(x - enemy_init_pos[0]) +
                          np.abs(y - enemy_init_pos[1]))

      relic_defense_zone &= (enemy_relic_dist <= (dist + 1))
      defense_zone |= relic_defense_zone

    attack_path_mask = maximum_filter(attack_path_mask, size=7)

    enemy_side_mask = (dist_to_init_pos >= MAP_WIDTH)
    fire_zone = (team_point_mask & enemy_side_mask)

    fire_zone_range = mm.unit_sap_range * 2 + 1
    fire_zone = maximum_filter(fire_zone, fire_zone_range)

    fire_zone = fire_zone.astype(int) + attack_path_mask

    return fire_zone, defense_zone, attack_path_mask

  def compute_unit_to_cell(self):
    mm = self.mm

    final_stage_start_step = 70

    match_observed = mm.match_observed + anti_diag_sym(mm.match_observed)
    energy_threshold = 100
    if mm.match_step >= final_stage_start_step:
      energy_threshold = 60

    has_found_relic = mm.has_found_relic_in_match()
    n = (MAP_WIDTH * MAP_HEIGHT)
    n_explore = n - match_observed.sum()
    expore_score = n * EXPLORE_CELL_SCORE / (n_explore + 1)

    # print(
    # f' +++ step={mm.game_step} to-exp-cell-num={n_explore} exp_score={expore_score} has_found_relic={has_found_relic}, last_match_found_relic={mm.last_match_found_relic}',
    # file=sys.stderr)

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
    # energy_map[mm.cell_type == CELL_ASTERIOD] -= 100 # adding this will make unit move away from it.
    if not SUBMIT_AGENT:
      print(
          f'>>>>>>>>>>>>>>> nebula_energy_reduction={mm.nebula_energy_reduction}',
          file=sys.stderr)

    player_init_pos = get_player_init_pos(mm.player_id, self.mm.use_mirror)
    d1 = generate_manhattan_dist(MAP_SHAPE2,
                                 player_init_pos).astype(np.float32)
    fire_zone, defense_zone, attack_path_mask = self.gen_fire_zone(d1)
    self.defense_zone = defense_zone
    d1 /= MAP_WIDTH

    defense_start_step = 0
    if not has_found_relic and mm.last_match_found_relic:
      defense_start_step = 30
=======
      defense_start_step = 16
>>>>>>> c4d73b1 (sync agent.py)

    def get_fuel_energy(upos, energy, cpos):
      e = fuel = energy_map[cpos[0]][cpos[1]]
      fuel = right_tailed_exp(energy, fuel, energy_threshold)

      is_in_defense_zone = defense_zone[cpos[0]][cpos[1]]
      is_in_fire_zone = fire_zone[cpos[0]][cpos[1]]
<<<<<<< HEAD
      is_on_attack_path = attack_path_mask[cpos[0]][cpos[1]]

      is_in_boost_zone = False
      if mm.match_step >= defense_start_step:
        # if mm.match_step < 50 and energy < BOOST_SAP_ENERGY_THRESHOOD:
        # if energy < BOOST_SAP_ENERGY_THRESHOOD:
        # is_in_fire_zone = False
        # is_on_attack_path = False

        # if mm.game_step >= 50 or energy >= BOOST_SAP_ENERGY_THRESHOOD:
        if mm.game_step >= 50 and energy >= BOOST_SAP_ENERGY_THRESHOOD:
          is_in_defense_zone = False

        is_in_boost_zone = (is_in_fire_zone or is_in_defense_zone
                            or is_on_attack_path)

      d = d1[cpos[0]][cpos[1]]
      if e > 0 and is_in_boost_zone:
        boost = (e * d) * 0.1  # defense: <5,  fire: 6-10
        if is_in_fire_zone or is_in_defense_zone:
          boost += (7 * d)  # ~5 * 3
        fuel += boost
=======

      is_in_boost_zone = False
      if mm.match_step >= defense_start_step:
        if mm.match_step < 50 and energy < BOOST_SAP_ENERGY_THRESHOOD:
          is_in_fire_zone = False

        if mm.game_step >= 50 or energy >= BOOST_SAP_ENERGY_THRESHOOD:
          is_in_defense_zone = False

        is_in_boost_zone = is_in_fire_zone or is_in_defense_zone

      if e > 0 and is_in_boost_zone:
        fuel += (e * d1[cpos[0]][cpos[1]])
>>>>>>> c4d73b1 (sync agent.py)

      return fuel

    enemy_sap_cost = self.get_enemy_sap_cost_map()
    self.enemy_sap_cost = enemy_sap_cost

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
      if on_enemy_side(cpos, mm.player_id, mm.use_mirror):
        # v = mm.unit_sap_cost / 10 * p
        v = mm.unit_sap_cost / 10

      # If enemy may sap it, lower its weight
      if enemy_sap_cost[cpos[0]][cpos[1]] > 0:
        v = mm.unit_sap_cost / 10 * 0.5

      last_visited_step = mm.last_visited_step[cpos[0]][cpos[1]]
      t = mm.game_step - last_visited_step
      alpha = np.log(t + 1) / LOG3
      w = min(alpha, 1) * v

      # cpos_nb_mask = gen_sap_range(cpos, self.mm.unit_sap_range)
      # if (mm.enemy_max_energy[cpos_nb_mask] > energy).sum() > 0:
      # w = -1
      return w

    init_pos = get_player_init_pos(mm.enemy_id, mm.use_mirror)
    enemy_half = generate_manhattan_mask(MAP_SHAPE2,
                                         init_pos,
                                         range_limit=MAP_WIDTH - 1)
    blind_shot_targets = ((~mm.visible) &
                          (mm.team_point_mass > IS_RELIC_CELL_PROB)
                          & enemy_half)
    # blind_shot_targets = np.zeros(MAP_SHAPE2, dtype=bool)  # disable blind shot
    self.blind_shot_targets = blind_shot_targets

<<<<<<< HEAD
    # unit_positions_ext1 = maximum_filter(mm.unit_positions, size=3)
    # updated by each unit's last target cell
    last_step_unit_target_mask = np.zeros(MAP_SHAPE2, dtype=bool)

    def next_by_team_units(upos, energy, cpos):
      # Do not penality cell in the relic nb
      is_relic_nb = mm.is_relic_neighbour[cpos[0]][cpos[1]]
      if is_relic_nb:
        return 0
=======
    unit_positions_ext1 = maximum_filter(mm.unit_positions, size=3)

    def next_by_team_units(upos, energy, cpos):
      # TODO: use energy to determine the ownership
      if not pos_equal(upos, cpos) and unit_positions_ext1[cpos[0]][cpos[1]]:
        return -5
      return 0

    def stay_on_relic(upos, energy, cpos, is_shadow_position, unit_id):
      # If the relic node has been occupied by unit but not this one, lower its score
      # if (mm.unit_positions[cpos[0]][cpos[1]] and not pos_equal(upos, cpos)):
      # return 0

      # For a relic node with units on it.
      uc = mm.unit_count[cpos[0]][cpos[1]]

      # if mm.game_step == 53 and unit_id == 10 and pos_equal(cpos, (19, 9)):
      # print(
      # f" stay_on_relic: upos={upos}, cpos={cpos} uc={uc}, is_shadow_position={is_shadow_position}",
      # file=sys.stderr)

      if not is_shadow_position:
        # Single unit on relic, but not this one, return
        if uc == 1 and not pos_equal(upos, cpos):
          return 0

        # For mutiple units on relic, not the min energy unit
        if (uc > 1 and pos_equal(upos, cpos)
            and mm.min_energy_unit_id[cpos[0]][cpos[1]] != unit_id):
          return 0
      else:
        # Do not call unit from other relic position
        unit_on_relic = mm.team_point_mass[pos[0]][pos[1]] > IS_RELIC_CELL_PROB
        if unit_on_relic:
          return 0

        # For shadow position, if unit has enough energy or more than what's on
        # the relic, then skip
        assert uc == 1
        uc_energy = mm.unit_min_energy[cpos[0]][cpos[1]]
        if (energy > uc_energy or energy >= mm.unit_sap_cost):
          return 0
>>>>>>> c4d73b1 (sync agent.py)

      if last_step_unit_target_mask[cpos[0]][cpos[1]]:
        return -5
      return 0

    def stay_on_relic(upos, energy, cpos, is_shadow_position, unit_id):
      # If the relic node has been occupied by unit but not this one, lower its score
      # if (mm.unit_positions[cpos[0]][cpos[1]] and not pos_equal(upos, cpos)):
      # return 0

      # For a relic node with units on it.
      uc = mm.unit_count[cpos[0]][cpos[1]]

      # if mm.game_step == 53 and unit_id == 10 and pos_equal(cpos, (19, 9)):
      # print(
      # f" stay_on_relic: upos={upos}, cpos={cpos} uc={uc}, is_shadow_position={is_shadow_position}",
      # file=sys.stderr)

      if not is_shadow_position:
        # Single unit on relic, but not this one, return
        if uc >= 1 and not pos_equal(upos, cpos):
          return 0

        # For mutiple units on relic, not the min energy unit
        if (uc > 1 and pos_equal(upos, cpos)
            and mm.min_energy_unit_id[cpos[0]][cpos[1]] != unit_id):
          return 0
      else:
        # Do not call unit from other relic position
        unit_on_relic = mm.team_point_mass[pos[0]][pos[1]] > IS_RELIC_CELL_PROB
        if unit_on_relic:
          return 0

        # For shadow position, if unit has enough energy or more than what's on
        # the relic, then skip
        assert uc == 1
        uc_energy = mm.unit_min_energy[cpos[0]][cpos[1]]
        if (energy > uc_energy or energy >= mm.unit_sap_cost):
          return 0

      # # Relic unit do not change relic position
      # p_unit = mm.team_point_mass[upos[0]][upos[1]]
      # if (p_unit > IS_RELIC_CELL_PROB and not pos_equal(upos, cpos)):
      # return 0

      # Will score only for relic p > 0.8
      v = RELIC_SCORE
      p = mm.team_point_mass[cpos[0]][cpos[1]]

      # Do not goto enemy side if energy below threshold
      if on_enemy_side(cpos, mm.player_id, mm.use_mirror):
        v = mm.unit_sap_cost / 10

      # If enemy may sap it, lower its weight
      # if not pos_equal(cpos, upos) and enemy_sap_cost[cpos[0]][cpos[1]] > 0:
      # v = mm.unit_sap_cost / 10 * 0.5

      w = 0
      if p > IS_RELIC_CELL_PROB:
        w += v * p
      return w

    hit_factor = 10
    enemy_hit_map = self.get_sap_hit_map(hit_factor)
    enemy_hit_map[blind_shot_targets] += (mm.unit_sap_cost - 1
                                          )  # roughly adding 3 points
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

      # boost attacking enemy in the defense zone
      sap_in_defense_zone = sap_range & defense_zone
      if (sap_in_defense_zone.sum() > 0
          and enemy_hit_map[sap_in_defense_zone].max() > 0):
        h += self.mm.unit_sap_cost / hit_factor

      # sap if energy is large (and unit not on relic)
      # if self.mm.team_point_mass[pos[0]][pos[1]] < 0.6:
      # h *= max((energy / energy_threshold), 1)
      return h

    score_debug = {}

    enemy_max_energy = maximum_filter(mm.enemy_max_energy, size=5)
    self.enemy_max_energy = enemy_max_energy

    def get_unit_cell_wt(upos, energy, cpos, unit_cost_map, is_shadow_position,
                         unit_id):
      if cant_move_to(upos, cpos, mm):
        return -9999

      unit_on_relic = mm.team_point_mass[pos[0]][pos[1]] > IS_RELIC_CELL_PROB
      if (not unit_on_relic) and energy <= enemy_max_energy[cpos[0]][cpos[1]]:
        return -9999

      # mdist = manhatten_distance(upos, cpos) + 7
      mdist = dd(manhatten_distance(upos, cpos) + 1)
      wt = 0.0001

      energy_ratio = energy / energy_threshold

      expore_wt = get_explore_weight(upos, energy, cpos)

      fuel_wt = get_fuel_energy(upos, energy, cpos)

      relic_nb_wt = get_open_relic_nb(upos, energy, cpos)

      on_relic_wt = stay_on_relic(upos, energy, cpos, is_shadow_position,
                                  unit_id)

      sap_wt = get_sap_enemy_score(upos, energy, cpos)

      next_by_team_wt = 0
      # next_by_team_wt = next_by_team_units(upos, energy, cpos)

      # If unit do not have much energy for one sap attack
      if (not (unit_on_relic
               and on_team_side(upos, mm.player_id, mm.use_mirror))
          and self.enemy_sap_cost[cpos[0]][cpos[1]] >= energy):
        wt -= self.mm.unit_sap_cost / 10

      wt += (expore_wt + fuel_wt + relic_nb_wt + on_relic_wt + sap_wt +
             next_by_team_wt) / mdist

      dbg = {
          'explore_wt': expore_wt,
          'fuel_wt': fuel_wt,
          'relic_nb_wt': relic_nb_wt,
          'on_relic_wt': on_relic_wt,
          'sap_wt': sap_wt,
          'energy_ratio': energy_ratio,
          'wt': wt,
          'mdist': mdist,
      }
      score_debug[(tuple(upos), tuple(cpos))] = dbg

<<<<<<< HEAD
<<<<<<< HEAD
      # DEBUG1
      # if mm.game_step == 38 and unit_id in (9, 10, 11):
      # if mm.game_step == 74 and unit_id in (3, ):
      # if pos_equal(cpos, (0, 8)) or pos_equal(cpos, (9, 9)):
      # print(
      # f'step={mm.game_step} unit_id={unit_id}, upos={upos}, cpos={cpos} wt={wt} cpos_on_relic={mm.team_point_mass[cpos[0]][cpos[1]]}, score_debug = {dbg},',
      # file=sys.stderr)
=======
      if mm.game_step == 79 and unit_id in (7, 5):
        if pos_equal(cpos, (21, 9)) or pos_equal(cpos, (22, 10)):
          print(
              f'step={mm.game_step} unit_id={unit_id}, upos={upos}, cpos={cpos} wt={wt} cpos_on_relic={mm.team_point_mass[cpos[0]][cpos[1]]}, score_debug = {dbg},',
              file=sys.stderr)
>>>>>>> c4d73b1 (sync agent.py)
=======
      # if mm.game_step == 79 and unit_id in (7, 5):
      # if pos_equal(cpos, (21, 9)) or pos_equal(cpos, (22, 10)):
      # print(
      # f'step={mm.game_step} unit_id={unit_id}, upos={upos}, cpos={cpos} wt={wt} cpos_on_relic={mm.team_point_mass[cpos[0]][cpos[1]]}, score_debug = {dbg},',
      # file=sys.stderr)
>>>>>>> 0c88147 (mute agent log)

      # if USE_RANDOM:
      # wt += np.random.rand() / 1000
      return wt

    cell_index = list(range(N_CELLS))
    if USE_RANDOM:
      np.random.shuffle(cell_index)

    # Adding relic positions with large energy unit, try release it
<<<<<<< HEAD
=======
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.player_id, i, t=0)
      if not mask:
        continue
      has_enough_energy = (energy >= 100 or energy
                           >= (mm.unit_sap_cost * 2 + mm.unit_move_cost * 5))
      uc = mm.unit_count[pos[0]][pos[1]]
      unit_on_relic = mm.team_point_mass[pos[0]][pos[1]] > IS_RELIC_CELL_PROB
      if uc == 1 and has_enough_energy and unit_on_relic:
        cell_index.append(pos_to_cell_idx(pos))

    weights = np.ones((MAX_UNIT_NUM, len(cell_index))) * -9999

>>>>>>> c4d73b1 (sync agent.py)
    for i in range(MAX_UNIT_NUM):
      mask, pos, energy = self.mm.get_unit_info(self.mm.player_id, i, t=0)
      if not mask:
        continue
      has_enough_energy = (energy >= 100 or energy
                           >= (mm.unit_sap_cost * 2 + mm.unit_move_cost * 5))
      uc = mm.unit_count[pos[0]][pos[1]]
      unit_on_relic = mm.team_point_mass[pos[0]][pos[1]] > IS_RELIC_CELL_PROB
      if uc == 1 and has_enough_energy and unit_on_relic:
        cell_index.append(pos_to_cell_idx(pos))

<<<<<<< HEAD
    weights = np.ones((MAX_UNIT_NUM, len(cell_index))) * -9999

    ordered_unit_ids = list(range(MAX_UNIT_NUM))
    # ordered_unit_ids.sort(
    # key=lambda unit_id: (-self.unit_score[unit_id], unit_id))
    for i in ordered_unit_ids:
      unit_id = i
      mask, pos, energy = self.mm.get_unit_info(self.mm.player_id,
                                                unit_id,
                                                t=0)
      if not mask:
        continue

      # print(f' unit_id={i}, unit_score={self.unit_score[unit_id]}',
      # file=sys.stderr)
      # if self.unit_to_cell:
      # last_cpos = self.unit_to_cell.get(unit_id)
      # if last_cpos is not None:
      # set_value_by_range(last_step_unit_target_mask,
      # last_cpos,
      # d=1,
      # val=True)

=======
      unit_id = i
>>>>>>> c4d73b1 (sync agent.py)
      for cell_id, idx in enumerate(cell_index):
        target_cell_pos = cell_idx_to_pos(idx)
        is_shadow_position = cell_id >= N_CELLS
        weights[i, cell_id] = get_unit_cell_wt(
            pos,
            energy,
            target_cell_pos,
            unit_cost_map=None,
            is_shadow_position=is_shadow_position,
            unit_id=unit_id)

    unit_to_cell = {}
    unit_score = defaultdict(int)
    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    for unit_id, cell_id in zip(rows, cols):
      wt = weights[unit_id, cell_id]
      if wt < 1e-6:  # TODO: use 0?
        # if wt < -1:  # TODO: use 0?
        continue

      idx = cell_index[cell_id]
      cpos = cell_idx_to_pos(idx)
      unit_to_cell[unit_id] = cpos
      unit_score[unit_id] = wt
      if not SUBMIT_AGENT:
        _, upos, _ = self.mm.get_unit_info(self.mm.player_id, unit_id, t=0)
        wts = score_debug[(tuple(upos), tuple(cpos))]
        print(f" unit[{unit_id}]={upos} assgined to cell={cpos}, wts={wts}",
              file=sys.stderr)

    self.unit_to_cell = unit_to_cell
    self.unit_score = unit_score
    return unit_to_cell

  def compute_energy_cost_map(self,
                              target_pos,
                              asteriod_cost=20 * 25,
                              N=MAP_WIDTH * 2,
                              extra_step_cost=5,
                              enemy_cost=None):
    """Using `extra_step_cost` to control the balance between cost and path length."""
    mm = self.mm
    cost_map = np.full((MAP_WIDTH, MAP_HEIGHT), float(mm.unit_move_cost))

    # nebula energy reduction adds extra cost
    cost_map[mm.cell_type == CELL_NEBULA] += mm.nebula_energy_reduction

    # Add extra step cost for favouring shorter path
    cost_map += extra_step_cost
    cost_map -= mm.cell_energy

    # cell energy cost change the cost map but max at 0 to prevent from loop
    cost_map = np.maximum(cost_map, 1)

    # use a big value for asteriod
    cost_map[mm.cell_type == CELL_ASTERIOD] += asteriod_cost

    if enemy_cost is not None:
      cost_map += enemy_cost

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
          # f"game_step={mm.game_step}, unit={unit_id} t staction={ACTION_ID_TO_NAME[k]}, from={unit_pos} to {(nx, ny)} dir={DIRECTIONS[k]} cost={cost}",
<<<<<<< HEAD
          # f"game_step={mm.game_step}, unit={unit_id} action={ACTION_ID_TO_NAME[k]}, from={unit_pos} to {(nx, ny)} dir={DIRECTIONS[k]} cost={cost}",
=======
>>>>>>> c4d73b1 (sync agent.py)
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

      # if not SUBMIT_AGENT:
      # if self.player == PLAYER1:
      # print(
      # f"pid=[{self.mm.player}] game_step={mm.game_step} sending unit={i} pos={unit_pos} to cell={cell_pos}",
      # file=sys.stderr)

      # sap_dead_zone = self.enemy_sap_cost >= unit_energy
      # enemy_sap_cost = self.enemy_sap_cost.copy()
      # enemy_sap_cost[~sap_dead_zone] = 0

      enemy_pos_cost = self.get_enemy_max_energy_level(unit_energy)
      # enemy_cost = (enemy_pos_cost * 1000) + (enemy_sap_cost * 5)
      enemy_cost = (enemy_pos_cost * 1000)

      # try not move onto team units
      # up = self.mm.unit_positions
      # unit_positions_ext1 = maximum_filter(mm.unit_positions, size=3)
      # enemy_cost[unit_positions_ext1 > 0] += (self.mm.unit_move_cost * 10)

      energy_cost = self.compute_energy_cost_map(cell_pos,
                                                 enemy_cost=enemy_cost)
      self.energy_cost_ = energy_cost
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

      # Boost attacking enemy in my defense_zone
      is_in_defense_zone = self.defense_zone[cpos[0]][cpos[1]]
      if is_in_defense_zone:
        h += self.mm.unit_sap_cost
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
      if (enemy_energy[sap_mask & (enemy_energy >= 0)].sum() <= 0
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

  def mirror_action(self, unit_actions):
    for i in range(MAX_UNIT_NUM):
      a, x, y = unit_actions[i]

      if self.mm.use_mirror:
        a = MIRRORED_ACTION[a]
        x, y = -y, -x

      unit_actions[i][:] = (a, x, y)
    return unit_actions

  def act(self, step: int, raw_obs, remainingOverageTime: int = 60):
    """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
    # if not SUBMIT_AGENT:
    print(f"============ game step {self.mm.game_step + 1} ========== ",
          file=sys.stderr)
    self.mm.update(raw_obs, self.prev_model_action)
    self.mm.add_sap_locations(self.last_sap_locations)
    self.mm.remainingOverageTime = remainingOverageTime

    unit_to_cell = self.compute_unit_to_cell()
    unit_actions = self.encode_unit_actions(unit_to_cell)
    self.attack(unit_actions, unit_to_cell)
    self.prev_model_action = {UNITS_ACTION: unit_actions}

    # if self.mm.use_mirror:
    # unit_actions = self.mirror_action(unit_actions)
    return unit_actions

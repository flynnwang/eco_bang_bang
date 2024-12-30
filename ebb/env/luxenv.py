from collections import OrderedDict, deque, defaultdict

import gym
import numpy as np
from gym import spaces
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from random import randint

from .const import *

EXT_ACTION_SHAPE = (MAX_UNIT_NUM, MOVE_ACTION_NUM)

# Let's use move action only first
ACTION_SPACE = spaces.Dict({
    UNITS_ACTION:
    spaces.MultiDiscrete(np.zeros(MAX_UNIT_NUM, dtype=int) + MOVE_ACTION_NUM)
})

MAP_SHAPE = (1, MAP_WIDTH, MAP_HEIGHT)
MAP_SHAPE2 = (MAP_WIDTH, MAP_HEIGHT)

OB = OrderedDict([
    # Game params
    ('unit_move_cost', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('unit_sensor_range', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),

    # Time & Match
    ('game_step', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('match_step', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('units_team_points', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('enemy_team_points', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),

    # Map info
    ('cell_type', spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + N_CELL_TYPES)),
    ('visible', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('observed', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('visited', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('is_relic_node', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('is_relic_neighbour', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('team_point_prob', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('cell_energy', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('is_team_born_cell', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
])


# Unit info
def _add_unit_info(prefix, i, max_time):
  for t in range(max_time):
    OB[f'{prefix}_{i}_loc_{t}'] = spaces.Box(low=0, high=1, shape=MAP_SHAPE)
    OB[f'{prefix}_{i}_energy_{t}'] = spaces.Box(low=0, high=1, shape=MAP_SHAPE)


def _add_all_unit_info(max_time=1):
  for i in range(MAX_UNIT_NUM):
    _add_unit_info('unit', i, max_time)
  for i in range(MAX_UNIT_NUM):
    _add_unit_info('enemy', i, max_time)


_add_all_unit_info()
OBSERVATION_SPACE = spaces.Dict(OB)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


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

  MAX_PAST_OB_NUM = 1

  def __init__(self, player, env_cfg):
    self.player_id = int(player[-1])
    self.player = player
    self.env_cfg = env_cfg
    self.cell_type = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.visible = None
    self.observed = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.last_observed_num = 0
    self.visited = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.is_relic_node = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.last_relic_node_num = 0
    self.is_relic_neighbour = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.last_relic_nb_visited = 0

    self.prev_team_point = 0
    self.team_point_mass = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.float32)

    self.cell_energy = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.game_step = 0
    self.match_step = 0

    self.past_obs = deque([])

  @property
  def enemy_id(self):
    return 1 - self.player_id

  @property
  def unit_move_cost(self):
    return self.env_cfg['unit_move_cost']

  @property
  def unit_sensor_range(self):
    return self.env_cfg['unit_sensor_range']

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
    self.visible = ob['sensor_mask'].astype(np.int32)
    self.last_observed_num = self.observed.sum()
    self.observed |= self.visible
    self.observed |= anti_diag_sym(self.visible)

  @property
  def step_new_observed_num(self):
    return self.observed.sum() - self.last_observed_num

  def update(self, ob):
    # Match restarted
    if ob['match_steps'] == 0:
      self.prev_team_point = 0

    self.game_step = ob['steps']
    self.match_step = ob['match_steps']

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
    if len(self.past_obs) > self.MAX_PAST_OB_NUM:
      self.past_obs.pop()

  def get_unit_info(self, pid, i, t):
    ob = self.past_obs[t]
    mask = ob['units_mask'][pid][i]
    position = ob['units']['position'][pid][i]
    energy = ob['units']['energy'][pid][i]
    return mask, position, energy

  def get_visited_relic_nb_num(self):
    sym_visited = self.visited | anti_diag_sym(self.visited)
    return ((sym_visited > 0) & (self.is_relic_neighbour > 0)).sum()

  @property
  def step_new_visited_relic_nb_num(self):
    return self.get_visited_relic_nb_num() - self.last_relic_nb_visited

  def update_visited_node(self, unit_positions, ob):
    self.last_relic_nb_visited = self.get_visited_relic_nb_num()
    self.visited[unit_positions[:, 0], unit_positions[:, 1]] = 1

  @property
  def step_new_found_relic_node_num(self):
    return self.is_relic_node.sum() - self.last_relic_node_num

  def update_relic_node(self, ob):
    self.last_relic_node_num = self.is_relic_node.sum()

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

    change = 30
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


def gen_dummy_action():
  return np.zeros((MAX_UNIT_NUM, 3), np.int32)


class LuxS3Env(gym.Env):

  def __init__(self, reward_schema=None, game_env=None):
    self.reward_schema = reward_schema
    self.game = game_env or LuxAIS3GymEnv(numpy_output=True)
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

  def _update_mms(self, obs):
    self.mms[0].update(obs[PLAYER0])
    self.mms[1].update(obs[PLAYER1])

  def reset(self, seed=None):
    if seed is None:
      seed = randint(-(1 << 31), 1 << 31)
    if self._seed is not None:
      seed = self._seed
    raw_obs, info = self.game.reset(seed=seed)

    env_cfg = info['params']
    self.mms = [MapManager(PLAYER0, env_cfg), MapManager(PLAYER1, env_cfg)]
    self._update_mms(raw_obs)

    self.prev_raw_obs = raw_obs
    done = False
    reward = self._convert_reward(raw_obs, info)
    action = {PLAYER0: gen_dummy_action(), PLAYER1: gen_dummy_action()}
    self._actions_taken_mask = [{
        UNITS_ACTION:
        np.zeros(EXT_ACTION_SHAPE, dtype=bool)
    }, {
        UNITS_ACTION:
        np.zeros(EXT_ACTION_SHAPE, dtype=bool)
    }]

    info = self.get_info(action, raw_obs, reward, model_action=None)

    return self.observation(raw_obs), reward, done, info

  def _encode_action(self, action):
    """Translate the model action into game env action.

    TODO: to encode SAP action, prev observation is required.
    """
    action = action[UNITS_ACTION]
    unit_actions = np.zeros((MAX_UNIT_NUM, 3), dtype=np.int32)
    for i, a in enumerate(action):
      unit_actions[i][0] = np.int32(a)
    return unit_actions

  def compute_actions_taken(self, model_actions):
    return [
        self.get_actions_taken_mask(model_actions[i], self.mms[i])
        for i, player in enumerate([PLAYER0, PLAYER1])
    ]

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
        PLAYER0: self._encode_action(model_action[0]),
        PLAYER1: self._encode_action(model_action[1]),
    }
    raw_obs, step_reward, terminated, truncated, info = self.game.step(action)
    self._update_mms(raw_obs)

    done = False
    if raw_obs[PLAYER0]['steps'] >= MAX_GAME_STEPS:
      done = True

    obs = self.observation(raw_obs)
    reward = self._convert_reward(raw_obs, info)
    info = self.get_info(action, raw_obs, reward, model_action, done)

    self.prev_raw_obs = raw_obs
    return obs, reward, done, info

  def _convert_observation(self, ob, mm):
    """Construct all features using MAP_SHAPE2."""
    o = {}

    def scalar(v, maxv):
      return np.zeros(MAP_SHAPE2) + (v / maxv)

    # Game params
    o['unit_move_cost'] = scalar(mm.unit_move_cost, MAX_MOVE_COST)
    o['unit_sensor_range'] = scalar(mm.unit_sensor_range, MAX_SENSOR_RANGE)

    # Time & Match
    o['game_step'] = scalar(mm.game_step, MAX_GAME_STEPS)
    o['match_step'] = scalar(mm.match_step, MAX_MATCH_STEPS)

    team_points = ob['team_points']
    units_points = team_points[mm.player_id]
    enemy_points = team_points[mm.enemy_id]
    o['units_team_points'] = scalar(units_points, TEAM_POINTS_NORM)
    o['enemy_team_points'] = scalar(enemy_points, TEAM_POINTS_NORM)

    # Map info
    o['cell_type'] = mm.cell_type.copy()
    o['visible'] = mm.visible.astype(np.float32)
    o['observed'] = mm.observed.astype(np.float32)
    o['visited'] = mm.visited.astype(np.float32)
    o['is_relic_node'] = mm.is_relic_node.astype(np.float32)
    o['is_relic_neighbour'] = mm.is_relic_neighbour.astype(np.float32)

    team_point_prob = mm.team_point_mass.copy()
    team_point_prob[mm.is_relic_neighbour == 0] = NON_TEAM_POINT_MASS
    o['team_point_prob'] = sigmoid(team_point_prob / 20)
    o['cell_energy'] = mm.cell_energy / MAX_ENERTY_PER_TILE

    team_born_cell = np.zeros(MAP_SHAPE2)
    if mm.player_id == 0:
      team_born_cell[0][0] = 1
    else:
      team_born_cell[MAP_WIDTH - 1][MAP_HEIGHT - 1] = 1
    o['is_team_born_cell'] = team_born_cell

    def add_unit_feature(prefix, player_id, i, t):
      mask, pos, energy = mm.get_unit_info(player_id, i, t)
      unit_pos = np.zeros(MAP_SHAPE2)
      unit_energy = np.zeros(MAP_SHAPE2)
      if mask:
        # pid=1, pos=15, 8; energy=-12, mask=True
        # assert MAP_WIDTH > pos[
        # 0] >= 0, f"pid={player_id}, pos={pos[0]}, {pos[1]}; energy={energy}, mask={mask}"
        # assert MAP_HEIGHT > pos[
        # 1] >= 0, f"pid={player_id}, pos={pos[0]}, {pos[1]}; energy={energy}, mask={mask}"
        # assert energy >= 0, f"step={mm.game_step}, pid={player_id}, pos={pos[0]}, {pos[1]}; energy={energy}, mask={mask}"
        # Why energy is negative
        unit_pos[pos[0]][pos[1]] = 1
        unit_energy[pos[0]][pos[1]] = energy / MAX_UNIT_ENERGY

      o[f'{prefix}_{i}_loc_{t}'] = unit_pos
      o[f'{prefix}_{i}_energy_{t}'] = unit_energy

    # Unit info
    for i in range(MAX_UNIT_NUM):
      add_unit_feature('unit', mm.player_id, i, t=0)

    for i in range(MAX_UNIT_NUM):
      add_unit_feature('enemy', mm.enemy_id, i, t=0)

    assert len(o) == len(OB), f"len(o)={len(o)}, len(OB)={len(OB)}"
    # expand all feature map with dummy dim 1
    o = {k: np.expand_dims(v, 0) for k, v in o.items()}
    # for k, v in o.items():
    # print(k, v.shape)
    return o

  def observation(self, raw_obs):
    assert len(
        raw_obs
    ) == 2, f"len(raw_obs)={len(raw_obs)}, self.total_agent_controls={self.total_agent_controls}"
    if SINGLE_PLAER:
      return [self._convert_observation(raw_obs[PLAYER0],
                                        self.mms[0])]  # single player
    else:
      return [
          self._convert_observation(raw_obs[PLAYER0], self.mms[0]),
          self._convert_observation(raw_obs[PLAYER1], self.mms[1])
      ]

  def _convert_exploration_reward(self, raw_obs):

    def _convert(mm, ob):
      r = 0

      # reward for open unobserved cells
      r_explore = mm.step_new_observed_num * 0.01

      # reward for newly found relic node
      r_find_relic = mm.step_new_found_relic_node_num * 0.5

      # reward for each new visited relic nb
      r_visit_relic_nb = mm.step_new_visited_relic_nb_num * 0.1

      r_match = 0
      if mm.match_step == MAX_MATCH_STEPS:
        team_points = raw_obs[mm.player]['team_points'][mm.player_id]
        r_match = team_points * 0.001

      r = r_explore + r_find_relic + r_visit_relic_nb + r_match
      # print(
      # f'step={mm.game_step} match-step={mm.match_step}, explore={r_explore:.3f} '
      # f'find_relic={r_find_relic:.3f}, visit_relc_nb={r_visit_relic_nb:.3f} match={r_match:.3f}'
      # )
      return r

    return [
        _convert(self.mms[i], raw_obs[p])
        for i, p in enumerate([PLAYER0, PLAYER1])
    ]
    pass

  def _convert_reward(self, raw_obs, info):
    """Use the match win-loss reward for now."""
    assert self.reward_schema in ('match_win_loss',
                                  'relic_boosted_match_score',
                                  'exploration_reward')

    team_wins = raw_obs[PLAYER0]['team_wins']
    prev_team_wins = self.prev_raw_obs[PLAYER0]['team_wins']
    diff = team_wins - prev_team_wins
    max_v = max(diff[0], diff[1])

    reward = [0, 0]
    if max_v > 0:
      if diff[0] > diff[1]:
        reward = [1, -1]
      if diff[1] >= diff[0]:
        reward = [-1, 1]

    mm = self.mms[0]
    if self.reward_schema == 'relic_boosted_match_score' and mm.match_step > 0:
      team_points = raw_obs[PLAYER0]['team_points']
      prev_team_points = self.prev_raw_obs[PLAYER0]['team_points']
      pdiff = team_points - prev_team_points
      for i in range(2):
        reward[i] += pdiff[i] / 500

    if self.reward_schema == 'exploration_reward':
      reward = self._convert_exploration_reward(raw_obs)

    # print(
    # f'step={mm.game_step}, match_step={mm.match_step}, r0={reward[0]}, r1={reward[1]}'
    # )
    if SINGLE_PLAER:
      return [reward[0]]  # single player
    else:
      return reward

  def _get_available_action_mask(self, mm):
    """Mask for unit action: compute available action based on unit position"""
    actions_mask = np.zeros(EXT_ACTION_SHAPE, dtype=bool)
    for i in range(MAX_UNIT_NUM):

      # TODO: when use unit position inference, update here
      unit_mask, pos, energy = mm.get_unit_info(mm.player_id, i, t=0)
      if not unit_mask:
        continue

      # If units has run out of energy, if can only move_center
      if energy < mm.unit_move_cost:
        continue

      actions_mask[i][ACTION_CENTER] = 1  # can always stay
      for k in range(1, MOVE_ACTION_NUM):
        nx, ny = (pos[0] + DIRECTIONS[k][0], pos[1] + DIRECTIONS[k][1])
        if nx < 0 or nx >= MAP_WIDTH:
          continue
        if ny < 0 or ny >= MAP_HEIGHT:
          continue
        if mm.cell_type[nx][ny] == CELL_ASTERIOD:
          continue
        actions_mask[i][k] = 1

    return {UNITS_ACTION: actions_mask}

  def get_actions_taken_mask(self, model_action, mm):
    """Should ignore all the actions that can not be performed. Compute this
    before env.step() to make use of mm from prev step."""
    mask = np.zeros(EXT_ACTION_SHAPE, dtype=bool)
    units_action = model_action[UNITS_ACTION]
    for i, a in enumerate(units_action):
      unit_mask, pos, energy = mm.get_unit_info(mm.player_id, i, t=0)
      if not unit_mask:
        continue

      # If units has run out of energy, if can only move_center
      if energy < mm.unit_move_cost:
        continue

      mask[i][a] = 1
    return {UNITS_ACTION: mask}

  def get_info(self, action, raw_obs, reward, model_action, done=False):

    def count_actions(info, agent_action):
      action_count = {a: 0 for a in ACTION_ID_TO_NAME.values()}
      for i in range(MAX_UNIT_NUM):
        a = agent_action[i][0]
        name = ACTION_ID_TO_NAME[a]
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

    def _get_info(agent_action, raw_obs1, prev_obs1, agent_reward,
                  model_action, mm):
      info = {}

      info['actions_taken_mask'] = self._actions_taken_mask[mm.player_id]

      # action mask for current state, (for sample action)
      info['available_action_mask'] = self._get_available_action_mask(mm)

      step_reward = reward[mm.player_id]
      info['_step_reward'] = step_reward
      info['player'] = mm.player_id  # for testing

      # Team points stats
      tp0 = raw_obs1['team_points'][mm.player_id]
      tp1 = prev_obs1['team_points'][mm.player_id]
      # print(
      # f"step={raw_obs[PLAYER0]['steps']}, match_steps={match_step} done={done}, player_id={mm.player_id} team_point={tp0}"
      # )
      info['_step_team_points'] = max(tp0 - tp1, 0)

      info['_match_team_points'] = 0
      info['_match_played'] = 0
      match_step = raw_obs[PLAYER0]['match_steps']
      if match_step == MAX_MATCH_STEPS:
        info['_match_team_points'] = tp0
        info['_match_played'] = 1

      step = raw_obs[PLAYER0]['steps']
      # print(f"step={step} match_step={match_step}, step_reward={step_reward}")
      count_actions(info, agent_action)
      add_unit_total_energy(info, mm)
      return info

    if model_action is None:
      model_action = [None, None]

    if SINGLE_PLAER:
      return [
          _get_info(action[PLAYER0], raw_obs[PLAYER0],
                    self.prev_raw_obs[PLAYER0], model_action[0], self.mms[0])
      ]  # single player
    else:
      return [
          _get_info(action[player], raw_obs[player], self.prev_raw_obs[player],
                    reward[i], model_action[i], self.mms[i])
          for i, player in enumerate([PLAYER0, PLAYER1])
      ]

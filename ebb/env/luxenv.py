from collections import OrderedDict, deque, defaultdict

import gym
import numpy as np
from gym import spaces
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from .const import *

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

    # Map info
    ('cell_type', spaces.MultiDiscrete(np.zeros(MAP_SHAPE) + N_CELL_TYPES)),
    ('visible', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('observed', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('visited', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('is_relic_node', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('is_relic_neighbour', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('team_point_prob', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
    ('cell_energy', spaces.Box(low=0, high=1, shape=MAP_SHAPE)),
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
    self.env_cfg = env_cfg
    self.cell_type = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.visible = None
    self.observed = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.visited = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.is_relic_node = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)
    self.is_relic_neighbour = np.zeros((MAP_WIDTH, MAP_HEIGHT), np.int32)

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

  def update(self, ob):
    # Match restarted
    if ob['match_steps'] == 0:
      self.prev_team_point = 0

    self.game_step = ob['steps']
    self.match_step = ob['match_steps']

    self.visible = ob['sensor_mask'].astype(np.int32)
    self.observed |= self.visible

    self.update_cell_type(ob)

    unit_masks = ob['units_mask'][self.player_id]
    unit_positions = ob['units']['position'][self.player_id][unit_masks]
    self.visited[unit_positions[:, 0], unit_positions[:, 1]] = 1

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

    must_be_team_point = (self.team_point_mass
                          >= TEAM_POINT_MASS) & (unit_nearby_relic)
    non_team_point = (self.team_point_mass
                      <= NON_TEAM_POINT_MASS) & (unit_nearby_relic)

    delta = team_point - self.prev_team_point
    delta -= must_be_team_point.sum()

    team_point_candidate = unit_nearby_relic & (~must_be_team_point) & (
        ~non_team_point)
    num = team_point_candidate.sum()
    if num > 0:
      if delta == 0:
        self.team_point_mass[team_point_candidate] = NON_TEAM_POINT_MASS
        self.team_point_mass[anti_diag_sym(
            team_point_candidate)] = NON_TEAM_POINT_MASS
      elif num == delta:
        self.team_point_mass[team_point_candidate] = TEAM_POINT_MASS
        self.team_point_mass[anti_diag_sym(
            team_point_candidate)] = TEAM_POINT_MASS
      else:
        assert delta < num
        # print('>>>>>>>>>>>>>>', ob['steps'], delta, num, must_be_team_point.sum(), non_team_point.sum())
        self.team_point_mass[anti_diag_sym(
            team_point_candidate)] += delta / num
      assert delta >= 0


def gen_dummy_action():
  return np.zeros((MAX_UNIT_NUM, 3), np.int32)


class LuxS3Env(gym.Env):

  def __init__(self, reward_schema=None):
    self.reward_schema = reward_schema
    self.game = LuxAIS3GymEnv(numpy_output=True)
    self.mms = None
    self.prev_raw_obs = None

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
    pass

  def _update_mms(self, obs):
    self.mms[0].update(obs[PLAYER0])
    self.mms[1].update(obs[PLAYER1])

  def reset(self, seed=None):
    raw_obs, info = self.game.reset()

    env_cfg = info['params']
    self.mms = [MapManager(PLAYER0, env_cfg), MapManager(PLAYER1, env_cfg)]
    self._update_mms(raw_obs)

    self.prev_raw_obs = raw_obs
    done = False
    reward = self._convert_reward(raw_obs, info)
    action = {PLAYER0: gen_dummy_action(), PLAYER1: gen_dummy_action()}
    info = self.get_info(action, raw_obs, model_action=None)

    return self.observation(raw_obs), reward, done, info

  def _encode_action(self, action):
    """Translate the model action into game env action.

    TODO: to encode SAP action, prev observation is required.
    """
    unit_actions = np.zeros((MAX_UNIT_NUM, 3), dtype=np.int32)
    for i, a in enumerate(action):
      unit_actions[i][0] = np.int32(a)
    return unit_actions

  def step(self, model_action):
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
    info = self.get_info(action, raw_obs, model_action)

    self.prev_raw_obs = raw_obs
    return obs, reward, done, info

  def _convert_observation(self, ob, mm):
    """Construct all features using MAP_SHAPE2."""
    o = {}

    def scalar(v, maxv):
      return (np.zeros(MAP_SHAPE2) + v) / maxv

    # Game params
    o['unit_move_cost'] = scalar(mm.unit_move_cost, MAX_MOVE_COST)
    o['unit_sensor_range'] = scalar(mm.unit_sensor_range, MAX_SENSOR_RANGE)

    # Time & Match
    o['game_step'] = scalar(mm.game_step, MAX_GAME_STEPS)
    o['match_step'] = scalar(mm.game_step, MAX_MATCH_STEPS)

    # Map info
    o['cell_type'] = mm.cell_type.copy()
    o['visible'] = mm.visible.astype(np.float32)
    o['observed'] = mm.observed.astype(np.float32)
    o['visited'] = mm.visited.astype(np.float32)
    o['is_relic_node'] = mm.is_relic_node.astype(np.float32)
    o['is_relic_neighbour'] = mm.is_relic_neighbour.astype(np.float32)
    o['team_point_prob'] = sigmoid(mm.team_point_mass / 20)
    o['cell_energy'] = mm.cell_energy / MAX_ENERTY_PER_TILE

    def add_unit_feature(prefix, player_id, i, t):
      mask, pos, energy = mm.get_unit_info(player_id, i, t)
      unit_pos = np.zeros(MAP_SHAPE2)
      unit_energy = np.zeros(MAP_SHAPE2)
      if mask:
        unit_pos[pos] = 1
        unit_energy[pos] = energy / MAX_UNIT_ENERGY

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
    for k, v in o.items():
      print(k, v.shape)
    return o

  def observation(self, raw_obs):
    assert len(
        raw_obs
    ) == 2, f"len(raw_obs)={len(raw_obs)}, self.total_agent_controls={self.total_agent_controls}"
    return [
        self._convert_observation(raw_obs[PLAYER0], self.mms[0]),
        self._convert_observation(raw_obs[PLAYER1], self.mms[1])
    ]

  def _convert_reward(self, raw_obs, info):
    """Use the match win-loss reward for now."""
    assert self.reward_schema == 'match_win_loss'

    team_wins = raw_obs[PLAYER0]['team_wins']
    prev_team_wins = self.prev_raw_obs[PLAYER0]['team_wins']
    reward = team_wins - prev_team_wins
    return reward

  def _get_available_action_mask(self, mm):
    """Mask for unit action: compute available action based on unit position"""
    actions_mask = np.zeros((MAX_UNIT_NUM, MOVE_ACTION_NUM), np.int32)
    for i in range(MAX_UNIT_NUM):
      actions_mask[i][ACTION_CENTER] = 1  # can always stay
      unit_mask, pos, _ = mm.get_unit_info(mm.player_id, i, t=0)

      # TODO: when use unit position inference, update here
      if not unit_mask:
        continue

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

  def get_actions_taken_mask(self, model_action):
    mask = np.zeros((MAX_UNIT_NUM, MOVE_ACTION_NUM), np.int32)
    if model_action is None:
      mask[:, 0] = 1
      return {UNITS_ACTION: mask}

    for i, a in enumerate(model_action):
      mask[i][a] = 1
    return {UNITS_ACTION: mask}

  def get_info(self, action, raw_obs, model_action):

    def _info(agent_action, raw_obs1, prev_obs1, model_action, mm):
      info = {}

      # action mask matches with given action for last state (for compute logits)
      info['actions_taken_mask'] = self.get_actions_taken_mask(model_action)

      # action mask for current state, (for sample action)
      info['available_action_mask'] = self._get_available_action_mask(mm)

      return info

    if model_action is None:
      model_action = [None, None]

    return [
        _info(action[player], raw_obs[player], self.prev_raw_obs[player],
              model_action[i], self.mms[i])
        for i, player in enumerate([PLAYER0, PLAYER1])
    ]

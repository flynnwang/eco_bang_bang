import sys

import cProfile

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from ebb.env.luxenv import get_ob_sapce, LuxS3Env
from ebb.model import RewardSpec
from ebb.env.const import *

from ebb.env.mapmanager import *

sys.path.insert(
    0, "/Users/flynnwang/repo/flynn/eco_bang_bang/agents/python_starter")
from agent import Agent

reward_spec = RewardSpec(
    reward_min=-5,
    reward_max=+5,
    zero_sum=True,
)

reward_shaping_params = dict(
    new_observed_down_tri=0.0001,
    match_observed=0,
    match_win=1,
    match_result=1,
    use_hidden_relic_estimator=True,
    energy_cost_change=0.001,
)

obs_space_kwargs = {
    'use_energy_cost_map': True,
    'use_single_player': False,
}


def act(a, ob):
  player_id = a.player
  step = ob['steps']
  actions = a.act(step, ob)
  return actions


profiler = cProfile.Profile()

for _ in range(5):
  env = LuxS3Env(reward_schema='match_win_loss',
                 reward_shaping_params=reward_shaping_params,
                 obs_space_kwargs=obs_space_kwargs)

  obs, reward, done, info = env.reset()
  env_cfg = env.mms[0].env_cfg
  a0 = Agent('player_0', env_cfg)
  a1 = Agent('player_1', env_cfg)

  sap_indexer = SapIndexer()
  mm = MapManager(PLAYER0,
                  env_cfg,
                  transpose=False,
                  sap_indexer=sap_indexer,
                  use_mirror=False,
                  use_hidden_relic_estimator=True,
                  full_params=env.mms[0].full_params)

  while not done:
    if env.mms[0].game_step % 50 == 0:
      print(_, env.mms[0].game_step)
    action0 = env.agent_action_do_model_action(act(a0, env.mms[0].past_obs[0]),
                                               env.mms[0])
    action1 = env.agent_action_do_model_action(act(a1, env.mms[1].past_obs[0]),
                                               env.mms[1])
    player_actions = [{UNITS_ACTION: action0}, {UNITS_ACTION: action1}]
    obs, reward, done, info = env.step(player_actions)

    profiler.enable()
    mm.update(env.mms[0].past_obs[0], {UNITS_ACTION: action0})
    profiler.disable()

profiler.dump_stats(f"profile_run.prof")

import torch


def split_env_output_by_player(env_out, obs_per_player=1):
  if isinstance(env_out, dict):
    left_team, right_team = {}, {}
    for key, val in env_out.items():
      left_val, right_val = split_env_output_by_player(val)
      left_team[key] = left_val
      right_team[key] = right_val
    return left_team, right_team
  else:
    # torch tensor
    sz0 = env_out.shape[0]
    sz1p = env_out.shape[1:]
    per_team = env_out.view(sz0 // obs_per_player, obs_per_player, *sz1p)
    left_team = per_team[::2, :].reshape(sz0 // 2, *sz1p).to(env_out.device,
                                                             non_blocking=True)
    right_team = per_team[1::2, :].reshape(sz0 // 2,
                                           *sz1p).to(env_out.device,
                                                     non_blocking=True)
    return left_team, right_team


def pair_env_output_for_players(left_out, right_out, obs_per_player=1):
  if isinstance(left_out, dict):
    env_out = {}
    for key, left_val in left_out.items():
      right_val = right_out[key]
      env_out[key] = pair_env_output_for_players(left_val, right_val,
                                                 obs_per_player)
    return env_out
  else:
    sz0 = left_out.shape[0]
    sz1p = left_out.shape[1:]

    left_shaped = left_out.view(sz0 // obs_per_player, obs_per_player, *sz1p)
    right_shaped = right_out.view(sz0 // obs_per_player, obs_per_player, *sz1p)

    paired = torch.empty(left_shaped.shape[0] * 2,
                         *left_shaped.shape[1:],
                         dtype=left_shaped.dtype)
    paired[::2] = left_shaped
    paired[1::2] = right_shaped
    paired = paired.view(-1, *paired.shape[2:])
    return paired

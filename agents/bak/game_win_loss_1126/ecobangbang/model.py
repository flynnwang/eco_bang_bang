import logging

from typing import Callable, Dict, Optional, Tuple, Union, NamedTuple, Any

import gym
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .env.const import *
from .env.luxenv import ACTION_SPACE


class SqueezeExcitation(torch.nn.Module):
  """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

  def __init__(
      self,
      input_channels: int,
      squeeze_channels: int,
      activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
      scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
  ) -> None:
    super().__init__()
    self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
    self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
    self.activation = activation()
    self.scale_activation = scale_activation()

  def _scale(self, input: Tensor) -> Tensor:
    scale = self.avgpool(input)
    scale = self.fc1(scale)
    scale = self.activation(scale)
    scale = self.fc2(scale)
    return self.scale_activation(scale)

  def forward(self, input: Tensor) -> Tensor:
    scale = self._scale(input)
    return scale * input


class RewardSpec(NamedTuple):
  reward_min: float
  reward_max: float
  zero_sum: bool


def _index_select(embedding_layer: nn.Embedding,
                  x: torch.Tensor) -> torch.Tensor:
  out = embedding_layer.weight.index_select(0, x.reshape(-1))
  # out = embedding_layer.weight.index_select(0, x.view(-1))
  return out.view(*x.shape, -1)


def _forward_select(embedding_layer: nn.Embedding,
                    x: torch.Tensor) -> torch.Tensor:
  return embedding_layer(x)


def _get_select_func(use_index_select: bool) -> Callable:
  """
    Use index select instead of default forward to possibly speed up embedding.
    NB: This disables padding_idx functionality
    """
  if use_index_select:
    return _index_select
  else:
    return _forward_select


class DictInputLayer(nn.Module):

  @staticmethod
  def forward(
      x: Dict[str, Union[Dict, torch.Tensor]]
  ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    return x["obs"], x["info"]["available_action_mask"]


class ConvEmbeddingInputLayer(nn.Module):

  def __init__(self,
               obs_space: gym.spaces.Dict,
               embedding_dim: int,
               out_dim: int,
               use_index_select: bool = True,
               activation: Callable = nn.LeakyReLU):
    super(ConvEmbeddingInputLayer, self).__init__()

    embeddings = {}
    n_continuous_channels = 0
    n_embedding_channels = 0
    self.keys_to_op = {}
    for key, val in obs_space.spaces.items():
      # Skips hidden obs from conv layer
      if key.startswith('_'):
        continue

      assert val.shape == (1, MAP_WIDTH, MAP_HEIGHT), f"{key}={val.shape}"
      if isinstance(val, gym.spaces.MultiBinary) or isinstance(
          val, gym.spaces.MultiDiscrete):
        if isinstance(val, gym.spaces.MultiBinary):
          n_embeddings = 2
          padding_idx = 0
        elif isinstance(val, gym.spaces.MultiDiscrete):
          if val.nvec.min() != val.nvec.max():
            raise ValueError(
                f"MultiDiscrete observation spaces must all have the same number of embeddings. "
                f"Found: {np.unique(val.nvec)}")
          n_embeddings = val.nvec.ravel()[0]
          padding_idx = None
          # print(f'add n embeddings for key={key}, n_embeddings={n_embeddings}')
        else:
          raise NotImplementedError(f"Got gym space: {type(val)}")
        embeddings[key] = nn.Embedding(n_embeddings,
                                       embedding_dim,
                                       padding_idx=padding_idx)
        n_embedding_channels += embedding_dim
        self.keys_to_op[key] = "embedding"
      elif isinstance(val, gym.spaces.Box):
        n_continuous_channels += 1  # assuming all elements having the same meaning
        self.keys_to_op[key] = "continuous"
      else:
        raise NotImplementedError(
            f"{type(val)} is not an accepted observation space.")

    self.embeddings = nn.ModuleDict(embeddings)
    continuous_space_embedding_layers = []
    embedding_merger_layers = []
    merger_layers = []
    # logging.info(
    # f'n_continuous_channels={n_continuous_channels}, n_embedding_channels={n_embedding_channels}'
    # )

    continuous_space_embedding_layers.extend(
        [nn.Conv2d(n_continuous_channels, out_dim, (1, 1)),
         activation()])
    embedding_merger_layers.extend(
        [nn.Conv2d(n_embedding_channels, out_dim, (1, 1)),
         activation()])
    merger_layers.append(nn.Conv2d(out_dim * 2, out_dim, (1, 1)))

    self.continuous_space_embedding = nn.Sequential(
        *continuous_space_embedding_layers)
    self.embedding_merger = nn.Sequential(*embedding_merger_layers)
    self.merger = nn.Sequential(*merger_layers)
    self.select = _get_select_func(use_index_select)

  def forward(self, xx: Dict[str, torch.Tensor]) -> torch.Tensor:
    continuous_outs = []
    embedding_outs = {}
    for key, op in self.keys_to_op.items():
      in_tensor = xx[key]
      if op == "embedding":
        # out=(b, 1, x, y, n_embeddings)
        # drop 1, it's useless
        out = self.select(self.embeddings[key], in_tensor)

        # move channel into second column.
        # print(key, out.shape)
        # b, _, x, y, embedding_dim = out.shape
        # out = out.squeeze(1).permute(0, 3, 1, 2)
        b, x, y, embedding_dim = out.shape
        out = out.permute(0, 3, 1, 2)
        assert len(
            out.shape
        ) == 4, f"Expect embedding to have 5 dims, get {len(out.shape)}: in_shape={in_tensor.shape}{out.shape}"
        embedding_outs[key] = out
      elif op == "continuous":
        # __import__('ipdb').set_trace()
        b, x, y = in_tensor.shape
        # b*p, 1, x, y; where 1 is a channel of dim 1
        out = in_tensor.view(b, x, y).unsqueeze(1)

        # out = in_tensor
        # print(key, out.shape)
        assert len(out.shape) == 4, (key, out.shape)
        continuous_outs.append(out)
        # print("contiguous , ", key, out.shape, in_tensor.shape)
      else:
        raise RuntimeError(f"Unknown operation: {op}")

    continuous_out_combined = self.continuous_space_embedding(
        torch.cat(continuous_outs, dim=1))
    embedding_outs_combined = self.embedding_merger(
        torch.cat([v for v in embedding_outs.values()], dim=1))

    # print('continuous_out_combined shape, ', continuous_out_combined.shape)
    # print('embedding_outs_combined shape, ', embedding_outs_combined.shape)
    # print('continuous_outs', continuous_outs)
    # print('embedding_outs', embedding_outs)
    # print('continuous_out_combined', continuous_out_combined)
    # print('embedding_outs_combined', embedding_outs_combined)
    merged_outs = self.merger(
        torch.cat([continuous_out_combined, embedding_outs_combined], dim=1))
    # print('merged_outs', merged_outs)
    return merged_outs


class ResidualBlock(nn.Module):

  def __init__(self,
               in_channels: int,
               out_channels: int,
               height: int,
               width: int,
               kernel_size: int = 3,
               normalize: bool = False,
               activation: Callable = nn.ReLU,
               squeeze_excitation: bool = True,
               rescale_se_input: bool = True,
               **conv2d_kwargs):
    super(ResidualBlock, self).__init__()

    # Calculate "same" padding
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    # https://www.wolframalpha.com/input/?i=i%3D%28i%2B2x-k-%28k-1%29%28d-1%29%2Fs%29+%2B+1&assumption=%22i%22+-%3E+%22Variable%22
    assert "padding" not in conv2d_kwargs.keys()
    k = kernel_size
    d = conv2d_kwargs.get("dilation", 1)
    s = conv2d_kwargs.get("stride", 1)
    padding = (k - 1) * (d + s - 1) / (2 * s)
    assert padding == int(
        padding), f"padding should be an integer, was {padding:.2f}"
    padding = int(padding)

    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=(kernel_size, kernel_size),
                           padding=(padding, padding),
                           **conv2d_kwargs)

    # We use LayerNorm here since the size of the input "images" may vary based on the board size
    self.norm1 = nn.LayerNorm([in_channels, height, width
                               ]) if normalize else nn.Identity()
    self.act1 = activation()

    self.conv2 = nn.Conv2d(in_channels=out_channels,
                           out_channels=out_channels,
                           kernel_size=(kernel_size, kernel_size),
                           padding=(padding, padding),
                           **conv2d_kwargs)
    self.norm2 = nn.LayerNorm([in_channels, height, width
                               ]) if normalize else nn.Identity()
    self.final_act = activation()

    if in_channels != out_channels:
      self.change_n_channels = nn.Conv2d(in_channels, out_channels, (1, 1))
    else:
      self.change_n_channels = nn.Identity()

    if squeeze_excitation:
      squeeze_channels = out_channels // 16
      self.squeeze_excitation = SqueezeExcitation(out_channels,
                                                  squeeze_channels)
    else:
      self.squeeze_excitation = nn.Identity()

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
    identity = x
    x = self.conv1(x)
    x = self.act1(self.norm1(x))
    x = self.conv2(x)
    x = self.squeeze_excitation(self.norm2(x))
    x = x + self.change_n_channels(identity)
    return self.final_act(x)


class TeamActor(nn.Module):
  """Take"""

  def __init__(self, in_channels: int, conv_dim: int, out_channels: int):
    super(TeamActor, self).__init__()
    self.in_channels = in_channels
    self.conv_dim = conv_dim
    self.out_channels = out_channels
    # self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
    self.conv2d = nn.Conv2d(in_channels, conv_dim, (1, 1))

    linear_in_channels = conv_dim + N_TEAM_ACTIOR_EXTRA_DIM  # expand for unit features
    self.linear = nn.Linear(linear_in_channels, out_channels)

  def forward(self, actor_base_out: torch.Tensor,
              origin_input_x: Dict[str, torch.Tensor]) -> torch.Tensor:
    actor_base_out = self.conv2d(actor_base_out)

    b = actor_base_out.shape[0]
    units_info = origin_input_x['_units_info']
    w_indices = units_info[:, :, 0]  # (b, 16)
    h_indices = units_info[:, :, 1]  # (b, 16)

    # normalize the unit energy feature
    unit_energies = units_info[:, :, (2, )] / MAX_UNIT_ENERGY  # (b, 16, 1)

    batch_indices = torch.arange(b).unsqueeze(1)
    selected_channels = actor_base_out[batch_indices, :, w_indices,
                                       h_indices]  # (b, 16, in_channels)
    expanded_channels = torch.cat((selected_channels, unit_energies),
                                  dim=-1)  # (b, 16, conv_dim + 1)

    x = self.linear(expanded_channels)
    return x.view(b, MAX_UNIT_NUM, self.out_channels)


class DictActor(nn.Module):
  "A dict of actors, each is MultiDiscrete of 1 dimention, with same number of actions."

  def __init__(self, in_channels: int, out_channels: int,
               action_space: gym.spaces.Dict):
    super(DictActor, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    if not all([
        isinstance(space, gym.spaces.MultiDiscrete)
        for space in action_space.spaces.values()
    ]):
      act_space_types = {
          key: type(space)
          for key, space in action_space.spaces.items()
      }
      raise ValueError(
          f"All action spaces must be MultiDiscrete. Found: {act_space_types}")
    if not all(
        [len(space.shape) == 1 for space in action_space.spaces.values()]):
      act_space_ndims = {
          key: space.shape
          for key, space in action_space.spaces.items()
      }

      raise ValueError(
          f"All action spaces must have 1 dimensions. Found: {act_space_ndims}"
      )
    if not all([
        space.nvec.min() == space.nvec.max()
        for space in action_space.spaces.values()
    ]):
      act_space_n_acts = {
          key: np.unique(space.nvec)
          for key, space in action_space.spaces.items()
      }
      raise ValueError(
          f"Each action space must have the same number of actions throughout the space. "
          f"Found: {act_space_n_acts}")
    self.action_space = action_space

    actors = dict()
    for key, space in self.action_space.spaces.items():
      # n_actions = space.shape[0]  # == MAX_UNIT_NUM
      action_dim = space.nvec.max()
      actors[key] = TeamActor(in_channels, out_channels, action_dim)

    self.actors = nn.ModuleDict(actors)

  def forward(self,
              x: torch.Tensor,
              actions_mask: Dict[str, torch.Tensor],
              origin_input_x: Dict[str, torch.Tensor],
              sample: bool,
              probs_output: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    policy_logits_out = {}
    actions_out = {}
    for key, space in self.action_space.spaces.items():
      n_actions = space.shape[0]
      action_dim = space.nvec.max()

      actor = self.actors[key]
      logits = actor(x, origin_input_x)
      assert logits.shape[1] == n_actions
      assert logits.shape[2] == action_dim

      # print(key, type(actions_mask), actions_mask)
      aam = actions_mask[key]

      # in case no action available for the unit, drop action mask
      orig_dtype = aam.dtype
      aam_new_type = aam.to(dtype=torch.int64)
      aam_filled = torch.where(
          (~(aam.to(torch.bool))).all(dim=-1, keepdim=True),
          torch.ones_like(aam_new_type),
          aam_new_type.to(dtype=torch.int64)).to(orig_dtype)

      logits = torch.where(aam_filled > 0, logits,
                           torch.zeros_like(logits) + float("-inf"))

      actions, probs = DictActor.logits_to_actions(logits.view(-1, action_dim),
                                                   sample, aam_filled)
      actions = actions.view(*logits.shape[:-1], -1)

      actions_out[key] = actions
      policy_logits_out[key] = logits

    if probs_output:
      return policy_logits_out, actions_out, probs
    return policy_logits_out, actions_out

  @staticmethod
  @torch.no_grad()
  def logits_to_actions(logits: torch.Tensor,
                        sample: bool,
                        actions_mask,
                        actions_per_square=1) -> int:
    if sample:
      probs = F.softmax(logits, dim=-1)
      # In case there are fewer than actions_per_square available actions, we add a small eps value
      probs = torch.where((probs > 0.).sum(dim=-1, keepdim=True)
                          >= actions_per_square, probs, probs + 1e-10)

      actions = torch.multinomial(probs,
                                  num_samples=actions_per_square,
                                  replacement=False)
      return actions, probs
    else:
      return logits.argsort(dim=-1,
                            descending=True)[..., :actions_per_square], None


class BaselineLayer(nn.Module):

  def __init__(self, in_channels: int, reward_space: RewardSpec):
    super(BaselineLayer, self).__init__()
    self.reward_min = reward_space.reward_min
    self.reward_max = reward_space.reward_max
    hidden_channles = in_channels + N_BASELINE_EXTRA_DIM
    self.linear = nn.Linear(hidden_channles, hidden_channles)
    self.linear2 = nn.Linear(hidden_channles, 1)
    if reward_space.zero_sum:
      self.activation = nn.Softmax(dim=-1)
    else:
      self.activation = nn.Sigmoid()

  def forward(self, x: torch.Tensor, baseline_extras):
    # Average feature planes
    x = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)

    # Add extra inputs for baseline
    x = torch.cat((x, baseline_extras), dim=1)

    # Project and reshape input
    x = self.linear(x)
    # Rescale to [0, 1], and then to the desired reward space
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    v = x * (self.reward_max - self.reward_min) + self.reward_min
    # print(
    # f'v={v}, x={x}, reward_max={self.reward_max} reward_min={self.reward_min}'
    # )
    return v


class BasicActorCriticNetwork(nn.Module):

  def __init__(
      self,
      base_model: nn.Module,
      hidden_dim: int,
      base_out_channels: int,
      reward_space: RewardSpec,
      actor_critic_activation: Callable = nn.ReLU,
      n_action_value_layers: int = 2,
  ):
    super(BasicActorCriticNetwork, self).__init__()
    self.dict_input_layer = DictInputLayer()
    self.base_model = base_model
    self.hidden_dim = hidden_dim
    self.base_out_channels = base_out_channels

    if n_action_value_layers < 2:
      raise ValueError(
          "n_action_value_layers must be >= 2 in order to use spectral_norm")

    self.actor_base = self.make_spectral_norm_head_base(
        n_layers=n_action_value_layers,
        n_channels=self.hidden_dim,
        activation=actor_critic_activation)
    self.actor = DictActor(hidden_dim, self.base_out_channels, ACTION_SPACE)

    self.baseline_base = self.make_spectral_norm_head_base(
        n_layers=n_action_value_layers,
        n_channels=self.hidden_dim,
        activation=actor_critic_activation)
    self.baseline = BaselineLayer(
        in_channels=hidden_dim,
        reward_space=reward_space,
    )
    self.n_action_value_layers = n_action_value_layers
    self.actor_critic_activation = actor_critic_activation

  def forward(self,
              x1: Dict[str, Union[dict, torch.Tensor]],
              sample: bool = True,
              probs_output=False,
              **actor_kwargs) -> Dict[str, Any]:
    x, actions_mask = self.dict_input_layer(x1)
    baseline_extras = x['_baseline_extras']

    base_out = self.base_model(x)
    ret = self.actor(self.actor_base(base_out),
                     actions_mask=actions_mask,
                     origin_input_x=x,
                     sample=sample,
                     probs_output=probs_output,
                     **actor_kwargs)
    baseline = self.baseline(self.baseline_base(base_out), baseline_extras)

    if probs_output:
      policy_logits, actions, probs = ret
      return dict(actions=actions,
                  policy_logits=policy_logits,
                  baseline=baseline,
                  probs=probs)
    else:
      policy_logits, actions = ret
      return dict(actions=actions,
                  policy_logits=policy_logits,
                  baseline=baseline)

  def sample_actions(self, *args, **kwargs):
    return self.forward(*args, sample=True, **kwargs)

  def select_best_actions(self, *args, **kwargs):
    return self.forward(*args, sample=False, **kwargs)

  @staticmethod
  def make_spectral_norm_head_base(n_layers: int, n_channels: int,
                                   activation: Callable) -> nn.Module:
    """
        Returns the base of an action or value head, with the final layer of the base/the semifinal layer of the
        head spectral normalized.
        NB: this function actually returns a base with n_layer - 1 layers, leaving the final layer to be filled in
        with the proper action or value output layer.
        """
    assert n_layers >= 2
    layers = []
    for i in range(n_layers - 2):
      layers.append(nn.Conv2d(n_channels, n_channels, (1, 1)))
      layers.append(activation())

    layers.append(
        nn.utils.spectral_norm(nn.Conv2d(n_channels, n_channels, (1, 1))))
    layers.append(activation())
    return nn.Sequential(*layers)


def create_model(flags,
                 observation_space,
                 device: torch.device,
                 reset=False) -> nn.Module:
  reward_spec = None
  if flags.reward_schema == 'shaping':
    reward_spec = RewardSpec(
        reward_min=-10,
        reward_max=+10,
        zero_sum=False,
    )
  if flags.reward_schema == 'game_win_loss2':
    reward_spec = RewardSpec(
        reward_min=-5,
        reward_max=+5,
        zero_sum=False,
    )
  if flags.reward_schema in ('match_win_loss', ):
    reward_spec = RewardSpec(
        reward_min=-5,
        reward_max=+5,
        zero_sum=False,
    )
  if flags.reward_schema in ('match_explore_win_loss', ):
    reward_spec = RewardSpec(
        reward_min=-5,
        reward_max=+5,
        zero_sum=False,
    )
  assert reward_spec is not None

  md = _create_model(observation_space,
                     embedding_dim=flags.embedding_dim,
                     hidden_dim=flags.hidden_dim,
                     base_out_channels=flags.base_out_channels,
                     n_blocks=flags.n_blocks,
                     kernel_size=flags.kernel_size,
                     device=device,
                     reward_spec=reward_spec,
                     reset=reset)
  return md


def _create_model(observation_space,
                  embedding_dim=16,
                  hidden_dim=32,
                  base_out_channels=32,
                  n_blocks=4,
                  kernel_size=5,
                  device: torch.device = torch.device('cpu'),
                  reward_spec: RewardSpec = None,
                  reset=None):
  base_model = nn.Sequential(
      ConvEmbeddingInputLayer(observation_space, embedding_dim, hidden_dim), *[
          ResidualBlock(in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        height=MAP_HEIGHT,
                        width=MAP_WIDTH,
                        kernel_size=kernel_size) for _ in range(n_blocks)
      ])
  model = BasicActorCriticNetwork(base_model, hidden_dim, base_out_channels,
                                  reward_spec)
  return model.to(device=device)

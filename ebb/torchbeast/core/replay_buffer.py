from collections import deque
from random import choice
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import numpy as np


def _to_tensor(x: Union[Dict, np.ndarray], device) -> Dict[str, Union[Dict, torch.Tensor]]:
  if isinstance(x, dict):
    return {
        key: _to_tensor(val, device)
        for key, val in x.items()
    }
  else:
    return x.to(device, non_blocking=True)


class ReplayBuffer:

  def __init__(self, max_size):
    self.max_size = max_size
    self.buffers = deque()

  def __len__(self):
    return len(self.buffers)

  def append(self, buf, device = torch.device("cpu")):
    """Adds new item into replay buffer and pop oldest if it's full."""
    buf = _to_tensor(buf, device)
    self.buffers.append(buf)
    if len(self.buffers) > self.max_size:
      self.buffers.popleft()

  def sample(self, device = torch.device("cpu")):
    """Random samples 1 item from replay buffer."""
    assert len(self.buffers) > 0, "Can not sample from empty buffers."

    buf = choice(self.buffers)
    return _to_tensor(buf, device)

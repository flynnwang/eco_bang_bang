import torch

from types import SimpleNamespace
from typing import Any, Dict, List, NoReturn, Tuple


def flags_to_namespace(flags: Dict) -> SimpleNamespace:
  flags = SimpleNamespace(**flags)

  # Optimizer params
  flags.optimizer_class = torch.optim.__dict__[flags.optimizer_class]

  # Miscellaneous params
  flags.actor_device = torch.device(flags.actor_device)
  flags.learner_device = torch.device(flags.learner_device)
  return flags

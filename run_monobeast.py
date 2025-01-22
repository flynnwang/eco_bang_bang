import hydra
import logging
import os
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from torch import multiprocessing as mp
import wandb

from ebb.utils import flags_to_namespace
from ebb.torchbeast.monobeast import train

os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(
    format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] "
            "%(message)s"),
    level=0,
)

# Set logging level to WARNING or higher
logging.getLogger('jax').setLevel(logging.WARNING)


def get_default_flags(flags: DictConfig) -> DictConfig:
  flags = OmegaConf.to_container(flags)

  # Env params
  flags.setdefault("seed", None)
  flags.setdefault(
      "num_buffers",
      max(2 * flags["num_actors"],
          flags["batch_size"] // flags["n_actor_envs"]) + 2)
  flags.setdefault("obs_space_kwargs", {})
  flags.setdefault("reward_space_kwargs", {})
  flags.setdefault("ob_params", {})

  # Training params
  flags.setdefault("use_mixed_precision", True)
  flags.setdefault("discounting", 0.999)
  # flags.setdefault("reduction", "mean")
  flags.setdefault("clip_grads", 10.)
  flags.setdefault("checkpoint_freq", 10.)
  flags.setdefault("num_learner_threads", 1)
  flags.setdefault("use_teacher", False)
  flags.setdefault("teacher_baseline_cost",
                   flags.get("teacher_kl_cost", 0.) / 2.)

  # Model params
  flags.setdefault("use_index_select", True)
  if flags.get("use_index_select"):
    logging.info(
        "index_select disables padding_index and is equivalent to using a learnable pad embedding."
    )

  # Reloading previous run params
  flags.setdefault("load_dir", None)
  flags.setdefault("checkpoint_file", None)
  flags.setdefault("weights_only", False)
  flags.setdefault("n_value_warmup_batches", 0)

  # Miscellaneous params
  flags.setdefault("disable_wandb", True)
  flags.setdefault("debug", False)
  return OmegaConf.create(flags)


@hydra.main(version_base="1.1", config_path="conf", config_name="")
def main(flags: DictConfig):
  cli_conf = OmegaConf.from_cli()
  if Path("config.yaml").exists():
    new_flags = OmegaConf.load("config.yaml")
    flags = OmegaConf.merge(new_flags, cli_conf)

  if flags.get("load_dir", None) and not flags.get("weights_only", False):
    # this ignores the local config.yaml and replaces it completely with saved one
    # however, you can override parameters from the cli still
    # this is useful e.g. if you did total_steps=N before and want to increase it
    logging.info(
        "Loading existing configuration, we're continuing a previous run")
    new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
    # Overwrite some parameters

    # default_config = {"--config-name": "luxs3_reward_shaping_v1"}
    OmegaConf.set_struct(new_flags, False)
    OmegaConf.set_struct(flags, False)
    OmegaConf.set_struct(cli_conf, False)
    flags = OmegaConf.merge(new_flags, flags, cli_conf)

  flags = get_default_flags(flags)
  logging.info(OmegaConf.to_yaml(flags, resolve=True))
  OmegaConf.save(flags, "config.yaml")

  logging.info("disable_wandb? = %s", flags.disable_wandb)

  config = OmegaConf.to_container(flags, resolve=True)
  if not flags.disable_wandb:
    wandb.init(
        project=flags.project,
        config=config,
        entity=flags.entity,
        group=flags.group,
        name=flags.name,
    )

  flags = flags_to_namespace(OmegaConf.to_container(flags))
  mp.set_sharing_strategy(flags.sharing_strategy)
  train(flags)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  main()

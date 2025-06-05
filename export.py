import argparse

import torch
from hydra import compose, initialize
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig


def unwrap_model(cfg: DictConfig) -> None:

    block_cls = get_class(cfg.model.block._target_)
    stem = instantiate(cfg.model.stem)

    # Instantiate model with class objects
    model = instantiate(cfg.model, block=block_cls, stem=stem)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    lightning_module = instantiate(
        cfg.training,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    unwrapped_model = lightning_module.model
    torch.save(unwrapped_model.state_dict(), args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs")
    parser.add_argument("--config-name", type=str, default="train")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)  # collect Hydra-style overrides

    args = parser.parse_args()

    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    unwrap_model(cfg)

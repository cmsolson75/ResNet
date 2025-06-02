import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger

from hydra.utils import instantiate, get_class
import lightning as L
import torch

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    block_cls = get_class(cfg.model.block._target_)
    stem = instantiate(cfg.model.stem)

    # Instantiate model with class objects
    model = instantiate(cfg.model, block=block_cls, stem=stem)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    datamodule = instantiate(cfg.dataset)
    lightning_module = instantiate(
        cfg.training,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Callbacks
    callbacks = [instantiate(cb) for cb in cfg.trainer.callbacks.values()]
    logger = instantiate(cfg.trainer.logger)
    # Trainer
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        benchmark=True,
    )
    ckpt_path = cfg.checkpoint.resume.ckpt_path
    if ckpt_path:
        print(f"Resuming from checkpoint: {cfg.checkpoint.resume.ckpt_path}")
    else:
        print("Starting training from scratch")

    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
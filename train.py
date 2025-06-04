import hydra
import lightning as L
import torch
from hydra.utils import get_class, instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.debug:
        print(OmegaConf.to_yaml(cfg))
    block_cls = get_class(cfg.model.block._target_)
    stem = instantiate(cfg.model.stem)

    # Instantiate model with class objects
    model = instantiate(cfg.model, block=block_cls, stem=stem)
    if (
        "checkpoint" in cfg
        and "finetune" in cfg.checkpoint
        and cfg.checkpoint.finetune.path is not None
    ):
        print(f"Loading pretrained weights from {cfg.checkpoint.finetune.path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(cfg.checkpoint.finetune.path, map_location=device)

        # Remove classifier weights if num_classes mismatches
        if "fc.weight" in state_dict and state_dict["fc.weight"].shape[0] != cfg.model.num_classes:
            state_dict.pop("fc.weight", None)
            state_dict.pop("fc.bias", None)

        model.load_state_dict(state_dict, strict=False)

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
    logger = instantiate(cfg.logger)
    # Trainer
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
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

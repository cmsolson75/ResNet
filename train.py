import hydra
import lightning as L
import torch
from hydra.utils import get_class, instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.debug:
        print(OmegaConf.to_yaml(cfg))
    block_cls = get_class(cfg.model.block._target_)
    stem = instantiate(cfg.model.stem)

    # Instantiate model with class objects
    model = instantiate(cfg.model, block=block_cls, stem=stem)
    if cfg.mode == "finetune":
        print("Fine-tuning from {cfg.ckpt_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(cfg.ckpt_path, map_location=device)
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
    callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]
    logger = instantiate(cfg.logger)
    # Trainer
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer,
    )
    if cfg.mode == "resume":
        print(f"Resuming from checkpoint: {cfg.ckpt_path}")
        trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    else:
        print("Starting training from scratch" if cfg.mode == "train" else "Fine-tuning")
        trainer.fit(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

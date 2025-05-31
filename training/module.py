import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from hydra.utils import instantiate

class TrainModule(L.LightningModule):
    def __init__(self, model, optimizer, scheduler, num_classes: int = 10):
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.num_classes = num_classes

        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc.update(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc.update(logits, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, self.parameters())
        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
    


import os

import lightning as L
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, ImageNet


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        normalize: dict,
        input_shape: list,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.normalize = normalize
        self.input_shape = input_shape

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(input_shape[1], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
        self.val_dataset = CIFAR10(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


class CIFAR100DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        normalize: dict,
        input_shape: list,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(input_shape[1], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

    def prepare_data(self):
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = CIFAR100(self.data_dir, train=True, transform=self.transform_train)
        self.val_dataset = CIFAR100(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


class ImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        normalize: dict,
        input_shape: list,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(input_shape[1]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # self.train_dataset = ImageNet(
        #     self.data_dir, train=True, transform=self.transform_train
        # )
        # self.val_dataset = ImageNet(
        #     self.data_dir, train=False, transform=self.transform_test
        # )
        self.train_dataset = ImageFolder(
            root=os.path.join(self.data_dir, "train"), transform=self.transform_train
        )
        self.val_dataset = ImageFolder(
            root=os.path.join(self.data_dir, "val"), transform=self.transform_test
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


# Setup for future webdataset needs, this is just a stub and needs to be verified.
class ImageNetWDSDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_shards: str,
        val_shards: str,
        batch_size: int = 64,
        num_workers: int = 8,
        input_shape=(3, 224, 224),
    ):
        super().__init__()
        self.train_shards = train_shards
        self.val_shards = val_shards
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_shape = input_shape

        self.train_transform = T.Compose(
            [
                T.RandomResizedCrop(self.input_shape[1]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.val_transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(self.input_shape[1]),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def setup(self, stage=None):
        def decode_cls(x):
            return int(x.decode("utf-8"))

        self.train_dataset = (
            wds.WebDataset(self.train_shards, handler=wds.warn_and_continue)
            .decode("pil")
            .to_tuple("jpg", "cls")
            .map_tuple(self.train_transform, decode_cls)
        )

        self.val_dataset = (
            wds.WebDataset(self.val_shards, handler=wds.warn_and_continue)
            .decode("pil")
            .to_tuple("jpg", "cls")
            .map_tuple(self.val_transform, decode_cls)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset.batched(self.batch_size, partial=False),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset.batched(self.batch_size, partial=False),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

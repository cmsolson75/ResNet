import os
import torch
import lightning as L
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, ImageNet


def log_and_continue(exn):
    import logging

    logging.warning(f"WDS error: {repr(exn)}")
    return True


def is_valid_sample(sample):
    return ("png" in sample or "jpg" in sample) and "cls" in sample


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        paths: dict,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        normalize: dict,
        input_shape: list,
    ) -> None:
        super().__init__()
        self.data_dir = paths["root"]
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
        paths: dict,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        normalize: dict,
        input_shape: list,
    ) -> None:
        super().__init__()
        self.data_dir = paths["root"]
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


class CIFAR10WDSDataModule(L.LightningDataModule):
    def __init__(
        self,
        paths: dict,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        normalize: dict,
        input_shape: list,
    ) -> None:
        super().__init__()
        self.train_shards = paths["train"]
        self.val_shards = paths["val"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.input_shape = input_shape

        self.transform_train = T.Compose(
            [
                T.RandomCrop(input_shape[1], padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

        self.transform_test = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

    def _make_pipeline(self, shards, transform, num_samples, is_train: bool):
        def decode_cls(x):
            if isinstance(x, bytes):
                return int(x.decode("utf-8"))
            return int(x)

        def preprocess(sample):
            sample["png"] = transform(sample["png"])
            sample["cls"] = decode_cls(sample["cls"])
            return sample

        stages = [
            wds.SimpleShardList(shards),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode("pil", handler=log_and_continue),
            wds.map(preprocess, handler=log_and_continue),
            wds.select(is_valid_sample),
            wds.to_tuple("png", "cls", handler=log_and_continue),
        ]

        if is_train:
            stages.append(wds.shuffle(2048, initial=2048))

        stages.append(wds.batched(self.batch_size, partial=True))

        pipeline = wds.DataPipeline(*stages).with_length(num_samples // self.batch_size)
        return pipeline

    def setup(self, stage=None):
        self.train_dataset = self._make_pipeline(
            self.train_shards, self.transform_train, num_samples=50000, is_train=True
        )
        self.val_dataset = self._make_pipeline(
            self.val_shards, self.transform_test, num_samples=10000, is_train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

class ImageNetWDSDataModule(L.LightningDataModule):
    def __init__(
        self,
        paths: dict,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        normalize: dict,
        input_shape: list,
    ) -> None:
        super().__init__()
        self.train_shards = paths["train"]
        self.val_shards = paths["val"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.input_shape = input_shape
        self.normalize = normalize

        height = input_shape[1]

        self.transform_train = T.Compose([
            T.RandomResizedCrop(height),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=self.normalize["mean"], std=self.normalize["std"]),
        ])

        self.transform_val = T.Compose([
            T.Resize(256),
            T.CenterCrop(height),
            T.ToTensor(),
            T.Normalize(mean=self.normalize["mean"], std=self.normalize["std"]),
        ])

    def _make_pipeline(self, shards, transform, num_samples, is_train: bool):
        def decode_cls(x):
            return int(x.decode("utf-8")) if isinstance(x, bytes) else int(x)

        def preprocess(sample):
            try:
                img = transform(sample["jpg"])
                if not isinstance(img, torch.Tensor):
                    raise TypeError("Transform did not return a tensor")
                if img.device.type != "cpu":
                    img = img.to("cpu", non_blocking=True)
                sample["jpg"] = img.contiguous()
                sample["cls"] = decode_cls(sample["cls"])
                return sample
            except Exception as e:
                import logging
                logging.warning(f"Sample preprocessing failed: {e}")
                return None

        stages = [
            wds.SimpleShardList(shards),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode("pil", handler=log_and_continue),
            wds.select(is_valid_sample),
            wds.map(preprocess, handler=log_and_continue),
            wds.select(lambda s: s is not None and "jpg" in s and "cls" in s),
            wds.to_tuple("jpg", "cls", handler=log_and_continue),
        ]

        if is_train:
            stages.append(wds.shuffle(2048, initial=2048))

        stages.append(wds.batched(self.batch_size, partial=True))

        return wds.DataPipeline(*stages).with_length(num_samples // self.batch_size)

    def setup(self, stage=None):
        self.train_dataset = self._make_pipeline(
            self.train_shards, self.transform_train, num_samples=1281167, is_train=True
        )
        self.val_dataset = self._make_pipeline(
            self.val_shards, self.transform_val, num_samples=50000, is_train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

import io
import math
import os
import tarfile

import torch
import torchvision.transforms as T
import webdataset as wds
from torchvision.datasets import CIFAR10

train_dataset = CIFAR10(root="datasets", train=True, download=True)
test_dataset = CIFAR10(root="datasets", train=False, download=True)


def write_shard(samples, shard_path):
    with tarfile.open(shard_path, "w") as tar:
        for idx, (img, label) in enumerate(samples):
            key = f"{idx:06d}"
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            img_bytes = img_buf.getvalue()

            info = tarfile.TarInfo(name=f"{key}.png")
            info.size = len(img_bytes)
            tar.addfile(tarinfo=info, fileobj=io.BytesIO(img_bytes))

            label_bytes = str(label).encode("utf-8")
            info = tarfile.TarInfo(name=f"{key}.cls")
            info.size = len(label_bytes)
            tar.addfile(tarinfo=info, fileobj=io.BytesIO(label_bytes))


def create_shards(dataset, output_dir, samples_per_shard=1000):
    os.makedirs(output_dir, exist_ok=True)
    total = len(dataset)
    num_shards = math.ceil(total / samples_per_shard)

    for shard_id in range(num_shards):
        start = shard_id * samples_per_shard
        end = min((shard_id + 1) * samples_per_shard, total)
        shard_samples = []

        for i in range(start, end):
            img, label = dataset[i]  # img is already PIL.Image.Image
            shard_samples.append((img, label))

        shard_path = os.path.join(output_dir, f"shard-{shard_id:05d}.tar")
        write_shard(shard_samples, shard_path)


create_shards(train_dataset, "datasets/wds_cifar10/train")
create_shards(test_dataset, "datasets/wds_cifar10/test")

transform = T.Compose(
    [T.ToTensor(), T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])]
)

dataset = (
    wds.WebDataset("datasets/wds_cifar10/train/shard-{00000..00049}.tar")
    .decode("pil")
    .to_tuple("png", "cls")
    .map_tuple(transform, int)
)

loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)


# for batch in loader:
#     images, labels = batch
#     print(images.shape, labels.shape)
#     break

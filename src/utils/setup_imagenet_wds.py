import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main(output_root="datasets/wds_imagenet1k", repo_id="timm/imagenet-1k-wds"):
    print("Downloading ImageNet WebDataset shards from HuggingFace...")
    cache_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=output_root,
        local_dir_use_symlinks=False,
        repo_type="dataset",
        ignore_patterns=["*.json", "*.md"],
    )

    output_root = Path(output_root)
    train_dir = output_root / "train"
    val_dir = output_root / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("Organizing shards into train/ and validation/ subdirectories...")

    for tar_file in Path(cache_dir).rglob("*.tar"):
        name = tar_file.name.lower()
        if "train" in name or "training" in name:
            dest = train_dir / tar_file.name
        elif "val" in name or "validation" in name:
            dest = val_dir / tar_file.name
        else:
            continue
        if not dest.exists():
            os.rename(tar_file, dest)

    print("Setup complete.")
    print(f"Train shards: {len(list(train_dir.glob('*.tar')))}")
    print(f"Validation shards: {len(list(val_dir.glob('*.tar')))}")


if __name__ == "__main__":
    main()

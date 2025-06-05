import os

import numpy as np
from PIL import Image


def generate_dummy_imagenet(root, num_classes=2, num_train=5, num_val=2, image_size=(224, 224)):
    for split in ["train", "val"]:
        for class_id in range(num_classes):
            class_dir = os.path.join(root, split, f"n{str(class_id).zfill(8)}")
            os.makedirs(class_dir, exist_ok=True)
            num_images = num_train if split == "train" else num_val
            for i in range(num_images):
                img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(class_dir, f"img_{i}.jpg"), format="JPEG")


# Example usage
if __name__ == "__main__":
    generate_dummy_imagenet(
        "/home/cameronolson/documents/datasets", num_classes=3, num_train=4, num_val=2
    )

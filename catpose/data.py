from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .geometry import crop, gaussian, read_rgb


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@lru_cache(maxsize=12)
def decoded(path):
    return read_rgb(path)


def normalized(rgb):
    image = rgb.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
    return torch.from_numpy(np.ascontiguousarray((image - MEAN) / STD))


class CatDataset(Dataset):
    def __init__(self, root, rows, seed=0, augment=False, training=False):
        self.root = Path(root)
        self.rows = rows
        self.seed = seed
        self.augment = augment
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.training:
            index, draw = index
        row = self.rows[index]
        points = np.asarray(row["keypoints"], dtype=np.float32)
        rgb = decoded(self.root / row["file"])
        rng = np.random.default_rng([self.seed, draw]) if self.augment else None
        image, joints, matrix = crop(rgb, row["bbox"], points, rng)
        if self.training:
            heat, valid = gaussian(joints)
            return normalized(image), torch.from_numpy(heat), torch.from_numpy(valid)
        return {
            "inputs": normalized(image), "index": index,
            "inverse": cv2.invertAffineTransform(matrix),
        }


class DrawBatches:
    """Permute every epoch and index each draw for deterministic resumption."""
    def __init__(self, count, seed, micro_batch, first_step, end_step, accumulate):
        if count < 1:
            raise ValueError("Training set is empty")
        self.count = count
        self.seed = seed
        self.batch = micro_batch
        self.first = first_step * micro_batch * accumulate
        self.end = end_step * micro_batch * accumulate

    def __iter__(self):
        current_epoch = -1
        batch = []
        for draw in range(self.first, self.end):
            epoch, index = divmod(draw, self.count)
            if epoch != current_epoch:
                permutation = np.random.default_rng([self.seed, epoch]).permutation(self.count)
                current_epoch = epoch
            batch.append((int(permutation[index]), draw))
            if len(batch) == self.batch:
                yield batch
                batch = []

    def __len__(self):
        return (self.end - self.first) // self.batch


def configure_worker(_):
    cv2.setNumThreads(1)


def make_loaders(config, train_rows, val_rows, seed, start=0, end=None, augment=True):
    end = config["steps"] if end is None else end
    options = {
        "num_workers": config["workers"], "pin_memory": config["device"] == "cuda",
        "worker_init_fn": configure_worker, "generator": torch.Generator().manual_seed(seed),
    }
    train = CatDataset(config["data"], train_rows, seed, augment, training=True)
    batches = DrawBatches(
        len(train), seed, config["micro_batch"], start, end, config["accumulate"],
    )
    loader = DataLoader(train, batch_sampler=batches, **options)
    validate = CatDataset(config["data"], val_rows)
    val = DataLoader(validate, batch_size=config["micro_batch"], **options)
    return loader, val


def training_rows(manifest, size, seed):
    ids = manifest["subsets"][str(seed)][str(size)]
    selected = set(ids)
    return ids, [row for row in manifest["splits"]["train"] if row["image_id"] in selected]

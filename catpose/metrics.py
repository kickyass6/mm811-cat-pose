import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CatDataset
from .geometry import STRIDE, transform


def decode(heatmap):
    """Decode the peak, refined by a quarter of a heatmap cell toward its stronger neighbour."""
    n, height, width = heatmap.shape
    flat = heatmap.reshape(n, -1)
    valid = np.isfinite(flat).all(axis=1)
    peaks = np.where(np.isfinite(flat), flat, -np.inf).argmax(axis=1)
    y, x = np.divmod(peaks, width)
    scores = flat[np.arange(n), peaks].astype(np.float64)
    refined = np.stack((x, y), axis=1).astype(np.float64)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if valid[i] and 0 < xi < width-1:
            refined[i, 0] += 0.25 * np.sign(heatmap[i, yi, xi+1] - heatmap[i, yi, xi-1])
        if valid[i] and 0 < yi < height-1:
            refined[i, 1] += 0.25 * np.sign(heatmap[i, yi+1, xi] - heatmap[i, yi-1, xi])
    refined[~valid] = np.nan
    scores[~valid] = np.nan
    return refined * STRIDE, scores


def errors_for(row, points):
    truth = np.asarray(row["keypoints"])
    err = np.linalg.norm(points - truth[:, :2], axis=1) / max(row["bbox"][2:])
    err[~np.isfinite(err)] = np.inf
    return err, truth[:, 2] == 2


def counts(errors, visible, joints, threshold=0.10):
    errors, visible = errors[..., list(joints)], visible[..., list(joints)]
    return int(((errors <= threshold) & visible).sum()), int(visible.sum())


@torch.inference_mode()
def predictions(model, dataset, device, amp, *, batch=4, loader=None):
    model.eval()
    if loader is None:
        loader = DataLoader(dataset, batch_size=batch)
    for item in loader:
        inputs = item["inputs"].to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(inputs)
        for heatmap, index, inverse in zip(
            output.float().cpu().numpy(), item["index"].tolist(), item["inverse"].numpy(),
        ):
            points, scores = decode(heatmap)
            yield dataset.rows[index], transform(points, inverse), scores


def error_arrays(items):
    per_row = [errors_for(row, points) for row, points, _ in items]
    return tuple(np.stack(values) for values in zip(*per_row))

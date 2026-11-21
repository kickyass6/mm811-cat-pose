import cv2
import numpy as np

from .common import FLIP


WIDTH, HEIGHT = 192, 256
STRIDE = 4
HEAT_WIDTH, HEAT_HEIGHT = WIDTH // STRIDE, HEIGHT // STRIDE


def affine(bbox, scale=1.0, degrees=0.0, flip=False):
    x, y, width, height = bbox
    crop_width = max(width, height * WIDTH / HEIGHT) * 1.25 * scale
    center = np.array([x + width / 2, y + height / 2])
    matrix = cv2.getRotationMatrix2D(tuple(center), degrees, WIDTH / crop_width)
    matrix[:, 2] += [(WIDTH - 1) / 2 - center[0], (HEIGHT - 1) / 2 - center[1]]
    if flip:
        matrix[0] *= -1
        matrix[0, 2] += WIDTH - 1
    return matrix


def transform(points, matrix):
    return points @ matrix[:, :2].T + matrix[:, 2]


def crop(rgb, bbox, joints, rng=None):
    flip = False
    if rng is None:
        matrix = affine(bbox)
    else:
        flip = rng.random() < 0.5
        matrix = affine(bbox, rng.uniform(0.85, 1.15), rng.uniform(-20, 20), flip)
    image = cv2.warpAffine(rgb, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
    joints = np.array(joints, dtype=np.float32, copy=True)
    if flip:
        joints = joints[list(FLIP)]
    joints[:, :2] = transform(joints[:, :2], matrix)
    return image, joints, matrix


def gaussian(joints):
    x, y, visibility = joints.T[:, :, None, None]
    valid = (
        (visibility > 0) & (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)
    )
    dx = np.arange(HEAT_WIDTH, dtype=np.float32) - x / STRIDE
    dy = np.arange(HEAT_HEIGHT, dtype=np.float32)[:, None] - y / STRIDE
    heatmaps = np.exp((dx ** 2 + dy ** 2) / -8.0) * valid
    return heatmaps.astype(np.float32), valid[:, 0, 0]


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

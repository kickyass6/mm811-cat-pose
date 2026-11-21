import cv2
import numpy as np

from .common import EDGES, KNEES, PAWS


LEFT = (91, 224, 246)
RIGHT = (255, 174, 94)
NEUTRAL = (250, 239, 219)


def joint_color(index):
    if index in (0, 5, 6, 7, 11, 12, 13):
        return LEFT
    if index in (1, 8, 9, 10, 14, 15, 16):
        return RIGHT
    return NEUTRAL


def skeleton(rgb, points, show, emphasize=None):
    rgb = rgb.copy()
    height, width = rgb.shape[:2]
    show = (
        np.asarray(show, dtype=bool) & np.isfinite(points).all(axis=1) &
        (points[:, 0] >= 0) & (points[:, 0] < width) &
        (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    xy = np.where(show[:, None], points, 0).round().astype(int)
    thickness = max(2, round(min(height, width) / 160))
    for start, end in EDGES:
        if show[start] and show[end]:
            cv2.line(rgb, tuple(xy[start]), tuple(xy[end]), (22, 27, 34), thickness + 2, cv2.LINE_AA)
            cv2.line(rgb, tuple(xy[start]), tuple(xy[end]), joint_color(end), thickness, cv2.LINE_AA)
    for i, point in enumerate(xy):
        if show[i]:
            radius = thickness + (3 if i in (*PAWS, *KNEES) else 1)
            cv2.circle(rgb, tuple(point), radius + 1, (22, 27, 34), -1, cv2.LINE_AA)
            cv2.circle(rgb, tuple(point), radius, joint_color(i), -1, cv2.LINE_AA)
    if emphasize is not None and show[emphasize]:
        cv2.circle(rgb, tuple(xy[emphasize]), thickness + 10, (22, 27, 34), 4, cv2.LINE_AA)
        cv2.circle(rgb, tuple(xy[emphasize]), thickness + 10, (250, 255, 114), 2, cv2.LINE_AA)
    return rgb


def save_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write {path}")

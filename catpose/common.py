import hashlib
import json
import os
from pathlib import Path

import yaml


JOINTS = (
    "L_Eye", "R_Eye", "Nose", "Neck", "Root of tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw", "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw", "R_Hip", "R_Knee", "R_B_Paw",
)
PAWS = (7, 10, 13, 16)
KNEES = (12, 15)
FLIP = (1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13)
EDGES = (
    (0, 1), (0, 2), (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13), (4, 14), (14, 15), (15, 16),
)
SEEDS = (11, 22, 33)
SIZES = ("100", "25", "50", "all")
INITS = ("imagenet", "coco_pose")
DATA_HOME = "https://github.com/AlexTheBad/AP-10K#download"
DATA_URL = (
    "https://drive.google.com/file/d/1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad/view"
)


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def config_from(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["micro_batch"] * config["accumulate"] != 8:
        raise ValueError("Expected an effective batch of eight instances")
    return config


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_directory(root, size, init, seed):
    return Path(root) / f"{size}_{init}_s{seed}"

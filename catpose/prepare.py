import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from .common import DATA_HOME, DATA_URL, JOINTS, KNEES, PAWS, SEEDS, read_json, sha256, write_json
from .geometry import affine, read_rgb


SPLITS = ("train", "val", "test")


def hashes(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    low = cv2.dct(resized.astype(np.float32))[:8, :8].ravel()
    phash = int.from_bytes(np.packbits(low > np.median(low[1:])).tobytes(), "big")
    gradient = np.diff(cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16), axis=1)
    dhash = int.from_bytes(np.packbits(gradient > 0).tobytes(), "big")
    return phash, dhash


def valid_instance(ann, width, height):
    if ann.get("iscrowd"):
        return None, "crowd"
    box = np.asarray(ann.get("bbox", []), dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
        return None, "nonpositive_or_invalid_bbox"
    if (box[:2] >= [width, height]).any() or (box[:2] + box[2:] <= 0).any():
        return None, "bbox_outside_image"
    keypoints = np.asarray(ann.get("keypoints", []), dtype=np.float64)
    if keypoints.shape != (3 * len(JOINTS),):
        return None, "missing_or_invalid_keypoints"
    keypoints = keypoints.reshape(-1, 3)
    if not np.isin(keypoints[:, 2], [0, 1, 2]).all():
        return None, "invalid_visibility"
    invalid = (keypoints[:, 2] > 0) & (
        ~np.isfinite(keypoints[:, :2]).all(axis=1)
        | (keypoints[:, :2] < 0).any(axis=1)
        | (keypoints[:, :2] >= [width, height]).any(axis=1)
    )
    suppressed = [JOINTS[i] for i in np.flatnonzero(invalid)]
    keypoints[invalid | (keypoints[:, 2] == 0)] = 0
    if not (keypoints[:, 2] > 0).any():
        return None, "no_valid_annotated_joint"
    matrix = affine(box)
    return {
        "annotation_id": int(ann["id"]), "bbox": box.tolist(),
        "keypoints": keypoints.tolist(), "affine": matrix.tolist(),
        "inverse_affine": cv2.invertAffineTransform(matrix).tolist(),
    }, suppressed


def prepare(data, split_id, species, out, archive=None):
    data = Path(data)
    audit = {
        "source": DATA_HOME, "archive_url": DATA_URL, "archive_sha256": sha256(archive) if archive else None,
        "archive_file": Path(archive).name if archive else None,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "annotations": {},
        "splits": {}, "issues": [], "duplicates": [],
    }
    rows = {split: [] for split in SPLITS}
    records = {}
    for split in SPLITS:
        path = data / "annotations" / f"ap10k-{split}-split{split_id}.json"
        raw = read_json(path)
        audit["annotations"][path.name] = sha256(path)
        category = next(cat for cat in raw["categories"] if cat["name"].casefold() == species.casefold())
        names = category.get("keypoints")
        expected = (
            "left_eye", "right_eye", "nose", "neck", "root_of_tail",
            "left_shoulder", "left_elbow", "left_front_paw", "right_shoulder", "right_elbow",
            "right_front_paw", "left_hip", "left_knee", "left_back_paw", "right_hip", "right_knee",
            "right_back_paw",
        )
        if tuple(names or ()) != expected:
            raise ValueError(f"{path}: unknown joint ordering {names}")
        images = {int(obj["id"]): obj for obj in raw["images"]}
        anns = [ann for ann in raw["annotations"] if ann["category_id"] == category["id"]]
        counts = Counter(int(ann["image_id"]) for ann in anns)
        audit["splits"][split] = {
            "category_id": category["id"], "raw_photos": len(counts), "raw_instances": len(anns),
        }
        readable = {}
        for image_id in sorted(counts):
            entry = images[image_id]
            relative = Path("data") / entry["file_name"]
            full_path = data / relative
            try:
                image = read_rgb(full_path)
                height, width = image.shape[:2]
                if width != entry["width"] or height != entry["height"]:
                    raise ValueError("decoded size differs from the official annotation")
            except (ValueError, OSError) as error:
                audit["issues"].append({"split": split, "image_id": image_id, "reason": str(error)})
                continue
            photo = {
                "image_id": image_id, "file": relative.as_posix(), "width": width, "height": height,
                "image_sha256": sha256(full_path),
            }
            readable[image_id] = photo
            records[split, image_id] = (photo["image_sha256"], *hashes(image))
        for ann in sorted(anns, key=lambda item: (int(item["image_id"]), int(item["id"]))):
            image_id = int(ann["image_id"])
            photo = readable.get(image_id)
            if photo is None:
                continue
            instance, detail = valid_instance(ann, photo["width"], photo["height"])
            if detail:
                audit["issues"].append({
                    "split": split, "image_id": image_id, "annotation_id": int(ann["id"]),
                    "reason": detail if instance is None else "invalid_coordinates_masked",
                    "masked_joints": detail if instance is not None else [],
                })
            if instance is not None:
                rows[split].append({**photo, **instance})

    valid_ids = {(split, row["image_id"]) for split, values in rows.items() for row in values}
    dropped = set()
    for a, b in combinations(sorted(valid_ids), 2):
        sha_a, phash_a, dhash_a = records[a]
        sha_b, phash_b, dhash_b = records[b]
        p_dist, d_dist = (phash_a ^ phash_b).bit_count(), (dhash_a ^ dhash_b).bit_count()
        exact = sha_a == sha_b or a[1] == b[1]
        if not exact and (p_dist > 6 or d_dist > 8):
            continue
        conflict = a[0] != b[0]
        delete = (a if a[0] == "train" else b) if conflict and "train" in (a[0], b[0]) else None
        if delete:
            dropped.add(delete)
        audit["duplicates"].append({
            "a": a, "b": b, "exact": exact, "p_hash_distance": p_dist,
            "d_hash_distance": d_dist, "removed": delete,
        })
    for split in SPLITS:
        rows[split] = [row for row in rows[split] if (split, row["image_id"]) not in dropped]
        audit["splits"][split].update({
            "photos": len({row["image_id"] for row in rows[split]}), "instances": len(rows[split]),
            "visible": {
                JOINTS[i]: sum(row["keypoints"][i][2] == 2 for row in rows[split])
                for i in (*PAWS, *KNEES)
            },
        })
    id_sets = [set(row["image_id"] for row in rows[split]) for split in SPLITS]
    if any(a & b for a, b in combinations(id_sets, 2)):
        raise ValueError("Image IDs remain shared between splits")
    if not all(id_sets):
        raise ValueError("One or more splits have no eligible images")
    subsets = {}
    for seed in SEEDS:
        ordered = np.random.default_rng(seed).permutation(sorted(id_sets[0])).tolist()
        subsets[str(seed)] = {str(n): ordered[:n] for n in (25, 50, 100)}
        subsets[str(seed)]["all"] = ordered
    manifest = {"split": split_id, "species": species, "joints": JOINTS, "splits": rows, "subsets": subsets}
    write_json(out / "data_audit.json", audit)
    write_json(out / "manifest.json", manifest)
    print(audit["splits"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--species", default="cat")
    parser.add_argument("--out", type=Path, default=Path("manifests"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    prepare(args.data, args.split, args.species, args.out, args.archive)


if __name__ == "__main__":
    main()

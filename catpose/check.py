import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import FLIP, INITS, PAWS, config_from, read_json, write_json
from .data import make_loaders
from .geometry import WIDTH, affine, crop, gaussian, read_rgb, transform
from .metrics import counts, decode, error_arrays, errors_for, predictions
from .model import initialized
from .overlay import save_rgb, skeleton
from .train import configure_device, device_record, loss_step, setup


def check_annotations(config, manifest, out):
    rows = manifest["splits"]["train"] + manifest["splits"]["val"]
    image_counts = Counter(row["image_id"] for row in rows)
    rng = np.random.default_rng(11)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    missing = lambda row: sum(p[2] == 0 for p in row["keypoints"])
    width = lambda row: row["bbox"][2] / row["bbox"][3]
    subsets = [
        ("multi-cat", [row for row in rows if image_counts[row["image_id"]] > 1]),
        ("partial labels", sorted(rows, key=missing, reverse=True)),
        ("wide box", sorted(rows, key=width, reverse=True)),
        ("tall box", sorted(rows, key=width)),
        ("visible paws", [row for row in rows if all(row["keypoints"][i][2] == 2 for i in PAWS)]),
        ("random", rows),
    ]
    used, tiles, reviewed = set(), [], []
    max_error = 0.0
    for category, group in subsets:
        candidates = [row for row in group if row["image_id"] not in used]
        for row in candidates[:3]:
            used.add(row["image_id"])
            points = np.asarray(row["keypoints"])
            rgb = read_rgb(Path(config["data"]) / row["file"])
            cart, cropped, matrix = crop(rgb, row["bbox"], points)
            mirrored = affine(row["bbox"], flip=True)
            mirrored_points = transform(points[list(FLIP), :2], mirrored)
            expected = cropped[list(FLIP), :2].copy()
            expected[:, 0] = WIDTH - 1 - expected[:, 0]
            assert np.allclose(mirrored_points, expected, atol=0.001)
            restored = transform(cropped[:, :2], cv2.invertAffineTransform(matrix))
            max_error = max(max_error, float(np.linalg.norm(restored - points[:, :2], axis=1).max()))
            errors, visible = errors_for(row, points[:, :2])
            hits, total = counts(errors, visible, PAWS)
            assert hits == total
            missing = points[:, :2].copy()
            missing[visible] = np.nan
            assert counts(*errors_for(row, missing), PAWS) == (0, total)
            tiles.append(np.concatenate((
                skeleton(cart, cropped[:, :2], cropped[:, 2] > 0),
                skeleton(np.flip(cart, axis=1).copy(), mirrored_points, points[list(FLIP), 2] > 0),
            ), axis=1))
            reviewed.append({
                "image_id": row["image_id"], "annotation_id": row["annotation_id"],
                "category": category, "visible_paws": total,
            })
    assert max_error < 1 and len(tiles) >= 12
    for i, (image, info) in enumerate(zip(tiles, reviewed)):
        caption = np.full((27, image.shape[1], 3), 24, dtype=np.uint8)
        cv2.putText(caption, f"{info['image_id']}  {info['category']}  |  Original / flip",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
        tiles[i] = np.concatenate((image, caption))
    for start in range(0, len(tiles), 6):
        group = tiles[start:start + 6]
        group.extend([np.zeros_like(tiles[0])] * (6 - len(group)))
        page = np.concatenate([np.concatenate(group[i:i+2], axis=1) for i in range(0, 6, 2)])
        save_rgb(out / f"annotations_{start // 6 + 1}.jpg", page)
    spot = np.tile([96, 128, 2], (17, 1))
    targets, visible = gaussian(spot)
    decoded, _ = decode(targets)
    assert visible.all() and np.array_equal(decoded, spot[:, :2])
    return {"roundtrip_max_original_pixels": max_error, "overlays": reviewed,
            "perfect_paw_pck": 1.0, "missing_paw_pck": 0.0}


def overfit(config, manifest, n, steps, device):
    used, rows = set(), []
    train = manifest["splits"]["train"]
    for row in sorted(train, key=lambda row: -sum(p[2] == 2 for p in row["keypoints"])):
        if row["image_id"] not in used and len(used) < n:
            used.add(row["image_id"])
            rows.append(row)
    model, optimizer, scaler, _ = setup(config, "imagenet", 11, device)
    loader, val = make_loaders(config, rows, rows, 11, end=steps, augment=False)

    def score():
        return counts(*error_arrays(
            predictions(model, val.dataset, device, scaler.is_enabled(), loader=val),
        ), PAWS)

    before = score()
    iterator = iter(loader)
    losses = []
    for _ in range(steps):
        loss, _, _ = loss_step(model, optimizer, scaler, iterator, config["accumulate"], device)
        losses.append(loss)
    after = score()
    report = {
        "image_ids": sorted(used), "annotation_ids": [row["annotation_id"] for row in rows],
        "steps": steps, "initial_paw_hits": before[0], "final_paw_hits": after[0],
        "visible_paws": after[1], "first_loss": losses[0], "final_loss": losses[-1],
    }
    report["pass"] = (
        after[0] / after[1] >= 0.75 and (after[0] - before[0]) / after[1] >= 0.25
        and losses[-1] <= 0.6 * losses[0]
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overfit-images", type=int, default=4)
    parser.add_argument("--overfit-steps", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path("logs/check"))
    parser.add_argument("--annotations-only", action="store_true")
    args = parser.parse_args()
    config = config_from(args.config)
    manifest = read_json(config["manifest"])
    report = check_annotations(config, manifest, args.out)
    if not args.annotations_only:
        device = configure_device(config)
        report["device"] = device_record(device)
        hashes = []
        for init in INITS:
            _, loading = initialized(init, 11, config["weights"])
            write_json(args.out / f"initialization_{init}.json", loading)
            hashes.append(loading["head_sha256"])
        assert len(set(hashes)) == 1, "Paired starting heads differ"
        report["head_sha256"] = hashes[0]
        report["overfit"] = overfit(config, manifest, args.overfit_images, args.overfit_steps, device)
    write_json(args.out / "report.json", report)
    print({k: v for k, v in report.items() if k != "overlays"}, flush=True)
    if not args.annotations_only and not report["overfit"]["pass"]:
        raise SystemExit("Four-photo overfit gate failed")


if __name__ == "__main__":
    main()

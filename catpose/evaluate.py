import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from .common import INITS, JOINTS, KNEES, PAWS, SEEDS, SIZES, read_json, sha256, source_hash, write_json
from .data import CatDataset
from .geometry import affine, read_rgb, transform, WIDTH, HEIGHT
from .metrics import counts, errors_for, predictions
from .model import PoseModel
from .overlay import save_rgb, skeleton
from .train import configure_device, training_logic_hash


GROUPS = {"paws": PAWS, "knees": KNEES, "all_visible": tuple(range(len(JOINTS)))}
LABELS = {"imagenet": "ImageNet classification", "coco_pose": "COCO human pose"}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def complete_runs(root):
    items = []
    for marker in sorted(root.glob("*/complete.json")):
        directory = marker.parent
        run = read_json(directory / "run.json")
        result = read_json(marker)
        sha = sha256(directory / "best.pt")
        history = read_json(directory / "history.json")
        selected = max(history, key=lambda row: (row["val_paw_hits"], -row["step"]))
        if (result["step"] != run["config"]["steps"] or sha != result["best_sha256"]
                or selected != result["best"] or history[-1]["step"] != result["step"]):
            raise ValueError(f"Inconsistent completion or validation selection at {directory}")
        items.append({
            "directory": directory.name, "run": run, "result": result, "best_sha256": sha,
        })
    if not items:
        raise ValueError(f"No completed runs in {root}")
    return items


def by_condition(items):
    conditions = {}
    for item in items:
        run = item["run"]
        key = run["size"], run["init"], run["seed"]
        if key in conditions:
            raise ValueError(f"Duplicate run condition {key}")
        conditions[key] = item
    return conditions


def verify_experiment(items, require_all):
    conditions = by_condition(items)
    expected = {(size, init, seed) for size in SIZES for init in INITS for seed in SEEDS}
    actual = set(conditions)
    missing = expected - actual
    if actual - expected or (require_all and missing):
        raise ValueError(f"Unexpected conditions {sorted(actual - expected)} or missing {sorted(missing)}")
    if not all(("100", init, 11) in conditions for init in INITS):
        raise ValueError("Required primary pair is missing")
    first = items[0]["run"]
    for (size, init, seed), item in conditions.items():
        if (size, "coco_pose" if init == "imagenet" else "imagenet", seed) not in conditions:
            raise ValueError(f"Unpaired condition {size}, {init}, {seed}")
        run = item["run"]
        for field in ("manifest_sha256", "training_source_sha256"):
            if run[field] != first[field]:
                raise ValueError(f"Runs do not agree on {field}")
        if run["config"] != first["config"]:
            raise ValueError("Training configurations differ across runs")
    return [list(key) for key in sorted(missing)]


def make_lock(root, out, storyboard, allow_incomplete=False):
    frozen = out / "test_lock.json"
    if frozen.exists():
        raise FileExistsError(f"Test lock already exists: {frozen}")
    items = complete_runs(root)
    missing = verify_experiment(items, not allow_incomplete)
    config = items[0]["run"]["config"]
    if training_logic_hash() != items[0]["run"]["training_source_sha256"]:
        raise ValueError("Current training source differs from the completed runs")
    actual_manifest_hash = sha256(config["manifest"])
    if actual_manifest_hash != items[0]["run"]["manifest_sha256"]:
        raise ValueError("Current manifest differs from training manifest")
    if read_json(storyboard)["manifest_sha256"] != actual_manifest_hash:
        raise ValueError("Storyboard targets a different dataset")
    conditions = by_condition(items)
    for (size, init, seed), item in conditions.items():
        if init != "imagenet":
            continue
        pair = conditions[size, "coco_pose", seed]
        if sha256(root / item["directory"] / "selection.json") != sha256(root / pair["directory"] / "selection.json"):
            raise ValueError("Paired training selections differ")
        heads = {
            read_json(root / current["directory"] / "initialization.json")["head_sha256"]
            for current in (item, pair)
        }
        if len(heads) != 1 or item["run"]["device"] != pair["run"]["device"]:
            raise ValueError("Paired models differ in device or new head")
    write_json(frozen, {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hash(),
        "storyboard": str(storyboard), "storyboard_sha256": sha256(storyboard),
        "bootstrap_repetitions": 2000, "bootstrap_seed": 811,
        "checkpoint_selection": "Highest visible validation paw PCK@0.10. Earliest step on ties.",
        "late_fit_diagnostic": (
            "Sustained late decline means lower training loss at the last step and both final "
            "validation PCK evaluations below the selected checkpoint."
        ),
        "metric": "Euclidean distance in original coordinates / longest original ground truth box edge",
        "visible_label": 2, "thresholds": [0.10, 0.05],
        "runs": items, "missing_conditions": missing, "manifest_sha256": actual_manifest_hash,
    })
    print(f"Locked {len(items)} runs for test: {frozen}", flush=True)


def infer(path, data, rows, config, device):
    model = PoseModel().to(device, memory_format=torch.channels_last)
    saved = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(saved["model"], strict=True)
    dataset = CatDataset(data, rows)
    return list(predictions(model, dataset, device, config["amp"] and device.type == "cuda",
                            batch=config["micro_batch"]))


def metric(errors, visible, joints):
    hits, total = counts(errors, visible, joints)
    hits05, _ = counts(errors, visible, joints, threshold=0.05)
    subset = errors[:, list(joints)][visible[:, list(joints)]]
    finite = subset[np.isfinite(subset)]
    return {
        "hits_at_010": hits, "visible_points": total, "pck_at_010": hits / total if total else None,
        "hits_at_005": hits05, "pck_at_005": hits05 / total if total else None,
        "mean_norm_error": float(finite.mean()) if len(finite) else None,
        "norm_error_count": len(finite), "missing_predictions": total - len(finite),
    }


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def row_predictions(base, output):
    row, points, scores = output
    errors, visible = errors_for(row, points)
    for i, (x, y, v) in enumerate(row["keypoints"]):
        yield {
            **base, "image_id": row["image_id"], "annotation_id": row["annotation_id"],
            "joint": JOINTS[i], "visibility": int(v), "truth_x": x if v else None,
            "truth_y": y if v else None, "pred_x": finite_number(points[i, 0]),
            "pred_y": finite_number(points[i, 1]), "raw_peak": finite_number(scores[i]),
            "bbox_x": row["bbox"][0], "bbox_y": row["bbox"][1],
            "bbox_w": row["bbox"][2], "bbox_h": row["bbox"][3],
            "normalized_error": finite_number(errors[i]) if visible[i] else None,
            "hit_010": int(errors[i] <= 0.10) if visible[i] else None,
        }


def pair_analysis(records, rows, visible, repeats, seed):
    image_ids = sorted({row["image_id"] for row in rows})
    image_indices = {image_id: i for i, image_id in enumerate(image_ids)}
    index = [image_indices[row["image_id"]] for row in rows]
    rng = np.random.default_rng(seed)
    image_weights = rng.multinomial(len(image_ids), np.full(len(image_ids), 1 / len(image_ids)), repeats)
    paired, intervals = [], []
    for size in SIZES:
        seeds = [s for s in SEEDS if (size, "imagenet", s) in records and (size, "coco_pose", s) in records]
        if not seeds:
            continue
        for group, joints in GROUPS.items():
            joint_mask = visible[:, list(joints)]
            denominators = np.bincount(index, weights=joint_mask.sum(axis=1), minlength=len(image_ids))
            differences = []
            for s in seeds:
                c, p = (records[size, init, s][:, list(joints)] for init in INITS)
                hit_c = ((c <= 0.10) & joint_mask).sum(axis=1)
                hit_p = ((p <= 0.10) & joint_mask).sum(axis=1)
                visible_count = int(denominators.sum())
                paired.append({
                    "size": size, "seed": s, "group": group, "visible_points": visible_count,
                    "imagenet_hits": int(hit_c.sum()), "coco_pose_hits": int(hit_p.sum()),
                    "imagenet_pck": hit_c.sum() / visible_count, "coco_pose_pck": hit_p.sum() / visible_count,
                    "delta_pp": 100.0 * (hit_p.sum() - hit_c.sum()) / visible_count,
                })
                differences.append(np.bincount(index, weights=hit_p - hit_c, minlength=len(image_ids)))
            per_image_difference = np.mean(differences, axis=0)
            n_boot = image_weights @ denominators
            samples = (image_weights[n_boot > 0] @ per_image_difference) * 100.0 / n_boot[n_boot > 0]
            seed_points = 100.0 * np.stack(differences).sum(axis=1) / denominators.sum()
            lo, hi = np.quantile(samples, [0.025, 0.975])
            intervals.append({
                "size": size, "group": group, "n_seeds": len(seeds),
                "test_photos": len(image_ids), "visible_points_per_seed": int(denominators.sum()),
                "mean_delta_pp": float(seed_points.mean()),
                "seed_sd_pp": float(seed_points.std(ddof=1)) if len(seeds) > 1 else None,
                "ci95_low_pp": float(lo), "ci95_high_pp": float(hi),
                "bootstrap_replicates": len(samples),
            })
    return paired, intervals


def aggregate(summary):
    pooled = []
    for size in SIZES:
        for init in INITS:
            for group in GROUPS:
                group_rows = [
                    row for row in summary if row["size"] == size and row["init"] == init and row["group"] == group
                ]
                if not group_rows:
                    continue
                result = {
                    "size": size, "training_photos": group_rows[0]["training_photos"],
                    "init": init, "group": group, "n_seeds": len(group_rows),
                    "visible_points_per_seed": group_rows[0]["visible_points"],
                    "pooled_hits_at_010": sum(row["hits_at_010"] for row in group_rows),
                }
                values = [r["pck_at_010"] for r in group_rows]
                pooled.append({
                    **result, "mean_pck_at_010": float(np.mean(values)),
                    "seed_sd_pck_at_010": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                })
    return pooled


def plot_learning(aggregated, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True, constrained_layout=True)
    for ax, group in zip(axes, ("paws", "knees")):
        for init, color in zip(INITS, ("#386588", "#bf5140")):
            pts = sorted(
                (row for row in aggregated if row["init"] == init and row["group"] == group),
                key=lambda row: row["training_photos"],
            )
            ax.errorbar(
                [row["training_photos"] for row in pts], [100 * row["mean_pck_at_010"] for row in pts],
                yerr=[100 * (row["seed_sd_pck_at_010"] or 0) for row in pts],
                color=color, label=LABELS[init], marker="o", linewidth=2, capsize=4,
            )
        ax.set(title=f"Visible {group}", xlabel="Unique training photos")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Test PCK@0.10 (%)")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle("Domestic cat pose estimation · AP-10K split 1\nMean ± one standard deviation across seeds")
    fig.savefig(out / "learning_curves.png", dpi=180)
    plt.close(fig)


def export_training_histories(root, items, out):
    colors = {11: "#285F90", 22: "#B74544", 33: "#197959"}
    styles = {11: "-", 22: "--", 33: ":"}
    histories, comparisons, curves = [], [], {}
    for item in items:
        run = item["run"]
        history = read_json(root / item["directory"] / "history.json")
        best, last = item["result"]["best"], history[-1]
        loss_fell = last["train_loss"] < best["train_loss"]
        drop_hits = best["val_paw_hits"] - last["val_paw_hits"]
        late_checks_below = (
            len(history) > 1 and all(
                row["step"] > best["step"] and row["val_paw_hits"] < best["val_paw_hits"]
                for row in history[-2:]
            )
        )
        if loss_fell and late_checks_below:
            pattern = "sustained_late_decline"
        elif loss_fell and drop_hits:
            pattern = "one_final_decline"
        elif drop_hits:
            pattern = "validation_decline_without_lower_training_loss"
        else:
            pattern = "no_final_validation_decline"
        base = {"size": run["size"], "training_photos": run["photos"], "init": run["init"], "seed": run["seed"]}
        histories.extend({**base, **row, "best": row["step"] == best["step"]} for row in history)
        comparisons.append({
            **base, "best_step": best["step"], "last_step": last["step"],
            "visible_validation_paws": best["val_paw_total"], "best_val_hits": best["val_paw_hits"],
            "last_val_hits": last["val_paw_hits"], "best_val_pck": best["val_paw_pck"],
            "last_val_pck": last["val_paw_pck"], "last_minus_best_pp": -100.0 * drop_hits / best["val_paw_total"],
            "penultimate_val_hits": history[-2]["val_paw_hits"] if len(history) > 1 else None,
            "best_step_train_loss": best["train_loss"], "last_step_train_loss": last["train_loss"],
            "last_minus_best_train_loss": last["train_loss"] - best["train_loss"], "pattern": pattern,
        })
        curves[run["size"], run["init"], run["seed"]] = (history, best, run["photos"])
    write_csv(out / "training_histories.csv", histories)
    write_csv(out / "best_vs_last.csv", comparisons)

    sizes = [size for size in ("25", "50", "100", "all") if any(key[0] == size for key in curves)]
    for metric_key, filename, title, y_label, factor in (
        ("train_loss", "training_loss.png", "Training heatmap loss", "Joint-weighted MSE (log scale)", 1),
        ("val_paw_pck", "validation_paw_pck.png", "Validation paw hit rate", "Visible-paw PCK@0.10 (%)", 100),
    ):
        fig, axes = plt.subplots(
            2, len(sizes), figsize=(max(6, 3.5 * len(sizes)), 7.5), squeeze=False,
            sharex=True, sharey=True, constrained_layout=True,
        )
        legend = {}
        for row_index, init in enumerate(INITS):
            for col, size in enumerate(sizes):
                ax = axes[row_index, col]
                notes = []
                for seed in SEEDS:
                    group = curves.get((size, init, seed))
                    if group is None:
                        continue
                    history, best, photos = group
                    line, = ax.plot(
                        [row["step"] for row in history], [factor * row[metric_key] for row in history],
                        color=colors[seed], linestyle=styles[seed], label=f"Seed {seed}", linewidth=1.8,
                    )
                    legend[seed] = line
                    ax.scatter(best["step"], factor * best[metric_key], marker="D", s=28, color=colors[seed], zorder=3)
                    last = history[-1]
                    ax.scatter(last["step"], factor * last[metric_key], s=24, facecolors="none",
                               edgecolors=colors[seed], zorder=3)
                    notes.append(f"{seed}→{best['step']}")
                if not notes:
                    ax.set_visible(False)
                    continue
                ax.set_title(f"{photos} photos  ·  {LABELS[init]}\nBest steps  {'   '.join(notes)}", fontsize=9.5)
                ax.set_xticks([100, 400, 700, 1000])
                ax.grid(axis="y", alpha=0.25)
                if metric_key == "train_loss":
                    ax.set_yscale("log")
                if col == 0:
                    ax.set_ylabel(y_label)
                if row_index == 1:
                    ax.set_xlabel("Optimizer step")
        fig.suptitle(f"{title} · AP-10K domestic cats")
        markers = [
            Line2D([], [], color="#36424B", marker="D", linestyle="none", label="Selected checkpoint"),
            Line2D([], [], color="#36424B", marker="o", markerfacecolor="none",
                   linestyle="none", label="Last checkpoint"),
        ]
        fig.legend(handles=[*legend.values(), *markers], loc="outside lower center",
                   ncols=5 if len(sizes) > 1 else 3, frameon=False, fontsize=9)
        fig.savefig(out / filename, dpi=175)
        plt.close(fig)


def error_cases(data, outputs, out, difficult_sample):
    examples = {}
    difficulty_key = difficult_sample["image_id"], difficult_sample["annotation_id"]
    difficulty_joint = difficult_sample["joint_index"]
    flip_paw = {7: 10, 10: 7, 13: 16, 16: 13}
    for label, values in outputs.items():
        for row, points, scores in values:
            err, vis = errors_for(row, points)
            truth = np.asarray(row["keypoints"])
            for j in PAWS:
                if not vis[j]:
                    continue
                tags = []
                if err[j] <= 0.025:
                    tags.append("accurate")
                if (row["image_id"], row["annotation_id"]) == difficulty_key and j == difficulty_joint:
                    tags.append("occlusion_context")
                other = [i for i in PAWS if i != j and vis[i]]
                if other and err[j] > 0.10 and np.isfinite(points[j]).all():
                    near = min(other, key=lambda i: np.linalg.norm(points[j] - truth[i, :2]))
                    if np.linalg.norm(points[j] - truth[near, :2]) <= 0.10 * max(row["bbox"][2:]):
                        tags.append("left_right_confusion" if near == flip_paw[j] else "front_back_confusion")
                for tag in tags:
                    examples.setdefault(tag, []).append((label, row, j, err[j], points, scores))
    directory = out / "error_cases"
    table = []
    chosen_images = set()
    for tag in ("left_right_confusion", "front_back_confusion", "occlusion_context", "accurate"):
        options = sorted(examples.get(tag, []), key=lambda item: (item[3], item[1]["image_id"]), reverse=tag != "accurate")
        option = next((candidate for candidate in options if candidate[1]["image_id"] not in chosen_images),
                      options[0] if options else None)
        if option is None:
            continue
        init, row, j, error, points, scores = option
        chosen_images.add(row["image_id"])
        truth = np.asarray(row["keypoints"])
        matrix = affine(row["bbox"])
        photo = cv2.warpAffine(read_rgb(Path(data) / row["file"]), matrix, (WIDTH, HEIGHT))
        gt_image = skeleton(photo, transform(truth[:, :2], matrix), truth[:, 2] == 2, j)
        pred_image = skeleton(photo, transform(points, matrix), np.isfinite(scores), j)
        overlay = np.concatenate((gt_image, pred_image), axis=1)
        caption = np.full((27, overlay.shape[1], 3), 24, dtype=np.uint8)
        cv2.putText(caption, f"AP-10K {row['image_id']}  |  Label / prediction", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
        missing_labels = [JOINTS[i] for i in PAWS if truth[i, 2] == 0]
        filename = f"{tag}_{row['image_id']}.jpg"
        save_rgb(directory / filename, np.concatenate((overlay, caption)))
        table.append({
            "category": tag, "image_id": row["image_id"], "annotation_id": row["annotation_id"],
            "init": init, "joint": JOINTS[j], "visibility": int(truth[j, 2]),
            "normalized_error": finite_number(error), "unlabeled_paws": ", ".join(missing_labels),
            "overlay": filename,
        })
    write_csv(directory / "index.csv", table)


def evaluate(root, out, split, repeats):
    lock = read_json(out / "test_lock.json")
    if lock["source_sha256"] != source_hash() or sha256(lock["storyboard"]) != lock["storyboard_sha256"]:
        raise ValueError("Locked source code or storyboard changed")
    if repeats != lock["bootstrap_repetitions"]:
        raise ValueError("Bootstrap repetitions differ from the test lock")
    config = lock["runs"][0]["run"]["config"]
    if sha256(config["manifest"]) != lock["manifest_sha256"]:
        raise ValueError("Locked manifest changed")
    rows = read_json(config["manifest"])["splits"][split]
    for filename, expected in {row["file"]: row["image_sha256"] for row in rows}.items():
        if sha256(Path(config["data"]) / filename) != expected:
            raise ValueError(f"Test image content changed: {filename}")
    summary, per_joint, all_predictions = [], [], []
    arrays, visual_examples = {}, {}
    visible = np.asarray([row["keypoints"] for row in rows])[:, :, 2] == 2
    device = configure_device(config)
    for item in lock["runs"]:
        name, run = item["directory"], item["run"]
        path = root / name / "best.pt"
        if sha256(path) != item["best_sha256"]:
            raise ValueError(f"Checkpoint changed for {name}")
        outputs = infer(path, config["data"], rows, config, device)
        errors = np.stack([errors_for(row, points)[0] for row, points, _ in outputs])
        key = run["size"], run["init"], run["seed"]
        arrays[key] = errors
        base = dict(zip(("size", "init", "seed"), key))
        base.update(training_photos=run["photos"], best_step=item["result"]["best"]["step"])
        summary.extend({**base, "group": group, **metric(errors, visible, joints)}
                       for group, joints in GROUPS.items())
        per_joint.extend({**base, "joint": name, **metric(errors, visible, (i,))}
                         for i, name in enumerate(JOINTS))
        all_predictions.extend(prediction for output in outputs for prediction in row_predictions(base, output))
        if run["size"] == "100" and run["seed"] == 11:
            visual_examples[run["init"]] = outputs
        print(f"Scored {item['directory']}", flush=True)
    paired, intervals = pair_analysis(arrays, rows, visible, repeats, lock["bootstrap_seed"])
    aggregated = aggregate(summary)
    for name, table in (
        ("summary", summary), ("per_joint", per_joint), ("predictions", all_predictions),
        ("paired_seeds", paired), ("conditions", aggregated), ("confidence_intervals", intervals),
    ):
        write_csv(out / f"{name}.csv", table)
    plot_learning(aggregated, out)
    export_training_histories(root, lock["runs"], out)
    difficult_sample = read_json(lock["storyboard"])["samples"][1]
    error_cases(config["data"], visual_examples, out, difficult_sample)
    write_json(out / "resource_usage.json", {
        "runs": [
            {"name": i["directory"], "best_checkpoint": str(root / i["directory"] / "best.pt"), **i["result"]}
            for i in lock["runs"]
        ],
        "training_plus_validation_gpu_hours": sum(
            i["result"]["measured_training_plus_validation_hours"] for i in lock["runs"]
        ),
        "maximum_allocated_bytes": max(i["result"]["peak_allocated_bytes"] for i in lock["runs"]),
    })
    print(f"Saved test results to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--storyboard", type=Path, default=Path("demo/storyboard.json"))
    args = parser.parse_args()
    if args.lock:
        make_lock(args.runs, args.out, args.storyboard, args.allow_incomplete)
    else:
        evaluate(args.runs, args.out, args.split, args.bootstrap)


if __name__ == "__main__":
    main()

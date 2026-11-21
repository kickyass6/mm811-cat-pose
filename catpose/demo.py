import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import JOINTS, read_json, sha256, write_json
from .geometry import read_rgb, transform
from .overlay import LEFT, RIGHT, skeleton


FRAME = (1280, 720)
BG, INK, MUTED = "#101922", "#F1F2EA", "#B4C4CD"
EDGE, LIME, TRUTH = "#334651", (250, 255, 114), (240, 110, 178)
DISPLAY_PEAK = 0.07
SHORT = {"imagenet": "ImageNet classification", "coco_pose": "COCO human pose"}
PAW_LABEL = {13: "left rear paw", 16: "right rear paw"}


def font(size, weight=""):
    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return ImageFont.truetype(str(folder / f"DejaVuSans{weight}.ttf"), size)


def canvas(title, subtitle=None):
    page = Image.new("RGB", FRAME, BG)
    d = ImageDraw.Draw(page)
    d.text((46, 29), title, fill=INK, font=font(43, "-Bold"))
    if subtitle:
        d.text((47, 91), subtitle, fill=MUTED, font=font(21))
    return page, d


def footer(d, text):
    d.line((46, 659, 1234, 659), fill=EDGE, width=2)
    d.text((46, 675), text, font=font(18), fill=MUTED)


def photo_crop(image, box, width, height, center=None, zoom=None):
    x, y, w, h = box
    center = (x + w / 2, y + h / 2) if center is None else center
    span = max(w, h * width / height) * 1.12 if zoom is None else zoom
    scale = width / span
    matrix = np.asarray([
        [scale, 0, width / 2 - scale * center[0]],
        [0, scale, height / 2 - scale * center[1]],
    ])
    photo = cv2.warpAffine(image, matrix, (width, height), borderValue=(17, 27, 38))
    return photo, matrix


def side_panel(row, image, output, joint, reveal, width=578, height=392):
    photo, matrix = photo_crop(image, row["bbox"], width, height)
    points, scores = output
    mapped = transform(points, matrix)
    show = np.isfinite(scores) & (scores >= DISPLAY_PEAK)
    if reveal and np.isfinite(points[joint]).all():
        show[joint] = True
    photo = skeleton(photo, mapped, show, joint)
    if reveal:
        gt = transform(np.asarray(row["keypoints"])[joint, :2][None], matrix)[0]
        x, y = gt.round().astype(int)
        if np.isfinite(mapped[joint]).all():
            point = tuple(mapped[joint].round().astype(int))
            cv2.line(photo, point, (x, y), (14, 23, 32), 6, cv2.LINE_AA)
            cv2.line(photo, point, (x, y), TRUTH, 3, cv2.LINE_AA)
        cv2.drawMarker(photo, (x, y), (14, 23, 32), cv2.MARKER_CROSS, 25, 7, cv2.LINE_AA)
        cv2.drawMarker(photo, (x, y), TRUTH, cv2.MARKER_CROSS, 25, 4, cv2.LINE_AA)
    return Image.fromarray(photo)


def accuracy_text(row, joint, output):
    point, score = output[0][joint], output[1][joint]
    if not np.isfinite(point).all():
        return "Missing prediction", False
    gt = np.asarray(row["keypoints"][joint][:2])
    pixels = math.dist(point, gt)
    normalized = pixels / max(row["bbox"][2:])
    weak = " · weak peak" if score < DISPLAY_PEAK else ""
    result = "HIT" if normalized <= 0.10 else "MISS"
    return f"{pixels:.1f} px  ·  {100 * normalized:.1f}% of box  ·  {result}{weak}", normalized <= 0.10


def compare_page(sample, row, image, paired, storyboard, mode):
    joint = sample["joint_index"]
    reveal = mode not in ("blind", "vote")
    title, subtitle = {
        "blind": ("Which model found the paw?", "Both models trained on the same 100 cat photos."),
        "vote": ("Vote now: Model A or Model B?", "Use the yellow ring to compare the two predictions."),
        "reveal": ("Reveal: the labeled paw", "The pink cross is the human annotation. The pink line shows the error."),
        "difficult": ("A harder pose", "The cat is curled up. Foreshortening makes the paws harder to tell apart."),
        "final": ("One more test photo", "The supplied box selects one cat in this multi-cat photo."),
    }[mode]
    page, d = canvas(title, subtitle)
    target = PAW_LABEL[joint]
    for slot, x in zip(("A", "B"), (46, 656)):
        init = storyboard["mapping"][slot]
        label = f"Model {slot}  ·  {SHORT[init]}" if reveal else f"Model {slot}"
        d.text((x, 139), label, fill=INK, font=font(24, "-Bold"))
        panel = side_panel(row, image, paired[init], joint, reveal)
        page.paste(panel, (x, 178))
        d.rectangle((x, 178, x + 578, 570), outline=EDGE, width=2)
        if reveal:
            message, hit = accuracy_text(row, joint, paired[init])
            d.text((x, 588), message, fill=LIME if hit else "#FFB9B3", font=font(20, "-Bold"))
        else:
            d.text((x, 588), f"Yellow ring = predicted {target}", fill=LIME, font=font(21))
    footer(d, f"AP-10K photo {sample['image_id']}  ·  Cyan left, amber right  ·  Weak peaks hidden except a revealed target")
    return page


def opening(sample, row, image):
    page, d = canvas("Which model found the paw?", "Two starting points. The same 100 labeled cat photos.")
    whole, _ = photo_crop(image, row["bbox"], 714, 435)
    center = row["keypoints"][sample["joint_index"]][:2]
    detail, _ = photo_crop(image, row["bbox"], 440, 350, center=center, zoom=max(row["bbox"][2:]) * 0.37)
    page.paste(Image.fromarray(whole), (46, 169))
    page.paste(Image.fromarray(detail), (794, 169))
    d.rectangle((794, 169, 1233, 518), outline=EDGE, width=2)
    target = PAW_LABEL[sample["joint_index"]]
    d.text((795, 540), f"Look for the cat’s {target}.", font=font(22, "-Bold"), fill=LIME)
    footer(d, f"AP-10K photo {sample['image_id']}  ·  Supplied cat box  ·  Single-frame pose estimation")
    return page


def summary_page(summary_path, size, seed, lock):
    with summary_path.open(newline="", encoding="utf-8") as stream:
        table = {
            (r["init"], r["group"]): r for r in csv.DictReader(stream)
            if r["size"] == size and int(r["seed"]) == seed
        }
    page, d = canvas("How they scored on the full test set",
                     f"100 training photos per model  ·  seed {seed}  ·  AP-10K split 1")
    test_photos = len({r["image_id"] for r in read_json(lock["runs"][0]["run"]["config"]["manifest"])["splits"]["test"]})
    for group, x in zip(("paws", "knees"), (46, 656)):
        d.rounded_rectangle((x, 150, x + 578, 546), radius=14, fill="#182733", outline=EDGE, width=2)
        d.text((x + 24, 174), f"Visible {group}", fill=INK, font=font(28, "-Bold"))
        values = []
        for i, (init, color) in enumerate(zip(SHORT, (LEFT, RIGHT))):
            record = table[init, group]
            score = float(record["pck_at_010"])
            values.append(score)
            y = 242 + 103 * i
            d.text((x + 24, y), SHORT[init], fill=MUTED, font=font(22))
            d.text((x + 554, y), f"{score * 100:.1f}%  ({record['hits_at_010']}/{record['visible_points']})",
                   fill=INK, font=font(22, "-Bold"), anchor="ra")
            d.rounded_rectangle((x + 24, y + 40, x + 554, y + 55), radius=7, fill=EDGE)
            if score:
                d.rounded_rectangle((x + 24, y + 40, x + 24 + 530 * score, y + 55), radius=7, fill=color)
        d.text((x + 24, 474), f"Pose minus classification: {(values[1] - values[0]) * 100:+.1f} pp",
               fill=LIME, font=font(24, "-Bold"))
    d.text((46, 584), "A hit falls within 10% of the longest side of the labeled box.",
           fill=INK, font=font(25))
    footer(d, f"{test_photos} held-out photos  ·  Visible annotations only  ·  All model peaks scored")
    return page


def read_points(path, samples, size, seed):
    result = {}
    lookup = {(s["image_id"], s["annotation_id"]) for s in samples}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = int(row["image_id"]), int(row["annotation_id"])
            if row["size"] != size or int(row["seed"]) != seed or key not in lookup:
                continue
            pair = result.setdefault(key, {})
            coords, scores = pair.setdefault(row["init"], (np.full((17, 2), np.nan), np.full(17, np.nan)))
            j = JOINTS.index(row["joint"])
            if row["raw_peak"]:
                coords[j] = [float(row["pred_x"]), float(row["pred_y"])]
                scores[j] = float(row["raw_peak"])
    return result


def render(runs, results, storyboard_path, size, seed, output):
    lock = read_json(results / "test_lock.json")
    storyboard = read_json(storyboard_path)
    if sha256(storyboard_path) != lock["storyboard_sha256"]:
        raise ValueError("Storyboard differs from the locked selection")
    if storyboard["size"] != size or storyboard["seed"] != seed:
        raise ValueError("Requested model pair differs from the storyboard")
    config = lock["runs"][0]["run"]["config"]
    dataset = read_json(config["manifest"])
    test = {(r["image_id"], r["annotation_id"]): r for r in dataset["splits"]["test"]}
    sources = {}
    points = read_points(results / "predictions.csv", storyboard["samples"], size, seed)
    for sample in storyboard["samples"]:
        key = sample["image_id"], sample["annotation_id"]
        row = test[key]
        if row["keypoints"][sample["joint_index"]][2] != 2:
            raise ValueError(f"Storyboard target is not visible in {key}")
        image = read_rgb(Path(config["data"]) / row["file"])
        sources[key] = sample, row, image, points[key], storyboard
    intro, difficult, final = sources.values()
    frames = {
        "question": opening(*intro[:3]),
        "blind": compare_page(*intro, "blind"),
        "vote": compare_page(*intro, "vote"),
        "reveal": compare_page(*intro, "reveal"),
        "difficult": compare_page(*difficult, "difficult"),
        "summary": summary_page(results / "summary.csv", size, seed, lock),
        "final": compare_page(*final, "final"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Install FFmpeg with the libx264 encoder")
    selected = [i for i in lock["runs"] if i["run"]["size"] == size and i["run"]["seed"] == seed]
    weights = [
        {"path": str(runs / i["directory"] / "best.pt"), "sha256": i["best_sha256"]}
        for i in selected
    ]
    for item in weights:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Demo checkpoint differs from the test lock: {item['path']}")
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        arguments = []
        for index, (stage, still) in enumerate(frames.items()):
            path = temp / f"{index}.png"
            still.save(path)
            arguments += ["-loop", "1", "-framerate", "25", "-t", str(storyboard["duration_seconds"][stage]),
                          "-i", str(path)]
        labels = "".join(f"[{i}:v]" for i in range(len(frames)))
        filtergraph = f"{labels}concat=n={len(frames)}:v=1:a=0,format=yuv420p"
        pending = temp / "rendered.mp4"
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *arguments,
            "-filter_complex_threads", "1", "-filter_complex", filtergraph, "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-movflags", "+faststart", str(pending),
        ], check=True)
        shutil.copyfile(pending, output)
    subprocess.run([ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"], check=True)
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate:"
        "format=duration", "-of", "json", str(output),
    ], text=True)
    write_json(output.parent / "provenance.json", {
        "command": f"python -m catpose.demo --runs {runs} --size {size} --seed {seed} "
                   f"--results {results} --storyboard {storyboard_path} --out {output}",
        "sha256": sha256(output), "manifest_sha256": sha256(config["manifest"]),
        "checkpoint_files": weights, "storyboard_sha256": sha256(storyboard_path),
        "photo_sources": [r["file"] for _, r, _, _, _ in sources.values()],
        "photo_source_note": storyboard["photo_source"],
        "source_record": storyboard["source_record"], "media_scope": storyboard["media_scope"],
        "individual_photo_source": storyboard["individual_photo_source"], "video": json.loads(probe),
    })
    print(f"Rendered and decoded {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--size", default="100")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--storyboard", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.runs, args.results, args.storyboard, args.size, args.seed, args.out)


if __name__ == "__main__":
    main()

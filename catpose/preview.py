"""Export predicted joints on the untouched validation images for the pilot review."""
import argparse
from pathlib import Path

import numpy as np

from .common import INITS, PAWS, read_json, run_directory, write_json
from .evaluate import infer
from .geometry import read_rgb
from .metrics import errors_for
from .overlay import save_rgb, skeleton
from .train import configure_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--size", default="100")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=Path("logs/pilot_validation"))
    args = parser.parse_args()
    index = []
    for init in INITS:
        directory = run_directory(args.runs, args.size, init, args.seed)
        config = read_json(directory / "run.json")["config"]
        rows = read_json(config["manifest"])["splits"]["val"]
        selected = sorted(rows, key=lambda r: (-sum(r["keypoints"][p][2] == 2 for p in PAWS),
                                                r["image_id"]))[:6]
        outputs = infer(directory / "best.pt", config["data"], selected, config, configure_device(config))
        for row, xy, scores in outputs:
            source = read_rgb(Path(config["data"]) / row["file"])
            visible = np.asarray(row["keypoints"])[:, 2] == 2
            truth = skeleton(source, np.asarray(row["keypoints"])[:, :2], visible)
            pred = skeleton(source, xy, np.isfinite(scores))
            combined = np.concatenate((truth, pred), axis=1)
            name = f"{init}_{row['image_id']}_{row['annotation_id']}.jpg"
            save_rgb(args.out / name, combined)
            errors, _ = errors_for(row, xy)
            index.append({"image_id": row["image_id"], "init": init, "overlay": name,
                          "paw_errors": [float(errors[i]) if visible[i] else None for i in PAWS]})
    write_json(args.out / "index.json", index)


if __name__ == "__main__":
    main()

import argparse
import fcntl
from pathlib import Path

from .common import INITS, SEEDS, SIZES, config_from, read_json, run_directory
from .train import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--inits", nargs="+", choices=INITS, default=INITS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = config_from(args.config)
    manifest = read_json(config["manifest"])
    root = Path(config["runs"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".matrix.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"A matrix job is active in {root}")
        for size in args.sizes:
            for seed in args.seeds:
                for init in args.inits:
                    destination = run_directory(root, size, init, seed)
                    fit(config, manifest, size, init, seed, destination, args.resume)


if __name__ == "__main__":
    main()

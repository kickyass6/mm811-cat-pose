import argparse
import statistics
import time

import torch

from .common import config_from, read_json, write_json
from .data import make_loaders, training_rows
from .train import configure_device, device_record, loss_step, setup, sync


def benchmark(config, warmup, steps):
    device = configure_device(config)
    manifest = read_json(config["manifest"])
    _, rows = training_rows(manifest, "100", 11)
    model, optimizer, scaler, _ = setup(config, "imagenet", 11, device)
    loader, _ = make_loaders(config, rows, manifest["splits"]["val"], 11, end=warmup + steps)
    iterator = iter(loader)
    times = []
    for step in range(warmup + steps):
        if step == warmup and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        loss_step(model, optimizer, scaler, iterator, config["accumulate"], device)
        sync(device)
        if step >= warmup:
            times.append(time.perf_counter() - start)
    average = statistics.mean(times)
    return {
        "device": device_record(device), "config": config, "warmup_steps": warmup,
        "measured_steps": steps, "mean_seconds_per_step": average,
        "median_seconds_per_step": statistics.median(times), "individual_step_seconds": times,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "projected_hours_training_only": average * config["steps"] * 24 / 3600,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out", default="logs/profile.json")
    args = parser.parse_args()
    report = benchmark(config_from(args.config), args.warmup, args.steps)
    write_json(args.out, report)
    print({k: v for k, v in report.items() if k != "individual_step_seconds"}, flush=True)


if __name__ == "__main__":
    main()

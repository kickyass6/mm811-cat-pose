import math
import os
import time
from pathlib import Path

import cv2
import torch

from .common import PAWS, read_json, sha256, write_json
from .data import make_loaders, training_rows
from .metrics import counts, error_arrays, predictions
from .model import SOURCES, initialized


def configure_device(config):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose one CUDA device via CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return device


def device_record(device):
    if device.type != "cuda":
        return {"type": device.type, "torch": str(torch.__version__)}
    spec = torch.cuda.get_device_properties(device)
    return {
        "type": "cuda", "name": spec.name, "memory_bytes": spec.total_memory,
        "compute_capability": [spec.major, spec.minor], "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def setup(config, init, seed, device):
    model, loading = initialized(init, seed, config["weights"])
    model = model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["backbone_lr"]},
        {"params": model.head.parameters(), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, init_scale=4096, enabled=amp)
    return model, optimizer, scaler, loading


def loss_step(model, optimizer, scaler, iterator, accumulate, device):
    model.training_with_fixed_bn()
    block = [next(iterator) for _ in range(accumulate)]
    denominator = sum(int(mask.sum()) for _, _, mask in block)
    if not denominator:
        raise RuntimeError("An entire effective batch has no supervised joints")
    for retry in range(8):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for inputs, target, valid in block:
            inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
            target, valid = target.to(device, non_blocking=True), valid.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                predicted = model(inputs)
            per_joint = (predicted.float() - target).square().mean(dim=(2, 3))
            loss = (per_joint * valid).sum() / denominator
            scaler.scale(loss).backward()
            losses.append(loss.detach())
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            value = float(torch.stack(losses).sum().item())
            if not math.isfinite(value):
                raise RuntimeError("Training loss is nonfinite")
            return value, denominator, retry
    raise RuntimeError("FP16 gradients remained nonfinite after eight attempts")


def ratio(step, config):
    if step <= config["warmup"]:
        return step / config["warmup"]
    progress = (step - config["warmup"]) / (config["steps"] - config["warmup"])
    return 0.5 * (1 + math.cos(math.pi * progress))


def atomic_save(path, value):
    pending = path.with_suffix(".pt.tmp")
    torch.save(value, pending)
    os.replace(pending, path)


def training_logic_hash():
    from hashlib import sha256 as make_hash

    digest = make_hash()
    root = Path(__file__).parent
    for name in ("common", "data", "geometry", "metrics", "model", "train"):
        digest.update((root / f"{name}.py").read_bytes())
    return digest.hexdigest()


def fit(config, manifest, size, init, seed, directory, resume=False):
    device = configure_device(config)
    ids, rows = training_rows(manifest, size, seed)
    identity = {
        "size": size, "photos": len(ids), "instances": len(rows), "init": init, "seed": seed,
        "config": config, "manifest_sha256": sha256(config["manifest"]),
        "weights_sha256": SOURCES[init]["sha256"], "training_source_sha256": training_logic_hash(),
        "device": device_record(device),
    }
    run_path = directory / "run.json"
    done = directory / "complete.json"
    best_path, last_path = directory / "best.pt", directory / "last.pt"
    directory.mkdir(parents=True, exist_ok=True)
    if run_path.exists() and read_json(run_path) != identity:
        raise ValueError(f"Run identity changed in {directory}; use a fresh runs directory")
    if done.exists() and best_path.exists() and last_path.exists():
        print(f"Complete, skipping {directory}", flush=True)
        return
    if last_path.exists() and not resume:
        raise ValueError(f"Resume {directory} with --resume")
    write_json(run_path, identity)
    write_json(directory / "selection.json", {
        "image_ids": ids, "annotation_ids": [row["annotation_id"] for row in rows],
    })
    model, optimizer, scaler, loading = setup(config, init, seed, device)
    write_json(directory / "initialization.json", loading)
    history, best, start = [], None, 0
    train_seconds, val_seconds, wall_seconds, peak = 0.0, 0.0, 0.0, 0
    if last_path.exists():
        saved = torch.load(last_path, weights_only=True, map_location="cpu")
        if saved["identity"] != identity:
            raise ValueError(f"Checkpoint identity differs in {directory}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, best, history = saved["step"], saved["best"], saved["history"]
        train_seconds, val_seconds = saved["train_seconds"], saved["val_seconds"]
        wall_seconds, peak = saved["wall_seconds"], saved["peak_allocated_bytes"]
        if not best_path.exists():
            raise ValueError(f"Missing best.pt for resumable run {directory}")
        print(f"Resuming {directory} at step {start}", flush=True)

    loader, val = make_loaders(config, rows, manifest["splits"]["val"], seed, start=start)
    iterator = iter(loader)
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    segment_since = time.perf_counter()
    accumulated_loss, accumulated_weight, retries = 0.0, 0, 0
    for step in range(start + 1, config["steps"] + 1):
        for group, base in zip(optimizer.param_groups, (config["backbone_lr"], config["head_lr"])):
            group["lr"] = base * ratio(step, config)
        loss, weight, retry = loss_step(model, optimizer, scaler, iterator, config["accumulate"], device)
        accumulated_loss += loss * weight
        accumulated_weight += weight
        retries += retry
        if step % config["validate_every"] and step != config["steps"]:
            continue
        current = time.perf_counter()
        elapsed = current - segment_since
        train_seconds += elapsed
        errors, visible = error_arrays(predictions(model, val.dataset, device, scaler.is_enabled(), loader=val))
        hits, total = counts(errors, visible, PAWS)
        if total == 0:
            raise ValueError("Validation split has no visible paws")
        sync(device)
        validation_time = time.perf_counter() - current
        val_seconds += validation_time
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device))
        row = {
            "step": step, "draws": step * 8, "train_loss": accumulated_loss / accumulated_weight,
            "val_paw_hits": hits, "val_paw_total": total, "val_paw_pck": hits / total,
            "train_seconds": elapsed, "val_seconds": validation_time, "amp_retries": retries,
        }
        history.append(row)
        if best is None or hits > best["val_paw_hits"]:
            best = row
            atomic_save(best_path, {"identity": identity, "step": step, "model": model.state_dict()})
        wall_seconds += time.perf_counter() - segment_since
        atomic_save(last_path, {
            "identity": identity, "step": step, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "best": best, "history": history, "train_seconds": train_seconds,
            "val_seconds": val_seconds, "wall_seconds": wall_seconds, "peak_allocated_bytes": peak,
        })
        write_json(directory / "history.json", history)
        print(f"{directory.name} step {step:4d}  loss {row['train_loss']:.6f}  "
              f"val paws {hits}/{total}  training {elapsed:.1f}s", flush=True)
        accumulated_loss, accumulated_weight, retries = 0.0, 0, 0
        segment_since = time.perf_counter()
    write_json(done, {
        "best": best, "best_sha256": sha256(best_path), "step": config["steps"],
        "train_seconds": train_seconds, "val_seconds": val_seconds,
        "measured_training_plus_validation_hours": (train_seconds + val_seconds) / 3600,
        "peak_allocated_bytes": peak, "total_instance_draws": config["steps"] * 8,
    })

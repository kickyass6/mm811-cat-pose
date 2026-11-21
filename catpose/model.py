import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50

from .common import JOINTS, sha256


SOURCES = {
    "imagenet": {
        "url": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "sha256": "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
        "version": "torchvision ResNet50_Weights.IMAGENET1K_V1",
        "pretraining": "ImageNet-1K image classification",
    },
    "coco_pose": {
        "url": (
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
            "td-hm_res50_8xb64-210e_coco-256x192-04af38ce_20220923.pth"
        ),
        "sha256": "04af38cedaab219648333b3e3bf3040ce88ab7e5a613fcb85652c40188d13a95",
        "version": "MMPose td-hm_res50_8xb64-210e_coco-256x192, 2022-09-23 release",
        "pretraining": "torchvision ResNet-50 initialization followed by COCO human keypoint training",
    },
}


class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        channels = (2048, 256, 256, 256)
        layers = []
        for source, target in zip(channels, channels[1:]):
            layers.extend([
                nn.ConvTranspose2d(source, target, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(target), nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(channels[-1], len(JOINTS), 1))
        self.head = nn.Sequential(*layers)
        for layer in self.head:
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, image):
        net = self.backbone
        features = net.maxpool(net.relu(net.bn1(net.conv1(image))))
        return self.head(net.layer4(net.layer3(net.layer2(net.layer1(features)))))

    def training_with_fixed_bn(self):
        self.train()
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class _IgnoredHistory:
    """Accept the inert MMPose training-log field while reading tensors."""


def load_human_checkpoint(path):
    numpy_globals = [
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.ndarray, np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType,
        np.dtypes.Int64DType, np.dtypes.Int32DType, np.dtypes.UInt8DType,
        (_IgnoredHistory, "mmengine.logging.history_buffer.HistoryBuffer"),
    ]
    with torch.serialization.safe_globals(numpy_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialized(init, seed, directory):
    torch.manual_seed(seed)
    model = PoseModel()
    head = state_hash(model.head.state_dict())
    path = Path(directory) / f"{init}.pth"
    source = SOURCES[init]
    if sha256(path) != source["sha256"]:
        raise ValueError(f"Unexpected SHA-256 for {path}")
    if init == "imagenet":
        raw = torch.load(path, map_location="cpu", weights_only=True)
        state = {key: value for key, value in raw.items() if not key.startswith("fc.")}
        metadata = {}
    else:
        bundle = load_human_checkpoint(path)
        prefix = "backbone."
        state = {
            key[len(prefix):]: value for key, value in bundle["state_dict"].items()
            if key.startswith(prefix)
        }
        metadata = {
            key: bundle["meta"].get(key) for key in ("epoch", "iter", "seed", "mmengine_version")
        }
    destination = model.backbone.state_dict()
    filled = []
    if init == "imagenet":
        for key, value in destination.items():
            if key.endswith(".num_batches_tracked") and key not in state:
                state[key] = torch.zeros_like(value)
                filled.append(key)
    if state.keys() != destination.keys():
        raise ValueError(f"Unexpected backbone keys: {state.keys() - destination.keys()}; "
                         f"missing: {destination.keys() - state.keys()}")
    for key, value in state.items():
        if value.shape != destination[key].shape or value.dtype != destination[key].dtype:
            raise ValueError(f"Shape or dtype mismatch in {init} at {key}")
    model.backbone.load_state_dict(state, strict=True)
    loaded = model.backbone.state_dict()
    if any(not torch.equal(loaded[name], value) for name, value in state.items()):
        raise ValueError("Loaded backbone differs from the source tensor values")
    if head != state_hash(model.head.state_dict()):
        raise ValueError("Pretrained loading changed the new cat prediction head")
    report = {
        **source, "checkpoint_metadata": metadata, "head_sha256": head,
        "backbone_sha256": state_hash(loaded), "legacy_counters_initialized_to_zero": filled,
        "backbone_tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in state.items()
        },
    }
    return model, report

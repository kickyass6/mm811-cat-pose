# MM811 · Cat paw and knee localization

The experiment compares two ways to initialize the same Simple Baselines ResNet‑50 before training on 25, 50, 100 or all 280 eligible domestic-cat training photos. Both groups start with an identical new head within each matched seed. The primary outcome is visible-paw PCK@0.10 on the official AP‑10K split 1 test set.

## Reproduce the experiment

Use Linux, a single CUDA GPU, FFmpeg with libx264, Python 3.11, and about 25 GB of disk space. The recorded environment uses Ubuntu 20.04, Python 3.11.14, PyTorch 2.7.1+cu118 and torchvision 0.22.1+cu118. The school server ran jobs directly on one idle RTX 2080 SUPER using tmux. Slurm was unavailable on that host.

Create the isolated environment and install the tested package lock. On the course host, which shipped with Python 3.8, the following project environment used uv-managed CPython 3.11.14.

```bash
python3 -m pip install --user 'uv==0.9.18'
~/.local/bin/uv python install 3.11.14
~/.local/bin/uv venv --python 3.11.14 --seed .venv
```

On a host that already provides Python 3.11, use `python3.11 -m venv .venv` instead. Install the recorded packages into either environment.

```bash
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
```

### Data and initialization files

Create the data directory first. Download the 3.4 GB `ap-10k.zip` from the dataset author's link. Save it as `data/ap-10k.zip`. SHA‑256 for the official archive used in this experiment is `420980abb135d6f66bcc8e29f289a46081214016192ae197ad24bc1525c8e62c`. The archive identifies itself as AP‑10K v1.0. The course server retrieved it on November 20, 2026, Edmonton time. [Official AP‑10K download and layout](https://github.com/AlexTheBad/AP-10K#download)

```bash
unzip -n data/ap-10k.zip 'ap-10k/annotations/*split1.json' 'ap-10k/data/*' -d data
mv data/ap-10k data/AP-10K
mkdir -p weights
curl --fail --location https://download.pytorch.org/models/resnet50-0676ba61.pth \
  --output weights/imagenet.pth
curl --fail --location \
  https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192-04af38ce_20220923.pth \
  --output weights/coco_pose.pth
```

The loader verifies the full SHA‑256 of each input and checks every backbone tensor against the source. The older torchvision V1 file packs convolution weights and BN parameters and statistics. It predates the serialized BN batch counters, which load as zero. The MMPose file includes the counters and identifies a 210‑epoch COCO keypoint training run initialized via `torchvision://resnet50`. The upstream configuration uses the torchvision alias, so the underlying original classification filename is not independently recoverable from the checkpoint. [MMPose model listing](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/resnet_coco.yml)

| File | Source | SHA‑256 |
| --- | --- | --- |
| `weights/imagenet.pth` | torchvision ImageNet‑1K V1 classification | `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a` |
| `weights/coco_pose.pth` | MMPose COCO human pose | `04af38cedaab219648333b3e3bf3040ce88ab7e5a613fcb85652c40188d13a95` |

### Run and resume

Expose a single available GPU in your shell with `CUDA_VISIBLE_DEVICES`. Then run the full workflow or the individual stages below. On a fresh checkout, use the included pre-test storyboard with the identical official archive and generated data manifest.

```bash
scripts/run.sh all
```

```bash
scripts/run.sh prepare
scripts/run.sh pilot
scripts/run.sh matrix
scripts/run.sh results
```

The matrix prioritizes the three 100-photo pairs. It skips complete runs of the same configuration and resumes each interrupted run from the last committed validation step. The run record checks source, data-manifest, configuration, weight and device identity before restoring. Each run contains the exact selection, loading report, training history, `best.pt`, `last.pt` and a completion record. The checkpoint choice uses maximum validation paw PCK@0.10 with the earlier step used on ties. The results stage writes a lock over the selected checkpoints, evaluation code and storyboard before testing. The lock check also checks matched heads and training selections.

### Outputs

The repository includes the source, configuration, package lock, and fixed demo storyboard. The commands above recreate the experiment outputs locally using the linked data and initialization files.

| Path | Contents |
| --- | --- |
| `manifests/data_audit.json` | Archive and annotation SHA‑256, counts, visibility and duplicate audit |
| `manifests/manifest.json` | AP‑10K IDs, boxes, annotations, affines and nested subsets |
| `logs/` | Server metadata, overlays, initialization checks, overfit, profile, resume proof and training logs |
| `runs/` | Per-condition checkpoints, training curves and configuration |
| `results/` | Test lock, CSVs, split-size learning curves, best-vs-last diagnostics, error cases and resource usage |
| `demo/storyboard.json` | Example IDs, selection time, timing and seeded A/B assignment |
| `demo/cat_pose_demo.mp4` | Classroom A/B video with English on-screen explanations |
| `demo/provenance.json` | Photo source, hashes, output metadata and rebuild command |
| `docs/literature_evidence.md` | 40 original-paper evidence entries aligned with the proposal |

### Evaluation details

PCK counts Euclidean distance from the labeled original-image point within a fraction of the longest side of the unexpanded box. The primary threshold is 0.10. Test PCK uses visibility flag 2. The normalization error column averages finite predictions and gives its count alongside the count of missing predictions. Missing predictions are failures for PCK. Raw heatmap peaks are stored as descriptive scores. The classroom display threshold has no role in scoring.

The 95% paired percentile interval uses 2,000 bootstrap samples of original image IDs with replacement. Every sampled image keeps its instances, paired models and observed seeds. The seed standard deviation describes variation across the three matched seeds separately from the image bootstrap.

The data audit used SHA‑256 for identical files and combined perceptual and difference-hash thresholds to identify likely near-duplicates. Image inspection checked flipped and unflipped joints in cat-centered crops, including multi-cat and occlusion examples. Some original photo license information is unresolved. The AP‑10K repository states a CC BY 4.0 license, and its archive includes a generic MIT license record with no per-photo source URLs. The demo records the AP‑10K photo IDs and was prepared for the MM811 classroom. [AP‑10K project and license](https://github.com/AlexTheBad/AP-10K#license)

### Best checkpoint and late-training diagnostics

Both training curves mark every selected checkpoint and its final 1,000-step counterpart. Training loss is joint-weighted heatmap MSE over the interval ending at each validation check. Validation records hits, denominator and PCK@0.10. Validation loss was not recorded in this run.

`results/best_vs_last.csv` compares the selected and final checkpoints using training loss and validation PCK. A sustained late decline means the final training loss is lower and both final validation checks are below the selected checkpoint. A final-check decline alone is listed separately. These are descriptive patterns in the same validation set used for checkpoint selection.

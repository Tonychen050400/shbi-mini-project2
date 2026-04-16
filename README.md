# Pascal VOC 2007 Semantic Segmentation: U-Net, DeepLabV3+, SAM2

Comparative study of semantic segmentation methods on Pascal VOC 2007, covering U-Net with ResNet encoders, DeepLabV3+ (from `segmentation-models-pytorch`), and SAM2-Large with a semantic head. Experiments are run at 512x512 with AdamW + cosine annealing.

## Environment

Python 3.10, CUDA 12.1 on a single 4090.

```bash
conda create -n shbi261-project2 python=3.10 -y
conda activate shbi261-project2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib pillow tqdm segmentation-models-pytorch
pip install git+https://github.com/facebookresearch/sam2.git
```

## Dataset

Pascal VOC 2007 segmentation (21 classes, 209 train / 213 val). Place data under `./dataset/VOCdevkit/VOC2007/` or let torchvision auto-download on first run.

## Checkpoint

SAM2-Large weights:
```bash
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Reproducing All 10 Experiments

```bash
python run_10.py
```

This runs U-Net (4 configs), DeepLabV3+ (3 configs), and SAM2-Large (3 configs) sequentially at 512x512, logs to `autoresearch_log.tsv`, and skips any experiment whose checkpoint already exists.

## Individual Commands

Train:
```bash
python scripts/train.py --model unet --backbone resnet50 --loss combined \
    --augment --normalize --epochs 150 --batch-size 8 --lr 1e-4 --img-size 512

python scripts/train.py --model deeplabv3plus --backbone resnet50 --output-stride 16 \
    --loss combined --augment --normalize --epochs 75 --batch-size 8 --lr 5e-5 --img-size 512

python scripts/train.py --model sam --sam-size large --loss combined \
    --augment --normalize --epochs 75 --batch-size 4 --lr 1e-4 --img-size 512
```

Evaluate:
```bash
python scripts/evaluate.py --model sam --sam-size large --normalize --img-size 512 \
    --checkpoint-dir checkpoints/sam_large_combined_aug_norm
```

Generate cross-experiment comparison plots:
```bash
python scripts/compare_results.py
```

## Evaluation Metrics

Mean IoU, Mean Dice, Pixel Accuracy, HD95 (boundary), per-class IoU and accuracy, confusion matrices, training curves, best/worst prediction mosaics.

## Ablations

1. **Backbone capacity** (U-Net R18 vs R50)
2. **Loss function** (Combined CE+Dice vs CE-only)
3. **Data augmentation** (with/without)
4. **Output stride** (OS=16 vs OS=8, DeepLabV3+)
5. **SAM2 encoder regime** (frozen vs fine-tuned)

## Repository Layout

```
mini-project-2/
├── run_10.py                   # Master runner for all 10 experiments
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── compare_results.py
├── models/
│   ├── unet.py
│   ├── deeplabv3.py
│   ├── deeplabv3plus.py
│   └── sam_seg.py
├── src/
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   └── visualize.py
├── checkpoints/                # saved weights + SAM2 checkpoint
├── results/                    # per-experiment metrics.json + plots
├── dataset/                    # Pascal VOC 2007
└── README.md
```

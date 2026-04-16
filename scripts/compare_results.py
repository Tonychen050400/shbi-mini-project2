import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.dataset import VOC_CLASSES


def load_metrics(results_dir):
    path = os.path.join(results_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # 10 experiments matching the example PDF
    experiments = {
        "U-Net R18": "results/unet_resnet18_combined_aug_norm",
        "U-Net R50": "results/unet_resnet50_combined_aug_norm",
        "U-Net R18 no-aug": "results/unet_resnet18_combined_norm",
        "U-Net R18 CE": "results/unet_resnet18_ce_aug_norm",
        "DLv3+ OS16": "results/deeplabv3plus_resnet50_os16_combined_aug_norm",
        "DLv3+ OS8": "results/deeplabv3plus_resnet50_os8_combined_aug_norm",
        "DLv3+ OS16 no-aug": "results/deeplabv3plus_resnet50_os16_combined_norm",
        "SAM2-L fine-tuned": "results/sam_large_combined_aug_norm",
        "SAM2-L frozen": "results/sam_large_combined_frozen_aug_norm",
        "SAM2-L frozen CE": "results/sam_large_ce_frozen_aug_norm",
    }

    metrics = {}
    for name, path in experiments.items():
        m = load_metrics(path)
        if m is not None:
            metrics[name] = m

    os.makedirs("results", exist_ok=True)

    colors = {
        "U-Net R18": "#2196F3",
        "U-Net R50": "#1565C0",
        "U-Net R18 no-aug": "#64B5F6",
        "U-Net R18 CE": "#90CAF9",
        "DLv3+ OS16": "#4CAF50",
        "DLv3+ OS8": "#2E7D32",
        "DLv3+ OS16 no-aug": "#81C784",
        "SAM2-L fine-tuned": "#9C27B0",
        "SAM2-L frozen": "#BA68C8",
        "SAM2-L frozen CE": "#CE93D8",
    }

    # 1. Overall comparison
    fig, axes = plt.subplots(1, 4, figsize=(22, 7))
    names = list(metrics.keys())
    metric_keys = [
        ("pixel_accuracy", "Pixel Accuracy"),
        ("mean_iou", "Mean IoU"),
        ("mean_dice", "Mean Dice"),
        ("mean_hd95", "Mean HD95"),
    ]
    for ax, (key, title) in zip(axes, metric_keys):
        vals = [metrics[n].get(key, 0) or 0 for n in names]
        cols = [colors.get(n, "#888") for n in names]
        bars = ax.bar(range(len(names)), vals, color=cols)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
        ax.set_title(title, fontsize=12)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    "{:.3f}".format(val), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig("results/comparison_overall.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Per-class IoU
    compare_names = ["U-Net R18", "U-Net R50", "DLv3+ OS8",
                     "SAM2-L frozen", "SAM2-L fine-tuned"]
    compare_names = [n for n in compare_names if n in metrics]
    compare_colors = [colors[n] for n in compare_names]

    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(VOC_CLASSES))
    width = 0.8 / len(compare_names)
    offsets = np.arange(len(compare_names)) - (len(compare_names) - 1) / 2

    for i, name in enumerate(compare_names):
        iou_dict = metrics[name]["per_class_iou"]
        vals = [iou_dict.get(cls, 0) for cls in VOC_CLASSES]
        ax.bar(x + offsets[i] * width, vals, width, label=name,
               color=compare_colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(VOC_CLASSES, rotation=45, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title("Per-Class IoU Comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/comparison_per_class_iou.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Ablation A: Backbone (R18 vs R50)
    fig, ax = plt.subplots(figsize=(8, 5))
    data = {"U-Net R18": metrics.get("U-Net R18", {}),
            "U-Net R50": metrics.get("U-Net R50", {})}
    cols_a = ["#2196F3", "#1565C0"]
    for i, (name, m) in enumerate(data.items()):
        vals = [m.get("pixel_accuracy", 0), m.get("mean_iou", 0), m.get("mean_dice", 0)]
        bars = ax.bar(np.arange(3) + i * 0.35, vals, 0.35, label=name, color=cols_a[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    "{:.3f}".format(val), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(3) + 0.175)
    ax.set_xticklabels(["Pixel Acc", "mIoU", "mDice"])
    ax.set_title("Ablation: Backbone (U-Net)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/ablation_backbone.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Ablation B: Loss
    fig, ax = plt.subplots(figsize=(10, 5))
    data = {"U-Net R18 combined": metrics.get("U-Net R18", {}),
            "U-Net R18 CE": metrics.get("U-Net R18 CE", {}),
            "SAM2-L frozen combined": metrics.get("SAM2-L frozen", {}),
            "SAM2-L frozen CE": metrics.get("SAM2-L frozen CE", {})}
    cols_b = ["#2196F3", "#90CAF9", "#9C27B0", "#CE93D8"]
    for i, (name, m) in enumerate(data.items()):
        vals = [m.get("pixel_accuracy", 0), m.get("mean_iou", 0), m.get("mean_dice", 0)]
        bars = ax.bar(np.arange(3) + i * 0.2, vals, 0.2, label=name, color=cols_b[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    "{:.3f}".format(val), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(3) + 0.3)
    ax.set_xticklabels(["Pixel Acc", "mIoU", "mDice"])
    ax.set_title("Ablation: Loss Function (Combined vs CE)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/ablation_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Ablation C: Augmentation
    fig, ax = plt.subplots(figsize=(10, 5))
    data = {
        "U-Net R18 +aug": metrics.get("U-Net R18", {}),
        "U-Net R18 -aug": metrics.get("U-Net R18 no-aug", {}),
        "DLv3+ OS16 +aug": metrics.get("DLv3+ OS16", {}),
        "DLv3+ OS16 -aug": metrics.get("DLv3+ OS16 no-aug", {}),
    }
    cols_c = ["#2196F3", "#64B5F6", "#4CAF50", "#81C784"]
    for i, (name, m) in enumerate(data.items()):
        vals = [m.get("pixel_accuracy", 0), m.get("mean_iou", 0), m.get("mean_dice", 0)]
        bars = ax.bar(np.arange(3) + i * 0.2, vals, 0.2, label=name, color=cols_c[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    "{:.3f}".format(val), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(3) + 0.3)
    ax.set_xticklabels(["Pixel Acc", "mIoU", "mDice"])
    ax.set_title("Ablation: Data Augmentation")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/ablation_augmentation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Ablation D: Output Stride
    fig, ax = plt.subplots(figsize=(8, 5))
    data = {"DLv3+ OS16": metrics.get("DLv3+ OS16", {}),
            "DLv3+ OS8": metrics.get("DLv3+ OS8", {})}
    cols_d = ["#4CAF50", "#2E7D32"]
    for i, (name, m) in enumerate(data.items()):
        vals = [m.get("pixel_accuracy", 0), m.get("mean_iou", 0), m.get("mean_dice", 0)]
        bars = ax.bar(np.arange(3) + i * 0.35, vals, 0.35, label=name, color=cols_d[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    "{:.3f}".format(val), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(3) + 0.175)
    ax.set_xticklabels(["Pixel Acc", "mIoU", "mDice"])
    ax.set_title("Ablation: Output Stride (DeepLabV3+)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/ablation_output_stride.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Ablation E: SAM2 frozen vs fine-tuned
    fig, ax = plt.subplots(figsize=(10, 5))
    data = {"SAM2-L fine-tuned": metrics.get("SAM2-L fine-tuned", {}),
            "SAM2-L frozen": metrics.get("SAM2-L frozen", {}),
            "SAM2-L frozen CE": metrics.get("SAM2-L frozen CE", {})}
    cols_e = ["#9C27B0", "#BA68C8", "#CE93D8"]
    for i, (name, m) in enumerate(data.items()):
        vals = [m.get("pixel_accuracy", 0), m.get("mean_iou", 0), m.get("mean_dice", 0)]
        bars = ax.bar(np.arange(3) + i * 0.25, vals, 0.25, label=name, color=cols_e[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    "{:.3f}".format(val), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(3) + 0.25)
    ax.set_xticklabels(["Pixel Acc", "mIoU", "mDice"])
    ax.set_title("Ablation: SAM2-Large Encoder Regime")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/ablation_sam_encoder.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary
    print("\n" + "=" * 90)
    print("10-EXPERIMENT SUMMARY (512x512)")
    print("=" * 90)
    print("{:<25} {:>8} {:>8} {:>8} {:>8}".format("Model", "PixAcc", "mIoU", "mDice", "HD95"))
    print("-" * 90)

    families = [
        ("--- U-Net ---", ["U-Net R18", "U-Net R50", "U-Net R18 no-aug", "U-Net R18 CE"]),
        ("--- DeepLabV3+ ---", ["DLv3+ OS16", "DLv3+ OS8", "DLv3+ OS16 no-aug"]),
        ("--- SAM2-Large ---", ["SAM2-L fine-tuned", "SAM2-L frozen", "SAM2-L frozen CE"]),
    ]

    if metrics:
        best_miou = max(m["mean_iou"] for m in metrics.values())
    else:
        best_miou = 0
    for header, names_list in families:
        print(header)
        for name in names_list:
            if name in metrics:
                m = metrics[name]
                hd = m.get("mean_hd95")
                hd_str = "{:8.2f}".format(hd) if hd else "     N/A"
                bold = " **" if m["mean_iou"] == best_miou else ""
                print("{:<25} {:8.4f} {:8.4f} {:8.4f} {}{}".format(
                    name, m["pixel_accuracy"], m["mean_iou"], m["mean_dice"], hd_str, bold))


if __name__ == "__main__":
    main()

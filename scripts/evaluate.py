import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders, VOC_CLASSES, NUM_CLASSES
from src.metrics import compute_all_metrics
from src.visualize import (
    plot_mosaic, plot_best_worst, plot_per_class_iou,
    plot_training_curves, plot_confusion_matrix, decode_mask,
)
from scripts.train import get_model


def denormalize(img_tensor, mean, std):
    for c in range(3):
        img_tensor[c] = img_tensor[c] * std[c] + mean[c]
    return img_tensor.clamp(0, 1)


def collect_predictions(model, loader, device, normalize=False):
    model.eval()
    all_images, all_masks, all_preds = [], [], []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            for i in range(images.size(0)):
                img = images[i].clone()
                if normalize:
                    img = denormalize(img, mean, std)
                all_images.append(img.permute(1, 2, 0).numpy())
                all_masks.append(masks[i].numpy())
                all_preds.append(preds[i])

    return all_images, all_masks, all_preds


def per_image_iou(preds, targets, num_classes=NUM_CLASSES, ignore_index=255):
    ious = []
    for pred, target in zip(preds, targets):
        valid = target != ignore_index
        intersection = np.zeros(num_classes)
        union = np.zeros(num_classes)
        for c in range(num_classes):
            pred_c = (pred == c) & valid
            gt_c = (target == c) & valid
            intersection[c] = (pred_c & gt_c).sum()
            union[c] = (pred_c | gt_c).sum()
        present = union > 0
        if present.any():
            ious.append(np.mean(intersection[present] / (union[present] + 1e-10)))
        else:
            ious.append(0.0)
    return np.array(ious)


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model, backbone=args.backbone, pretrained=True,
                      device=device, sam_size=args.sam_size,
                      freeze_encoder=args.freeze_encoder,
                      output_stride=args.output_stride)
    ckpt_path = os.path.join(args.checkpoint_dir, "best.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    print(f"Loaded: {ckpt_path}")

    _, val_loader = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size,
        normalize=args.normalize, img_size=args.img_size,
        num_workers=args.num_workers,
    )

    images, masks, preds = collect_predictions(model, val_loader, device,
                                                normalize=args.normalize)
    print(f"Evaluated {len(images)} images")

    metrics = compute_all_metrics(preds, masks)
    exp_name = os.path.basename(args.checkpoint_dir)
    out_dir = os.path.join("results", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Results: {exp_name}")
    print(f"{'='*50}")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  Mean IoU:       {metrics['mean_iou']:.4f}")
    print(f"  Mean Dice:      {metrics['mean_dice']:.4f}")
    print(f"  Mean HD95:      {metrics['mean_hd95']:.2f}")
    print(f"\n  Per-Class IoU:")
    for i, cls in enumerate(VOC_CLASSES):
        print(f"    {cls:15s}: IoU={metrics['per_class_iou'][i]:.4f}  "
              f"Dice={metrics['per_class_dice'][i]:.4f}  "
              f"Acc={metrics['per_class_accuracy'][i]:.4f}")

    save_metrics = {
        "pixel_accuracy": float(metrics["pixel_accuracy"]),
        "mean_iou": float(metrics["mean_iou"]),
        "mean_dice": float(metrics["mean_dice"]),
        "mean_hd95": float(metrics["mean_hd95"]) if not np.isnan(metrics["mean_hd95"]) else None,
        "per_class_iou": {VOC_CLASSES[i]: float(v) for i, v in enumerate(metrics["per_class_iou"])},
        "per_class_dice": {VOC_CLASSES[i]: float(v) for i, v in enumerate(metrics["per_class_dice"])},
        "per_class_accuracy": {VOC_CLASSES[i]: float(v) for i, v in enumerate(metrics["per_class_accuracy"])},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(save_metrics, f, indent=2)

    image_ious = per_image_iou(preds, masks)

    plot_mosaic(images, masks, preds, n=4,
                save_path=os.path.join(out_dir, "mosaic.png"))
    plt.close("all")

    plot_best_worst(images, masks, preds, image_ious, n=3,
                    save_path=os.path.join(out_dir, "best_worst.png"))
    plt.close("all")

    plot_per_class_iou(metrics["per_class_iou"], model_name=exp_name,
                       save_path=os.path.join(out_dir, "per_class_iou.png"))
    plt.close("all")

    plot_confusion_matrix(metrics["confusion_matrix"], model_name=exp_name,
                          save_path=os.path.join(out_dir, "confusion_matrix.png"))
    plt.close("all")

    person_idx = VOC_CLASSES.index("person")
    person_ious = []
    min_person_ratio = 0.05
    total_pixels = masks[0].shape[0] * masks[0].shape[1]
    min_person_pixels = int(total_pixels * min_person_ratio)
    for pred, mask in zip(preds, masks):
        valid = mask != 255
        gt_person = (mask == person_idx) & valid
        if gt_person.sum() > min_person_pixels:
            pred_person = (pred == person_idx) & valid
            inter = (gt_person & pred_person).sum()
            union = (gt_person | pred_person).sum()
            person_ious.append(inter / (union + 1e-10))
        else:
            person_ious.append(-1)

    person_ious = np.array(person_ious)
    has_person = person_ious >= 0
    if has_person.sum() >= 6:
        person_images = [images[i] for i in range(len(images)) if has_person[i]]
        person_masks = [masks[i] for i in range(len(masks)) if has_person[i]]
        person_preds = [preds[i] for i in range(len(preds)) if has_person[i]]
        person_ious_valid = person_ious[has_person]

        plot_best_worst(person_images, person_masks, person_preds,
                        person_ious_valid, n=3,
                        save_path=os.path.join(out_dir, "person_best_worst.png"))
        plt.close("all")

    history_path = os.path.join(args.checkpoint_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(
            history["train_loss"], history["val_loss"], history["val_miou"],
            save_path=os.path.join(out_dir, "training_curves.png"),
        )
        plt.close("all")

    print(f"\nVisualizations saved to: {out_dir}/")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "deeplabv3", "deeplabv3plus", "sam"])
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./dataset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sam-size", type=str, default="tiny",
                        choices=["tiny", "large"])
    parser.add_argument("--freeze-encoder", action="store_true", default=False)
    parser.add_argument("--output-stride", type=int, default=16,
                        choices=[8, 16])
    args = parser.parse_args()
    run_evaluation(args)

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from src.dataset import get_dataloaders, NUM_CLASSES
from src.metrics import compute_confusion_matrix, pixel_accuracy, per_class_iou, per_class_dice
from src.losses import DiceLoss, CombinedLoss


def get_model(model_name, backbone="resnet18", pretrained=True, device="cuda",
              sam_size="tiny", freeze_encoder=True, output_stride=16):
    if model_name == "unet":
        from models.unet import UNet
        return UNet(backbone=backbone, num_classes=NUM_CLASSES, pretrained=pretrained).to(device)
    elif model_name == "deeplabv3":
        from models.deeplabv3 import DeepLabV3
        return DeepLabV3(num_classes=NUM_CLASSES, pretrained_backbone=pretrained).to(device)
    elif model_name == "deeplabv3plus":
        from models.deeplabv3plus import DeepLabV3Plus
        encoder_weights = "imagenet" if pretrained else None
        return DeepLabV3Plus.build(
            num_classes=NUM_CLASSES, encoder_name=backbone,
            encoder_weights=encoder_weights, output_stride=output_stride,
        ).to(device)
    elif model_name == "sam":
        from models.sam_seg import build_sam_seg
        return build_sam_seg(
            size=sam_size, num_classes=NUM_CLASSES,
            freeze_encoder=freeze_encoder, device=device,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_loss_fn(loss_name, num_classes=NUM_CLASSES, ignore_index=255):
    if loss_name == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_name == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    elif loss_name == "combined":
        return CombinedLoss(num_classes=num_classes, ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=False, scaler=None):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        if use_amp:
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=NUM_CLASSES):
    model.eval()
    total_loss = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        confusion += compute_confusion_matrix(preds, targets, num_classes)

    avg_loss = total_loss / len(loader.dataset)
    iou = per_class_iou(confusion)
    dice = per_class_dice(confusion)
    present = confusion.sum(axis=1) > 0
    miou = np.mean(iou[present])
    mdice = np.mean(dice[present])
    pa = pixel_accuracy(confusion)

    return {
        "loss": avg_loss,
        "miou": miou,
        "mdice": mdice,
        "pixel_acc": pa,
        "per_class_iou": iou.tolist(),
        "per_class_dice": dice.tolist(),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size,
        augment=args.augment, normalize=args.normalize,
        img_size=args.img_size, num_workers=args.num_workers,
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    model = get_model(args.model, backbone=args.backbone,
                      pretrained=args.pretrained, device=device,
                      sam_size=args.sam_size, freeze_encoder=args.freeze_encoder,
                      output_stride=args.output_stride)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({args.backbone}), trainable params: {param_count:,}")

    criterion = get_loss_fn(args.loss)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = args.model == "sam" and not args.freeze_encoder
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using AMP (mixed precision)")

    best_miou = 0
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_mdice": [], "val_pa": []}

    exp_name = f"{args.model}_{args.backbone}_{args.loss}"
    if args.model == "sam":
        exp_name = f"sam_{args.sam_size}_{args.loss}"
        if args.freeze_encoder:
            exp_name += "_frozen"
    if args.model == "deeplabv3plus":
        exp_name = f"deeplabv3plus_{args.backbone}_os{args.output_stride}_{args.loss}"
    if args.augment:
        exp_name += "_aug"
    if args.normalize:
        exp_name += "_norm"
    if not args.pretrained:
        exp_name += "_scratch"
    save_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nExperiment: {exp_name}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'mIoU':>7} | {'mDice':>7} | {'PixAcc':>7} | {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     use_amp=use_amp, scaler=scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_miou"].append(val_metrics["miou"])
        history["val_mdice"].append(val_metrics["mdice"])
        history["val_pa"].append(val_metrics["pixel_acc"])

        is_best = val_metrics["miou"] > best_miou
        if is_best:
            best_miou = val_metrics["miou"]
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_metrics['loss']:10.4f} | "
              f"{val_metrics['miou']:7.4f} | {val_metrics['mdice']:7.4f} | "
              f"{val_metrics['pixel_acc']:7.4f} | {elapsed:5.1f}s"
              + (" *" if is_best else ""))

    torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(save_dir, "final_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)

    print(f"\nBest mIoU: {best_miou:.4f}")
    print(f"Saved to: {save_dir}/")
    return history, val_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "deeplabv3", "deeplabv3plus", "sam"])
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "dice", "combined"])
    parser.add_argument("--data-root", type=str, default="./dataset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sam-size", type=str, default="tiny",
                        choices=["tiny", "large"])
    parser.add_argument("--freeze-encoder", action="store_true", default=False)
    parser.add_argument("--output-stride", type=int, default=16,
                        choices=[8, 16])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.dataset import VOC_CLASSES, VOC_COLORMAP


def get_voc_cmap():
    cmap_arr = np.array(VOC_COLORMAP) / 255.0
    return ListedColormap(cmap_arr)


def decode_mask(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in enumerate(VOC_COLORMAP):
        rgb[mask == c] = color
    return rgb


def show_sample(image, mask_gt, mask_pred=None, title=None):
    ncols = 3 if mask_pred is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(decode_mask(mask_gt))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    if mask_pred is not None:
        axes[2].imshow(decode_mask(mask_pred))
        axes[2].set_title("Prediction")
        axes[2].axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_mosaic(images, masks_gt, masks_pred, n=4, save_path=None):
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    for i in range(min(n, len(images))):
        img = images[i]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(decode_mask(masks_gt[i]))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(decode_mask(masks_pred[i]))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_best_worst(images, masks_gt, masks_pred, ious, n=3, save_path=None):
    sorted_idx = np.argsort(ious)
    worst_idx = sorted_idx[:n]
    best_idx = sorted_idx[-n:][::-1]

    fig, axes = plt.subplots(2 * n, 3, figsize=(15, 5 * 2 * n))

    for row, idx in enumerate(best_idx):
        img = images[idx]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Best #{row+1} (IoU={ious[idx]:.3f})")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(decode_mask(masks_gt[idx]))
        axes[row, 1].set_title("GT")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(decode_mask(masks_pred[idx]))
        axes[row, 2].set_title("Pred")
        axes[row, 2].axis("off")

    for row_off, idx in enumerate(worst_idx):
        row = n + row_off
        img = images[idx]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Worst #{row_off+1} (IoU={ious[idx]:.3f})")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(decode_mask(masks_gt[idx]))
        axes[row, 1].set_title("GT")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(decode_mask(masks_pred[idx]))
        axes[row, 2].set_title("Pred")
        axes[row, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_class_iou(iou_dict, model_name="Model", save_path=None):
    classes = VOC_CLASSES
    ious = [iou_dict[i] if i in iou_dict else iou_dict[i] for i in range(len(classes))]
    if isinstance(ious[0], (np.floating, float)):
        pass
    else:
        ious = list(iou_dict)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(classes))
    ax.bar(x, ious, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title(f"Per-Class IoU - {model_name}")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(confusion, class_names=None, normalize=True,
                          model_name="Model", save_path=None):
    if class_names is None:
        class_names = VOC_CLASSES

    if normalize:
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = confusion.astype(np.float64) / row_sums
    else:
        cm = confusion.astype(np.float64)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0,
                   vmax=1 if normalize else None)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(train_losses, val_losses, val_mious, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    ax2.plot(epochs, val_mious, label="Val mIoU", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.set_title("Validation mIoU")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

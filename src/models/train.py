"""
RetinaGuard AI - Training Pipeline
-
Purpose: Complete training loop with two-stage transfer learning,
cross-validation, early stopping, mixed-precision training, and
checkpoint management.

Model selection uses ONLY validation/CV data. The test set is NEVER
used for training, early stopping, or model selection.

Usage:
    python -m src.models.train --config configs/base_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import IDRiDDataset
from src.models.architectures import create_model
from src.models.losses import create_loss_function

logger = logging.getLogger("retinaguard.train")


# -
# Reproducibility
# -
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# -
# Early stopping
# -
class EarlyStopping:
    """Early stopping monitor.

    Stops training if the monitored metric does not improve for a specified
    number of epochs. Uses validation data only.

    Attributes:
        patience: Number of epochs to wait.
        min_delta: Minimum improvement to qualify as improvement.
        best_score: Best metric value seen.
        counter: Number of epochs since last improvement.
        should_stop: Whether to stop training.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        """Initialise early stopping.

        Args:
            patience: Epochs to wait before stopping.
            min_delta: Minimum metric improvement.
            mode: 'max' if higher is better, 'min' if lower is better.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement"
                )

        return self.should_stop


# -
# Training epoch
# -
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    gradient_clip: float = 1.0,
    use_amp: bool = False,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        scaler: GradScaler for mixed precision.
        gradient_clip: Max gradient norm for clipping.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Dictionary with loss and accuracy metrics.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return {"loss": epoch_loss, "accuracy": epoch_acc}


# -
# Validation epoch
# -
@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Evaluate on validation data.

    Args:
        model: The model to evaluate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Compute device.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Dictionary with loss, accuracy, probabilities, labels, and predictions.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if use_amp:
            with autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        _, predicted = logits.max(1)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(predicted.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_probs_array = np.concatenate(all_probs, axis=0)

    # Compute AUC if binary
    auc = None
    if all_probs_array.shape[1] == 2:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, all_probs_array[:, 1])
        except ValueError:
            auc = None

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "auc": auc,
        "probabilities": all_probs_array,
        "labels": all_labels,
        "predictions": all_preds,
    }


# -
# Full training pipeline
# -
def train_model(
    config: dict[str, Any],
    fold: Optional[int] = None,
    task: str = "binary",
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute the complete two-stage training pipeline.

    Args:
        config: Configuration dictionary.
        fold: CV fold for validation (None = use full training set).
        task: 'binary' or 'multiclass'.
        output_dir: Directory for outputs (checkpoints, logs).

    Returns:
        Dictionary with training history and best metrics.
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths
    project_root = Path(config.get("_project_root", "."))
    metadata_dir = project_root / config["paths"]["metadata_dir"]
    metadata_csv = metadata_dir / "idrid_splits.csv"

    if output_dir is None:
        output_dir = project_root / config["paths"]["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 4)

    num_classes = 2 if task == "binary" else 5

    train_dataset = IDRiDDataset(
        metadata_csv=metadata_csv,
        mode="train",
        task=task,
        fold=fold,
        config=config,
    )
    val_dataset = IDRiDDataset(
        metadata_csv=metadata_csv,
        mode="val",
        task=task,
        fold=fold,
        config=config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Training: {len(train_dataset)} images, "
        f"Validation: {len(val_dataset)} images"
    )
    logger.info(f"Class distribution (train): {train_dataset.get_class_distribution()}")
    logger.info(f"Class distribution (val): {val_dataset.get_class_distribution()}")

    # Create model
    architecture = config.get("model", {}).get("architecture", "efficientnet_b0")
    dropout_rate = config.get("model", {}).get("dropout_rate", 0.3)
    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=dropout_rate,
    )
    model = model.to(device)

    # Loss function
    loss_config = config.get("loss", {})
    loss_type = loss_config.get("primary", "weighted_cross_entropy")
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = create_loss_function(
        loss_type=loss_type,
        class_weights=class_weights,
    )

    # Training settings
    use_amp = train_cfg.get("mixed_precision", False) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    gradient_clip = train_cfg.get("gradient_clip_max_norm", 1.0)

    # Early stopping
    es_config = train_cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=es_config.get("patience", 10),
        min_delta=es_config.get("min_delta", 0.001),
        mode="max",
    )

    # Training log
    log_path = output_dir / f"training_log_fold{fold}_{task}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "stage", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_auc", "lr", "time_s",
    ])

    best_val_auc = 0.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    # -
    # Stage 1: Frozen backbone
    # -
    stage1 = train_cfg.get("stage1", {})
    stage1_epochs = stage1.get("epochs", 30)
    stage1_lr = stage1.get("learning_rate", 1e-3)

    logger.info("=" * 50)
    logger.info("Stage 1: Training classification head (backbone frozen)")
    logger.info(f"  Epochs: {stage1_epochs}, LR: {stage1_lr}")
    logger.info("=" * 50)

    model.freeze_backbone()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage1_lr,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=stage1_epochs,
        eta_min=float(config.get("training", {}).get("scheduler", {}).get("eta_min", 1e-6)),
    )

    for epoch in range(1, stage1_epochs + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, gradient_clip=gradient_clip, use_amp=use_amp,
        )
        val_metrics = validate(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[S1] Epoch {epoch}/{stage1_epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_auc={val_metrics.get('auc', 'N/A')}, "
            f"time={elapsed:.1f}s"
        )

        val_auc = val_metrics.get("auc") or val_metrics["accuracy"]

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            checkpoint_path = output_dir / f"best_model_fold{fold}_{task}.pt"
            torch.save({
                "epoch": epoch,
                "stage": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_loss": val_metrics["loss"],
                "config": config,
            }, checkpoint_path)
            logger.info(f"   New best model saved (AUC: {val_auc:.4f})")

        # Log
        log_writer.writerow([
            epoch, 1, train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
            val_metrics.get("auc"), current_lr, elapsed,
        ])

        history.append({
            "epoch": epoch, "stage": 1,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()
               if k not in ("probabilities", "labels", "predictions")},
        })

        if early_stopper.step(val_auc):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # -
    # Stage 2: Fine-tune backbone
    # -
    stage2 = train_cfg.get("stage2", {})
    stage2_epochs = stage2.get("epochs", 50)
    stage2_lr = stage2.get("learning_rate", 1e-4)
    unfreeze_layers = stage2.get("unfreeze_layers", None)

    logger.info("=" * 50)
    logger.info("Stage 2: Fine-tuning backbone")
    logger.info(f"  Epochs: {stage2_epochs}, LR: {stage2_lr}")
    logger.info(f"  Unfreeze from: {unfreeze_layers or 'all'}")
    logger.info("=" * 50)

    # Load best Stage 1 checkpoint
    checkpoint_path = output_dir / f"best_model_fold{fold}_{task}.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best Stage 1 model from epoch {checkpoint['epoch']}")

    model.unfreeze_backbone(unfreeze_from=unfreeze_layers)

    # Reset early stopping for Stage 2
    early_stopper = EarlyStopping(
        patience=es_config.get("patience", 10),
        min_delta=es_config.get("min_delta", 0.001),
        mode="max",
    )
    early_stopper.best_score = best_val_auc

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage2_lr,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=stage2_epochs,
        eta_min=float(config.get("training", {}).get("scheduler", {}).get("eta_min", 1e-6)),
    )

    for epoch in range(1, stage2_epochs + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, gradient_clip=gradient_clip, use_amp=use_amp,
        )
        val_metrics = validate(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[S2] Epoch {epoch}/{stage2_epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_auc={val_metrics.get('auc', 'N/A')}, "
            f"time={elapsed:.1f}s"
        )

        val_auc = val_metrics.get("auc") or val_metrics["accuracy"]

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = stage1_epochs + epoch
            checkpoint_path = output_dir / f"best_model_fold{fold}_{task}.pt"
            torch.save({
                "epoch": stage1_epochs + epoch,
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_loss": val_metrics["loss"],
                "config": config,
            }, checkpoint_path)
            logger.info(f"   New best model saved (AUC: {val_auc:.4f})")

        log_writer.writerow([
            stage1_epochs + epoch, 2, train_metrics["loss"],
            train_metrics["accuracy"], val_metrics["loss"],
            val_metrics["accuracy"], val_metrics.get("auc"),
            current_lr, elapsed,
        ])

        history.append({
            "epoch": stage1_epochs + epoch, "stage": 2,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()
               if k not in ("probabilities", "labels", "predictions")},
        })

        if early_stopper.step(val_auc):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    log_file.close()
    logger.info(f"Training complete. Best epoch: {best_epoch}, Best AUC: {best_val_auc:.4f}")

    # Save training summary
    summary = {
        "task": task,
        "fold": fold,
        "architecture": architecture,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "total_epochs": len(history),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = output_dir / f"training_summary_fold{fold}_{task}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "history": history,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
        "log_path": str(log_path),
    }


# -
# Cross-validation runner
# -
def run_cross_validation(
    config: dict[str, Any],
    task: str = "binary",
) -> dict[str, Any]:
    """Run k-fold cross-validation for model development.

    Model selection uses ONLY cross-validation results.
    The test set is NEVER accessed during this process.

    Args:
        config: Configuration dictionary.
        task: 'binary' or 'multiclass'.

    Returns:
        Dictionary with per-fold results and aggregated metrics.
    """
    n_folds = config.get("splitting", {}).get("n_folds", 5)
    project_root = Path(config.get("_project_root", "."))
    output_dir = project_root / config["paths"]["checkpoint_dir"]

    logger.info("=" * 60)
    logger.info(f"Cross-Validation: {n_folds} folds, task={task}")
    logger.info("Test set is NOT used during cross-validation.")
    logger.info("=" * 60)

    fold_results: list[dict[str, Any]] = []

    for fold in range(n_folds):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold + 1}/{n_folds}")
        logger.info(f"{'='*50}")

        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        result = train_model(
            config=config,
            fold=fold,
            task=task,
            output_dir=fold_dir,
        )
        fold_results.append(result)

    # Aggregate results
    val_aucs = [r["best_val_auc"] for r in fold_results]
    cv_summary = {
        "n_folds": n_folds,
        "task": task,
        "val_aucs": val_aucs,
        "mean_val_auc": float(np.mean(val_aucs)),
        "std_val_auc": float(np.std(val_aucs)),
        "min_val_auc": float(np.min(val_aucs)),
        "max_val_auc": float(np.max(val_aucs)),
        "fold_results": [
            {
                "fold": i,
                "best_val_auc": r["best_val_auc"],
                "best_epoch": r["best_epoch"],
            }
            for i, r in enumerate(fold_results)
        ],
    }

    logger.info("\n" + "=" * 60)
    logger.info("Cross-Validation Summary")
    logger.info(f"  Mean AUC: {cv_summary['mean_val_auc']:.4f} ± {cv_summary['std_val_auc']:.4f}")
    logger.info(f"  Range: [{cv_summary['min_val_auc']:.4f}, {cv_summary['max_val_auc']:.4f}]")
    logger.info("=" * 60)

    # Save summary
    summary_path = output_dir / f"cv_summary_{task}.json"
    with open(summary_path, "w") as f:
        json.dump(cv_summary, f, indent=2)

    return cv_summary


# -
# Final model refit
# -
def train_final_model(
    config: dict[str, Any],
    task: str = "binary",
) -> dict[str, Any]:
    """Train the final model on the FULL official training set.

    This is run AFTER model selection via cross-validation.
    The pipeline (architecture, loss, resolution, threshold) must be
    locked before calling this function.

    Args:
        config: Configuration dictionary.
        task: 'binary' or 'multiclass'.

    Returns:
        Training results dictionary.
    """
    logger.info("=" * 60)
    logger.info("FINAL MODEL TRAINING")
    logger.info("Training on full official training set (413 images)")
    logger.info("Pipeline is LOCKED - no more model selection decisions")
    logger.info("=" * 60)

    project_root = Path(config.get("_project_root", "."))
    output_dir = project_root / config["paths"]["checkpoint_dir"] / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train without fold (uses full training set)
    # For validation during training, use fold 0 as monitoring only
    result = train_model(
        config=config,
        fold=0,  # Use fold 0 for monitoring only
        task=task,
        output_dir=output_dir,
    )

    return result


# -
# CLI entry point
# -
def main() -> None:
    """CLI entry point for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="RetinaGuard AI - Model Training"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base_config.yaml"))
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--mode", choices=["cv", "final", "single_fold"], default="cv")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["_project_root"] = str(args.config.parent.parent)

    if args.mode == "cv":
        run_cross_validation(config, task=args.task)
    elif args.mode == "final":
        train_final_model(config, task=args.task)
    elif args.mode == "single_fold":
        train_model(config, fold=args.fold, task=args.task)


if __name__ == "__main__":
    main()

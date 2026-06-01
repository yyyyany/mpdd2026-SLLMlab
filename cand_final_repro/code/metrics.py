from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def coral_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """序数标签编码：levels[i,k] = 1 if y_i > k else 0，形状 [B, K-1]。"""
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def coral_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """CORAL 损失：K-1 路 BCEWithLogits（P(y>k)）。"""
    levels = coral_levels(labels, num_classes)
    return F.binary_cross_entropy_with_logits(logits, levels)


def coral_decode(logits: torch.Tensor) -> torch.Tensor:
    """CORAL 解码：预测类别 = Σ_k [sigmoid(logit_k) > 0.5]。"""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    denom = y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2
    return safe_float(2 * cov / denom) if denom > 1e-10 else 0.0


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = {
        "acc": safe_float(accuracy_score(y_true, y_pred)),
        "f1": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": safe_float(cohen_kappa_score(y_true, y_pred)),
        "ccc": ccc(y_true, y_pred),
        "rmse": safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": safe_float(mean_absolute_error(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    metrics["selection_score"] = metrics["f1"]
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = {
        "ccc": ccc(y_true, y_pred),
        "rmse": safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": safe_float(mean_absolute_error(y_true, y_pred)),
        "r2": safe_float(r2_score(y_true, y_pred)),
    }
    metrics["selection_score"] = metrics["ccc"]
    return metrics


def denormalize_phq(phq_log: np.ndarray) -> np.ndarray:
    values = np.asarray(phq_log, dtype=np.float64)
    return np.clip(np.expm1(values), 0.0, 27.0)


def compute_class_proportions(labels: np.ndarray, num_classes: int) -> list[float]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return [1.0 / num_classes] * num_classes
    return (counts / total).tolist()


def apply_thresholds(phq_raw: np.ndarray, thresholds: list[float], num_classes: int) -> np.ndarray:
    phq_raw = np.asarray(phq_raw, dtype=np.float64)
    if num_classes == 2:
        if not thresholds:
            raise ValueError("binary task requires one threshold")
        return (phq_raw >= thresholds[0]).astype(np.int64)
    if num_classes == 3:
        if len(thresholds) < 2:
            raise ValueError("ternary task requires two thresholds")
        t1, t2 = sorted(thresholds[:2])
        preds = np.zeros(phq_raw.shape[0], dtype=np.int64)
        preds[phq_raw >= t1] = 1
        preds[phq_raw >= t2] = 2
        return preds
    raise ValueError(f"Unsupported num_classes={num_classes}")


def calibrate_threshold_dist_align(phq_raw: np.ndarray, class_proportions: list[float]) -> list[float]:
    phq_raw = np.asarray(phq_raw, dtype=np.float64)
    if phq_raw.size == 0:
        raise ValueError("phq_raw must not be empty")
    sorted_vals = np.sort(phq_raw)
    n = sorted_vals.size

    if len(class_proportions) == 2:
        pct = max(0.0, min(1.0, class_proportions[0]))
        idx = int(round(pct * (n - 1)))
        return [float(sorted_vals[idx])]

    if len(class_proportions) == 3:
        pct1 = max(0.0, min(1.0, class_proportions[0]))
        pct2 = max(0.0, min(1.0, class_proportions[0] + class_proportions[1]))
        idx1 = int(round(pct1 * (n - 1)))
        idx2 = int(round(pct2 * (n - 1)))
        t1 = float(sorted_vals[idx1])
        t2 = float(sorted_vals[max(idx1, idx2)])
        return [t1, t2]

    raise ValueError(f"Unsupported class_proportions length={len(class_proportions)}")


def calibrate_threshold_f1(
    phq_raw: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    grid_size: int = 50,
) -> tuple[list[float], float]:
    phq_raw = np.asarray(phq_raw, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if phq_raw.size == 0:
        raise ValueError("phq_raw must not be empty")

    if num_classes == 2:
        candidates = np.unique(phq_raw)
        if candidates.size > grid_size:
            quantiles = np.linspace(0.0, 1.0, grid_size)
            candidates = np.quantile(phq_raw, quantiles)
        best_threshold = float(candidates[0])
        best_f1 = -1.0
        for threshold in candidates:
            preds = apply_thresholds(phq_raw, [float(threshold)], num_classes=2)
            score = safe_float(f1_score(labels, preds, average="macro", zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        return [best_threshold], best_f1

    if num_classes == 3:
        candidates = np.unique(phq_raw)
        if candidates.size > grid_size:
            quantiles = np.linspace(0.0, 1.0, grid_size)
            candidates = np.quantile(phq_raw, quantiles)
        best_thresholds = [float(candidates[0]), float(candidates[-1])]
        best_f1 = -1.0
        for idx1 in range(candidates.size):
            for idx2 in range(idx1 + 1, candidates.size):
                thresholds = [float(candidates[idx1]), float(candidates[idx2])]
                preds = apply_thresholds(phq_raw, thresholds, num_classes=3)
                score = safe_float(f1_score(labels, preds, average="macro", zero_division=0))
                if score > best_f1:
                    best_f1 = score
                    best_thresholds = thresholds
        return best_thresholds, best_f1

    raise ValueError(f"Unsupported num_classes={num_classes}")


def joint_regression_metrics(
    class_true: np.ndarray,
    class_pred: np.ndarray,
    phq_true: np.ndarray,
    phq_pred: np.ndarray,
) -> dict[str, Any]:
    metrics = classification_metrics(class_true, class_pred)
    reg_metrics = regression_metrics(phq_true, phq_pred)
    metrics["ccc"] = reg_metrics["ccc"]
    metrics["rmse"] = reg_metrics["rmse"]
    metrics["mae"] = reg_metrics["mae"]
    metrics["r2"] = reg_metrics["r2"]

    phq_true_raw = denormalize_phq(phq_true)
    phq_pred_raw = denormalize_phq(phq_pred)
    raw_reg_metrics = regression_metrics(phq_true_raw, phq_pred_raw)
    metrics["raw_ccc"] = raw_reg_metrics["ccc"]
    metrics["raw_rmse"] = raw_reg_metrics["rmse"]
    metrics["raw_mae"] = raw_reg_metrics["mae"]
    metrics["raw_r2"] = raw_reg_metrics["r2"]
    metrics["selection_score"] = metrics["f1"]
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Any,
    device: torch.device,
    task: str,
    head_type: str = "softmax",
) -> dict[str, Any]:
    is_joint_regression = isinstance(criterion, (tuple, list))
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []
    all_ids: list[int] = []
    all_phq_preds: list[float] = []
    all_phq_labels: list[float] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            outputs = model(
                audio=batch["audio"].to(device) if "audio" in batch else None,
                video=batch["video"].to(device) if "video" in batch else None,
                gait=batch["gait"].to(device) if "gait" in batch else None,
                personality=batch["personality"].to(device),
                pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
            )
            if is_joint_regression:
                criterion_cls, criterion_reg = criterion
                phq9 = batch["phq9"].to(device)
                logits, reg_out = outputs
                if head_type == "coral":
                    num_classes = logits.shape[1] + 1
                    loss_cls = coral_loss(logits, labels, num_classes)
                    batch_preds = coral_decode(logits).cpu().numpy().tolist()
                else:
                    loss_cls = criterion_cls(logits, labels)
                    batch_preds = logits.argmax(dim=-1).cpu().numpy().tolist()
                loss_reg = criterion_reg(reg_out, phq9)
                loss = loss_cls + loss_reg
                batch_labels = labels.cpu().numpy().tolist()
                batch_phq_preds = reg_out.cpu().numpy().tolist()
                batch_phq_labels = phq9.cpu().numpy().tolist()
                total_cls_loss += float(loss_cls.item()) * len(batch_labels)
                total_reg_loss += float(loss_reg.item()) * len(batch_labels)
            else:
                logits = outputs
                loss = criterion(logits, labels)
                batch_preds = logits.argmax(dim=-1).cpu().numpy().tolist()
                batch_labels = labels.cpu().numpy().tolist()

            total_loss += float(loss.item()) * len(batch_labels)
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            all_ids.extend(batch["pid"].cpu().numpy().tolist())
            if is_joint_regression:
                all_phq_preds.extend(batch_phq_preds)
                all_phq_labels.extend(batch_phq_labels)

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    if is_joint_regression:
        phq_true = np.asarray(all_phq_labels, dtype=np.float64)
        phq_pred = np.asarray(all_phq_preds, dtype=np.float64)
        metrics = joint_regression_metrics(y_true, y_pred, phq_true, phq_pred)
        metrics["cls_loss"] = safe_float(total_cls_loss / max(1, len(all_labels)))
        metrics["reg_loss"] = safe_float(total_reg_loss / max(1, len(all_labels)))
        metrics["class_true"] = y_true.tolist()
        metrics["class_pred"] = y_pred.tolist()
        metrics["phq_true"] = phq_true.tolist()
        metrics["phq_pred"] = phq_pred.tolist()
        metrics["y_true"] = phq_true.tolist()
        metrics["y_pred"] = phq_pred.tolist()
    else:
        metrics = classification_metrics(y_true, y_pred)
        metrics["y_true"] = y_true.tolist()
        metrics["y_pred"] = y_pred.tolist()

    if task == "regression":
        metrics["selection_score"] = safe_float(metrics.get("ccc"))
    else:
        metrics["selection_score"] = safe_float(metrics.get("f1", metrics.get("ccc", 0.0)))

    metrics["loss"] = safe_float(total_loss / max(1, len(all_labels)))
    metrics["ids"] = all_ids
    return metrics

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import REGRESSION_TASK, MPDDElderDataset, collate_batch, load_task_maps, resolve_project_path
from metrics import (
    apply_thresholds,
    calibrate_threshold_dist_align,
    calibrate_threshold_f1,
    compute_class_proportions,
    coral_decode,
    denormalize_phq,
    evaluate_model,
)
from models import TorchcatBaseline
from train_val_split import _load_train_rows


# 仓库根目录（cand_final_repro/code 的上两级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUBTRACK_LOG_DIRS = {
    "A-V+P": "A-V-P",
    "A-V-G+P": "A-V-G+P",
    "G+P": "G-P",
}
METRIC_ARRAY_KEYS = {"ids", "y_true", "y_pred", "class_true", "class_pred", "phq_true", "phq_pred"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(resolve_project_path(config_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained MPDD-AVG baseline checkpoint.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--split_csv", default="")
    parser.add_argument("--personality_npy", default="")
    parser.add_argument("--trainval_split_csv", default="")
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--logs_dir", default=defaults["logs_dir"])
    parser.add_argument("--predict_only", action="store_true")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--threshold_mode", default="dist_align", choices=["dist_align", "f1"])
    parser.add_argument("--class_source", default="reg", choices=["reg", "cls"])
    parser.add_argument("--thresholds", default="")
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default="config.json")
    known_args, _ = base_parser.parse_known_args()
    defaults = load_config(known_args.config)
    parser = build_parser(defaults)
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(f"elder_track1_test_{time.time_ns()}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    return logger


def append_summary_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_track_task_dir(root: Path, track: str, subtrack: str, task: str, experiment_name: str) -> Path:
    subtrack_dir = SUBTRACK_LOG_DIRS.get(subtrack, subtrack.replace("+", "-"))
    return root / track / subtrack_dir / task / experiment_name


def to_project_relative_path(path_like: str | Path) -> str:
    path = resolve_project_path(path_like)
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def require_checkpoint_value(checkpoint: dict[str, Any], key: str) -> Any:
    value = checkpoint.get(key)
    if value in (None, ""):
        raise KeyError(f"Checkpoint missing required field: {key}")
    return value


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in METRIC_ARRAY_KEYS}


def remap_repo_path(path_like: str | Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        return path.as_posix()
    if path.exists():
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)

    for anchor in ("MPDD-AVG2026", "checkpoints", "logs"):
        if anchor not in path.parts:
            continue
        anchor_index = path.parts.index(anchor)
        candidate = PROJECT_ROOT.joinpath(*path.parts[anchor_index:])
        if candidate.exists() or anchor == "logs":
            return candidate.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def parse_thresholds(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def get_num_classes(task: str) -> int:
    if task == "binary":
        return 2
    if task == "ternary":
        return 3
    raise ValueError(f"Unsupported task for predict_only: {task}")


def load_trainval_class_proportions(trainval_split_csv: str, task: str) -> list[float]:
    rows = _load_train_rows(trainval_split_csv)
    label_column = "label2" if task == "binary" else "label3"
    labels = np.asarray([int(float(row[label_column])) for row in rows], dtype=np.int64)
    return compute_class_proportions(labels, num_classes=get_num_classes(task))


def run_model_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_regression_head: bool,
    head_type: str = "softmax",
) -> dict[str, np.ndarray]:
    model.eval()
    all_ids: list[int] = []
    all_cls_preds: list[int] = []
    all_phq_log_preds: list[float] = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                audio=batch["audio"].to(device) if "audio" in batch else None,
                video=batch["video"].to(device) if "video" in batch else None,
                gait=batch["gait"].to(device) if "gait" in batch else None,
                personality=batch["personality"].to(device),
                pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
            )
            if use_regression_head:
                logits, reg_out = outputs
                all_phq_log_preds.extend(reg_out.cpu().numpy().tolist())
            else:
                logits = outputs
            if head_type == "coral":
                all_cls_preds.extend(coral_decode(logits).cpu().numpy().tolist())
            else:
                all_cls_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_ids.extend(batch["pid"].cpu().numpy().tolist())

    payload: dict[str, np.ndarray] = {
        "ids": np.asarray(all_ids, dtype=np.int64),
        "cls_pred": np.asarray(all_cls_preds, dtype=np.int64),
    }
    if use_regression_head:
        payload["phq_log_pred"] = np.asarray(all_phq_log_preds, dtype=np.float64)
        payload["phq_raw_pred"] = denormalize_phq(payload["phq_log_pred"])
    return payload


def resolve_class_predictions(
    inference: dict[str, np.ndarray],
    task: str,
    class_source: str,
    threshold_mode: str,
    class_proportions: list[float],
    explicit_thresholds: list[float],
    oof_phq_raw: np.ndarray | None = None,
    oof_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    num_classes = get_num_classes(task)
    if class_source == "cls":
        return inference["cls_pred"], []

    if "phq_raw_pred" not in inference:
        raise ValueError("class_source=reg requires regression head outputs")

    phq_raw = inference["phq_raw_pred"]
    if explicit_thresholds:
        thresholds = explicit_thresholds
    elif threshold_mode == "f1":
        if oof_phq_raw is None or oof_labels is None:
            raise ValueError("threshold_mode=f1 requires OOF predictions and labels")
        thresholds, _ = calibrate_threshold_f1(oof_phq_raw, oof_labels, num_classes=num_classes)
    else:
        thresholds = calibrate_threshold_dist_align(phq_raw, class_proportions)

    class_pred = apply_thresholds(phq_raw, thresholds, num_classes=num_classes)
    return class_pred, thresholds


def write_submission_csv(
    output_csv: Path,
    ids: np.ndarray,
    class_pred: np.ndarray,
    phq_raw_pred: np.ndarray,
    task: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if task == "binary":
        fieldnames = ["id", "binary_pred", "phq9_pred"]
        rows = [
            {
                "id": int(sample_id),
                "binary_pred": int(label),
                "phq9_pred": float(phq),
            }
            for sample_id, label, phq in zip(ids, class_pred, phq_raw_pred)
        ]
    elif task == "ternary":
        fieldnames = ["id", "ternary_pred", "phq9_pred"]
        rows = [
            {
                "id": int(sample_id),
                "ternary_pred": int(label),
                "phq9_pred": float(phq),
            }
            for sample_id, label, phq in zip(ids, class_pred, phq_raw_pred)
        ]
    else:
        raise ValueError(f"Unsupported task for submission csv: {task}")

    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_predict_only(args: argparse.Namespace, logger: logging.Logger) -> Path:
    checkpoint_path = resolve_project_path(remap_repo_path(args.checkpoint))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    track = require_checkpoint_value(checkpoint, "track")
    task = require_checkpoint_value(checkpoint, "task")
    if task not in {"binary", "ternary"}:
        raise ValueError(f"predict_only supports binary/ternary, got task={task}")

    subtrack = require_checkpoint_value(checkpoint, "subtrack")
    data_root = remap_repo_path(args.data_root or require_checkpoint_value(checkpoint, "data_root"))
    split_csv = remap_repo_path(args.split_csv or require_checkpoint_value(checkpoint, "split_csv"))
    personality_npy = remap_repo_path(args.personality_npy or require_checkpoint_value(checkpoint, "personality_npy"))
    trainval_split_csv = remap_repo_path(
        args.trainval_split_csv
        or "MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/split_labels_train.csv"
    )
    target_t = int(require_checkpoint_value(checkpoint, "target_t"))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    if not args.output_csv:
        raise ValueError("--predict_only requires --output_csv")

    task_maps = load_task_maps(split_csv, task, "label2")
    test_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack=subtrack,
        task=task,
        audio_feature=require_checkpoint_value(checkpoint, "audio_feature"),
        video_feature=require_checkpoint_value(checkpoint, "video_feature"),
        personality_npy=personality_npy,
        phq_map=None,
        target_t=target_t,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    model_kwargs = dict(require_checkpoint_value(checkpoint, "model_kwargs"))
    model = TorchcatBaseline(**model_kwargs).to(device)
    model.load_state_dict(require_checkpoint_value(checkpoint, "model_state"))
    use_regression_head = bool(model_kwargs.get("use_regression_head", False))
    head_type = model_kwargs.get("head_type", "softmax")

    inference = run_model_inference(model, test_loader, device, use_regression_head, head_type)
    class_proportions = load_trainval_class_proportions(trainval_split_csv, task)
    explicit_thresholds = parse_thresholds(args.thresholds)

    if args.class_source == "reg" and not use_regression_head:
        raise ValueError("class_source=reg requires checkpoint with regression head")

    class_pred, thresholds = resolve_class_predictions(
        inference=inference,
        task=task,
        class_source=args.class_source,
        threshold_mode=args.threshold_mode,
        class_proportions=class_proportions,
        explicit_thresholds=explicit_thresholds,
    )

    if use_regression_head:
        phq_raw_pred = inference["phq_raw_pred"]
    else:
        phq_raw_pred = np.zeros(inference["ids"].shape[0], dtype=np.float64)

    output_csv = resolve_project_path(args.output_csv)
    write_submission_csv(output_csv, inference["ids"], class_pred, phq_raw_pred, task)

    logger.info("Predict-only CSV saved to: %s", to_project_relative_path(output_csv))
    logger.info(
        "class_source=%s threshold_mode=%s thresholds=%s class_proportions=%s",
        args.class_source,
        args.threshold_mode,
        thresholds,
        [round(item, 4) for item in class_proportions],
    )
    positive_rate = float(np.mean(class_pred == 1)) if task == "binary" else None
    if positive_rate is not None:
        logger.info("binary positive rate (pred=1): %.4f", positive_rate)
    return output_csv


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    if args.predict_only:
        run_predict_only(args, logger)
        return

    checkpoint_path = resolve_project_path(remap_repo_path(args.checkpoint))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    track = require_checkpoint_value(checkpoint, "track")
    task = require_checkpoint_value(checkpoint, "task")
    regression_label = checkpoint.get("regression_label", "")
    subtrack = require_checkpoint_value(checkpoint, "subtrack")
    encoder_type = require_checkpoint_value(checkpoint, "encoder_type")
    audio_feature = require_checkpoint_value(checkpoint, "audio_feature")
    video_feature = require_checkpoint_value(checkpoint, "video_feature")
    data_root = remap_repo_path(args.data_root or require_checkpoint_value(checkpoint, "data_root"))
    split_csv = remap_repo_path(args.split_csv or require_checkpoint_value(checkpoint, "split_csv"))
    personality_npy = remap_repo_path(args.personality_npy or require_checkpoint_value(checkpoint, "personality_npy"))
    target_t = int(require_checkpoint_value(checkpoint, "target_t"))
    experiment_name = checkpoint.get("experiment_name", checkpoint_path.parent.name)

    timestamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    logs_root = resolve_project_path(remap_repo_path(args.logs_dir))
    log_dir = resolve_track_task_dir(logs_root, track, subtrack, task, experiment_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    task_maps = load_task_maps(split_csv, task, regression_label or "label2")
    test_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack=subtrack,
        task=task,
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=personality_npy,
        phq_map=task_maps.get("test_phq_map"),
        target_t=target_t,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    model_kwargs = dict(require_checkpoint_value(checkpoint, "model_kwargs"))
    model = TorchcatBaseline(**model_kwargs).to(device)
    model.load_state_dict(require_checkpoint_value(checkpoint, "model_state"))
    use_regression_head = bool(model_kwargs.get("use_regression_head", False))
    is_regression_task = task == REGRESSION_TASK
    criterion = (nn.CrossEntropyLoss(), nn.MSELoss()) if use_regression_head else nn.CrossEntropyLoss()
    metrics = evaluate_model(model, test_loader, criterion, device, task)
    metric_summary = summarize_metrics(metrics)
    checkpoint_rel = to_project_relative_path(checkpoint_path)

    result_payload = {
        "checkpoint": checkpoint_rel,
        "track": track,
        "task": task,
        "subtrack": subtrack,
        "encoder_type": encoder_type,
        "audio_feature": audio_feature,
        "video_feature": video_feature,
        "regression_label": regression_label if is_regression_task else "",
        "metrics": metric_summary,
        "predictions_path": "",
    }
    result_path = log_dir / f"test_result_only_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    summary_row = {
        "timestamp": timestamp,
        "mode": "test_only",
        "track": track,
        "task": task,
        "subtrack": subtrack,
        "encoder_type": encoder_type,
        "audio_feature": audio_feature,
        "video_feature": video_feature,
        "checkpoint": checkpoint_rel,
        "predictions_path": "",
        "Macro-F1": f"{metrics.get('f1', 0.0):.6f}",
        "ACC": f"{metrics.get('acc', 0.0):.6f}",
        "Kappa": f"{metrics.get('kappa', 0.0):.6f}",
        "CCC": f"{metrics['ccc']:.6f}",
        "RMSE": f"{metrics['rmse']:.6f}",
        "MAE": f"{metrics['mae']:.6f}",
        "R2": f"{metrics.get('r2', ''):.6f}" if is_regression_task else "",
    }
    if is_regression_task:
        summary_row["regression_label"] = regression_label
    append_summary_row(log_dir / f"{experiment_name}_test_only.csv", summary_row)
    logger.info("Test-only metrics saved to: %s", to_project_relative_path(result_path))


if __name__ == "__main__":
    main()

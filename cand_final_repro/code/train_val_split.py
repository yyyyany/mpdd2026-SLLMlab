from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import LeaveOneOut, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

from dataset import REGRESSION_TASK, get_phq9_target, get_task_label, resolve_project_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POOL_SPLITS = {"", "train", "val"}


def _load_train_rows(split_csv: str | Path) -> list[dict[str, str]]:
    csv_path = resolve_project_path(split_csv)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Split CSV is empty: {csv_path}")

    train_rows = [row for row in rows if row.get("split", "train").strip().lower() in POOL_SPLITS]
    if not train_rows:
        raise ValueError(f"No train rows found in split CSV: {csv_path}")
    return train_rows


def create_train_val_split(
    split_csv: str | Path,
    task: str,
    val_ratio: float = 0.1,
    regression_label: str = "label2",
) -> dict[str, Any]:
    rows = _load_train_rows(split_csv)
    sample_ids = [int(row["ID"]) for row in rows]
    sample_labels = [get_task_label(row, task, regression_label) for row in rows]

    if len(sample_ids) < 1:
        raise ValueError("At least one train sample is required to create a train/val split.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    if val_ratio == 0.0:
        train_id_split = sorted(int(item) for item in sample_ids)
        val_id_split: list[int] = []
        train_id_set = set(train_id_split)
        val_id_set: set[int] = set()
        source_split_map = {int(row["ID"]): "train" for row in rows}
        train_map = {
            int(row["ID"]): get_task_label(row, task, regression_label)
            for row in rows
            if int(row["ID"]) in train_id_set
        }
        val_map: dict[int, int] = {}
        return {
            "train_ids": train_id_split,
            "val_ids": val_id_split,
            "train_map": train_map,
            "val_map": val_map,
            "source_split_map": source_split_map,
            "rows": rows,
            "split_label": regression_label if task == REGRESSION_TASK else ("label2" if task == "binary" else "label3"),
            "train_phq_map": {int(row["ID"]): get_phq9_target(row) for row in rows if int(row["ID"]) in train_id_set},
            "val_phq_map": {},
        }

    if len(sample_ids) < 2:
        raise ValueError("At least two train samples are required when val_ratio > 0.")

    label_counts = Counter(int(label) for label in sample_labels)
    splitter: StratifiedShuffleSplit | ShuffleSplit
    if label_counts and min(label_counts.values()) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=1.0 - val_ratio)
        train_indices, val_indices = next(splitter.split(sample_ids, sample_labels))
    else:
        splitter = ShuffleSplit(n_splits=1, train_size=1.0 - val_ratio)
        train_indices, val_indices = next(splitter.split(sample_ids))

    train_id_split = [sample_ids[index] for index in train_indices]
    val_id_split = [sample_ids[index] for index in val_indices]

    train_id_split = sorted(int(item) for item in train_id_split)
    val_id_split = sorted(int(item) for item in val_id_split)
    train_id_set = set(train_id_split)
    val_id_set = set(val_id_split)

    source_split_map = {int(row["ID"]): "train" for row in rows}
    train_map = {int(row["ID"]): get_task_label(row, task, regression_label) for row in rows if int(row["ID"]) in train_id_set}
    val_map = {int(row["ID"]): get_task_label(row, task, regression_label) for row in rows if int(row["ID"]) in val_id_set}

    payload = {
        "train_ids": train_id_split,
        "val_ids": val_id_split,
        "train_map": train_map,
        "val_map": val_map,
        "source_split_map": source_split_map,
        "rows": rows,
        "split_label": regression_label if task == REGRESSION_TASK else ("label2" if task == "binary" else "label3"),
        "train_phq_map": {int(row["ID"]): get_phq9_target(row) for row in rows if int(row["ID"]) in train_id_set},
        "val_phq_map": {int(row["ID"]): get_phq9_target(row) for row in rows if int(row["ID"]) in val_id_set},
    }
    return payload


def create_cv_splits(
    split_csv: str | Path,
    task: str,
    cv_mode: str = "kfold",
    n_splits: int = 5,
    seed: int = 42,
    regression_label: str = "label2",
) -> list[dict[str, Any]]:
    rows = _load_train_rows(split_csv)
    sample_ids = np.asarray([int(row["ID"]) for row in rows], dtype=np.int64)
    sample_labels = np.asarray([get_task_label(row, task, regression_label) for row in rows], dtype=np.int64)

    if sample_ids.size < 2:
        raise ValueError("At least two train samples are required for cross-validation.")

    label_counts = Counter(int(label) for label in sample_labels.tolist())
    use_stratified = min(label_counts.values()) >= 2 and cv_mode != "loo"

    folds: list[dict[str, Any]] = []
    if cv_mode == "loo":
        splitter = LeaveOneOut()
        split_iter = splitter.split(sample_ids)
    elif cv_mode == "kfold":
        if use_stratified:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            splitter = ShuffleSplit(n_splits=n_splits, test_size=max(1, sample_ids.size // n_splits), random_state=seed)
        split_iter = splitter.split(sample_ids, sample_labels if use_stratified else None)
    else:
        raise ValueError(f"Unsupported cv_mode: {cv_mode}")

    source_split_map = {int(row["ID"]): "train" for row in rows}
    for fold_idx, (train_indices, val_indices) in enumerate(split_iter):
        train_ids = sorted(int(sample_ids[index]) for index in train_indices)
        val_ids = sorted(int(sample_ids[index]) for index in val_indices)
        train_set = set(train_ids)
        val_set = set(val_ids)
        folds.append(
            {
                "fold": fold_idx,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_map": {
                    int(row["ID"]): get_task_label(row, task, regression_label)
                    for row in rows
                    if int(row["ID"]) in train_set
                },
                "val_map": {
                    int(row["ID"]): get_task_label(row, task, regression_label)
                    for row in rows
                    if int(row["ID"]) in val_set
                },
                "source_split_map": source_split_map,
                "train_phq_map": {int(row["ID"]): get_phq9_target(row) for row in rows if int(row["ID"]) in train_set},
                "val_phq_map": {int(row["ID"]): get_phq9_target(row) for row in rows if int(row["ID"]) in val_set},
            }
        )
    return folds


def save_split_preview(
    rows: list[dict[str, str]],
    train_ids: list[int],
    val_ids: list[int],
    save_path: str | Path,
) -> Path:
    save_path = resolve_project_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    train_set = set(train_ids)
    val_set = set(val_ids)
    fieldnames = list(rows[0].keys())

    with open(save_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            person_id = int(row["ID"])
            split_name = row.get("split", "train").strip().lower()
            new_row = dict(row)
            if split_name in POOL_SPLITS:
                if person_id in train_set:
                    new_row["split"] = "train"
                elif person_id in val_set:
                    new_row["split"] = "val"
            writer.writerow(new_row)
    return save_path


def to_project_relative_path(path_like: str | Path) -> str:
    path = resolve_project_path(path_like)
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a train/val split from official MPDD-AVG train IDs.")
    parser.add_argument("--task", required=True, choices=["binary", "ternary", REGRESSION_TASK])
    parser.add_argument("--regression_label", default="label2", choices=["label2", "label3"])
    parser.add_argument("--split_csv", default="MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/split_labels_train.csv")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save_path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_payload = create_train_val_split(
        split_csv=args.split_csv,
        task=args.task,
        val_ratio=args.val_ratio,
        regression_label=args.regression_label,
    )
    if args.save_path:
        preview_path = save_split_preview(
            rows=split_payload["rows"],
            train_ids=split_payload["train_ids"],
            val_ids=split_payload["val_ids"],
            save_path=args.save_path,
        )
        print(json.dumps({"save_path": to_project_relative_path(preview_path)}, ensure_ascii=False))
        return

    summary = {
        "task": args.task,
        "regression_label": args.regression_label if args.task == REGRESSION_TASK else "",
        "val_ratio": args.val_ratio,
        "train_count": len(split_payload["train_ids"]),
        "val_count": len(split_payload["val_ids"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

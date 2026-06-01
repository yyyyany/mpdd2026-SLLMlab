from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_map(rows: list[dict[str, str]], key: str) -> dict[str, str]:
    return {str(row["id"]): row[key] for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a hybrid submission: class preds from one source, phq9 from another."
    )
    parser.add_argument("--binary_class_csv", required=True, help="提供 binary_pred 的 CSV")
    parser.add_argument("--ternary_class_csv", required=True, help="提供 ternary_pred 的 CSV")
    parser.add_argument("--phq9_csv", required=True, help="提供 phq9_pred 的 CSV（用于两个文件）")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    binary_rows = read_csv(args.binary_class_csv)
    ternary_rows = read_csv(args.ternary_class_csv)
    phq9_rows = read_csv(args.phq9_csv)

    phq9_map = to_map(phq9_rows, "phq9_pred")
    binary_class_map = to_map(binary_rows, "binary_pred")
    ternary_class_map = to_map(ternary_rows, "ternary_pred")

    ids = [str(row["id"]) for row in binary_rows]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_out = output_dir / "binary_hybrid.csv"
    ternary_out = output_dir / "ternary_hybrid.csv"

    with open(binary_out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "binary_pred", "phq9_pred"])
        writer.writeheader()
        for sample_id in ids:
            writer.writerow(
                {
                    "id": sample_id,
                    "binary_pred": binary_class_map[sample_id],
                    "phq9_pred": phq9_map[sample_id],
                }
            )

    with open(ternary_out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "ternary_pred", "phq9_pred"])
        writer.writeheader()
        for sample_id in ids:
            writer.writerow(
                {
                    "id": sample_id,
                    "ternary_pred": ternary_class_map[sample_id],
                    "phq9_pred": phq9_map[sample_id],
                }
            )

    print(f"Generated: {binary_out}")
    print(f"Generated: {ternary_out}")


if __name__ == "__main__":
    main()

"""对比复现产物与 reference/submission，用于主办方验收。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
REFERENCE_DIR = PACKAGE_ROOT / "reference" / "submission"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_submission(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read {path}")


def compare_csv(name: str, ref_path: Path, gen_path: Path, phq_atol: float) -> list[str]:
    errors: list[str] = []
    if not gen_path.exists():
        return [f"{name}: missing {gen_path}"]
    ref = read_submission(ref_path).sort_values("id").reset_index(drop=True)
    gen = read_submission(gen_path).sort_values("id").reset_index(drop=True)
    if list(ref.columns) != list(gen.columns):
        errors.append(f"{name}: column mismatch {list(ref.columns)} vs {list(gen.columns)}")
        return errors
    if len(ref) != len(gen):
        errors.append(f"{name}: row count {len(ref)} vs {len(gen)}")
        return errors
    if not np.array_equal(ref["id"].astype(int).values, gen["id"].astype(int).values):
        errors.append(f"{name}: id mismatch")
        return errors
    pred_col = "binary_pred" if "binary" in name else "ternary_pred"
    if not np.array_equal(ref[pred_col].astype(int).values, gen[pred_col].astype(int).values):
        diff = ref[ref[pred_col].astype(int) != gen[pred_col].astype(int)][["id", pred_col]].head(5)
        errors.append(f"{name}: {pred_col} mismatch, examples:\n{diff}")
    if not np.allclose(ref["phq9_pred"].astype(float), gen["phq9_pred"].astype(float), atol=phq_atol, rtol=0):
        delta = np.abs(ref["phq9_pred"].astype(float) - gen["phq9_pred"].astype(float))
        worst = delta.max()
        errors.append(f"{name}: phq9_pred max abs diff {worst:.6f} > atol {phq_atol}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify cand_final reproduction against reference.")
    parser.add_argument(
        "--generated_dir",
        default=str(PACKAGE_ROOT / "work" / "submission"),
        help="复现脚本生成的 submission 目录",
    )
    parser.add_argument("--phq_atol", type=float, default=1e-4, help="phq9_pred 数值容差")
    args = parser.parse_args()
    gen_dir = Path(args.generated_dir)

    print("=== Reference SHA256 ===")
    for name in ("binary.csv", "ternary.csv"):
        ref = REFERENCE_DIR / name
        print(f"{name}: {sha256_file(ref)}")

    print("\n=== Compare generated vs reference ===")
    all_errors: list[str] = []
    all_errors.extend(compare_csv("binary.csv", REFERENCE_DIR / "binary.csv", gen_dir / "binary.csv", args.phq_atol))
    all_errors.extend(compare_csv("ternary.csv", REFERENCE_DIR / "ternary.csv", gen_dir / "ternary.csv", args.phq_atol))

    if all_errors:
        print("FAILED")
        for err in all_errors:
            print("-", err)
        return 1

    print("PASSED: generated submission matches reference within tolerance.")
    print(f"Reference CodaBench score (informational): see {PACKAGE_ROOT / 'reference' / 'scores.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

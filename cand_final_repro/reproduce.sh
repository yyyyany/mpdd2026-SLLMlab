#!/usr/bin/env bash
# cand_final复现（Linux/macOS）
# 在仓库根目录执行: bash cand_final_repro/reproduce.sh

set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
CODE_DIR="$PACKAGE_ROOT/code"
WORK_DIR="$PACKAGE_ROOT/work"
export PYTHONPATH="$CODE_DIR"
cd "$PROJECT_ROOT"

run_py() {
  conda run -n mpdd python "$@"
}

CKPT_BIN="checkpoints/Track1/A-V-G+P/binary/best_model_2026-04-30-09.48.21.pth"
CKPT_TER="checkpoints/Track1/A-V-G+P/ternary/best_model_2026-04-30-09.27.07.pth"
CKPT_REG="checkpoints/mainline/Track1/A-V-G+P/ternary/ternary_ws_reg3_ep50/best_model_2026-05-31-18.42.24.pth"
for p in "$CKPT_BIN" "$CKPT_TER" "$CKPT_REG"; do
  test -f "$p" || { echo "Missing checkpoint: $p"; exit 1; }
done

TEST_ROOT="MPDD-AVG2026/MPDD-AVG2026-test/Elder"
TEST_SPLIT="$TEST_ROOT/split_labels_test.csv"
PERS_NPY="MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/descriptions_embeddings_with_ids.npy"
test -f "$TEST_SPLIT" || { echo "Missing $TEST_SPLIT"; exit 1; }

mkdir -p "$WORK_DIR/baseline_lock" "$WORK_DIR/reg_best" "$WORK_DIR/hybrid" "$WORK_DIR/submission"

echo "[1/4] binary cls..."
run_py "$CODE_DIR/test.py" --predict_only --checkpoint "$CKPT_BIN" \
  --data_root "$TEST_ROOT" --split_csv "$TEST_SPLIT" --personality_npy "$PERS_NPY" \
  --output_csv "$WORK_DIR/baseline_lock/binary_cls.csv" --class_source cls --device cuda

echo "[2/4] ternary cls..."
run_py "$CODE_DIR/test.py" --predict_only --checkpoint "$CKPT_TER" \
  --data_root "$TEST_ROOT" --split_csv "$TEST_SPLIT" --personality_npy "$PERS_NPY" \
  --output_csv "$WORK_DIR/baseline_lock/ternary_cls.csv" --class_source cls --device cuda

echo "[3/4] reg phq9..."
run_py "$CODE_DIR/test.py" --predict_only --checkpoint "$CKPT_REG" \
  --data_root "$TEST_ROOT" --split_csv "$TEST_SPLIT" --personality_npy "$PERS_NPY" \
  --output_csv "$WORK_DIR/reg_best/ternary_reg.csv" --class_source cls --device cuda

echo "[4/4] hybrid + zip..."
run_py "$CODE_DIR/make_submission_forcodabench/build_hybrid.py" \
  --binary_class_csv "$WORK_DIR/baseline_lock/binary_cls.csv" \
  --ternary_class_csv "$WORK_DIR/baseline_lock/ternary_cls.csv" \
  --phq9_csv "$WORK_DIR/reg_best/ternary_reg.csv" \
  --output_dir "$WORK_DIR/hybrid"
run_py "$CODE_DIR/make_submission_forcodabench/make_submission_sample.py" \
  --binary_csv "$WORK_DIR/hybrid/binary_hybrid.csv" \
  --ternary_csv "$WORK_DIR/hybrid/ternary_hybrid.csv" \
  --output_dir "$WORK_DIR/submission"

echo "[5/5] verify..."
run_py "$PACKAGE_ROOT/verify_reproduction.py" --generated_dir "$WORK_DIR/submission"
echo "Done: $WORK_DIR/submission/submission.zip"

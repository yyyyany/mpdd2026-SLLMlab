# cand_final 复现（Windows PowerShell）
# 在仓库根目录执行: .\cand_final_repro\reproduce.ps1

$ErrorActionPreference = "Stop"
$PackageRoot = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $PackageRoot
$CodeDir = Join-Path $PackageRoot "code"
$WorkDir = Join-Path $PackageRoot "work"
$BaselineDir = Join-Path $WorkDir "baseline_lock"
$RegDir = Join-Path $WorkDir "reg_best"
$HybridDir = Join-Path $WorkDir "hybrid"
$SubmitDir = Join-Path $WorkDir "submission"

$env:PYTHONPATH = $CodeDir
Set-Location $ProjectRoot

$PY = "conda"
$PyArgs = @("run", "-n", "mpdd", "python")
function Invoke-Mpdd([string[]]$ScriptArgs) {
    & $PY @PyArgs @ScriptArgs
    if ($LASTEXITCODE -ne 0) { throw "Command failed: python $($ScriptArgs -join ' ')" }
}

Write-Host "=== cand_final reproduction ==="
Write-Host "Project root: $ProjectRoot"

$ckptBin = "checkpoints/Track1/A-V-G+P/binary/best_model_2026-04-30-09.48.21.pth"
$ckptTer = "checkpoints/Track1/A-V-G+P/ternary/best_model_2026-04-30-09.27.07.pth"
$ckptReg = "checkpoints/mainline/Track1/A-V-G+P/ternary/ternary_ws_reg3_ep50/best_model_2026-05-31-18.42.24.pth"
foreach ($p in @($ckptBin, $ckptTer, $ckptReg)) {
    if (-not (Test-Path (Join-Path $ProjectRoot $p))) {
        throw "Missing checkpoint: $p (see cand_final_repro/MANIFEST.md)"
    }
}

$testRoot = "MPDD-AVG2026/MPDD-AVG2026-test/Elder"
$testSplit = "$testRoot/split_labels_test.csv"
$persNpy = "MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/descriptions_embeddings_with_ids.npy"
if (-not (Test-Path (Join-Path $ProjectRoot $testSplit))) {
    throw "Missing test split: $testSplit"
}

New-Item -ItemType Directory -Force -Path $BaselineDir, $RegDir, $HybridDir, $SubmitDir | Out-Null

Write-Host "`n[1/4] Official binary classification (argmax)..."
Invoke-Mpdd @(
    (Join-Path $CodeDir "test.py"),
    "--predict_only",
    "--checkpoint", $ckptBin,
    "--data_root", $testRoot,
    "--split_csv", $testSplit,
    "--personality_npy", $persNpy,
    "--output_csv", (Join-Path $BaselineDir "binary_cls.csv"),
    "--class_source", "cls",
    "--device", "cuda"
)

Write-Host "`n[2/4] Official ternary classification (argmax)..."
Invoke-Mpdd @(
    (Join-Path $CodeDir "test.py"),
    "--predict_only",
    "--checkpoint", $ckptTer,
    "--data_root", $testRoot,
    "--split_csv", $testSplit,
    "--personality_npy", $persNpy,
    "--output_csv", (Join-Path $BaselineDir "ternary_cls.csv"),
    "--class_source", "cls",
    "--device", "cuda"
)

Write-Host "`n[3/4] Regression phq9 (ternary_ws_reg3_ep50)..."
Invoke-Mpdd @(
    (Join-Path $CodeDir "test.py"),
    "--predict_only",
    "--checkpoint", $ckptReg,
    "--data_root", $testRoot,
    "--split_csv", $testSplit,
    "--personality_npy", $persNpy,
    "--output_csv", (Join-Path $RegDir "ternary_reg.csv"),
    "--class_source", "cls",
    "--device", "cuda"
)

Write-Host "`n[4/4] Hybrid merge + submission.zip..."
Invoke-Mpdd @(
    (Join-Path $CodeDir "make_submission_forcodabench/build_hybrid.py"),
    "--binary_class_csv", (Join-Path $BaselineDir "binary_cls.csv"),
    "--ternary_class_csv", (Join-Path $BaselineDir "ternary_cls.csv"),
    "--phq9_csv", (Join-Path $RegDir "ternary_reg.csv"),
    "--output_dir", $HybridDir
)
Invoke-Mpdd @(
    (Join-Path $CodeDir "make_submission_forcodabench/make_submission_sample.py"),
    "--binary_csv", (Join-Path $HybridDir "binary_hybrid.csv"),
    "--ternary_csv", (Join-Path $HybridDir "ternary_hybrid.csv"),
    "--output_dir", $SubmitDir
)

Write-Host "`n[5/5] Verify against reference..."
Invoke-Mpdd @(
    (Join-Path $PackageRoot "verify_reproduction.py"),
    "--generated_dir", $SubmitDir
)

Write-Host "`nDone. Submission: $SubmitDir\submission.zip"

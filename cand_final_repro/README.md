# cand_final 复现（MPDD-AVG 2026 Track1 / Elder /A-V-G+P）

本目录为提交方案 **`cand_final`**（CodaBench Score **0.5226**）的复现。

## 方法说明

| 输出列 | 模型 | Checkpoint |
|--------|------|------------|
| `binary_pred` | 官方 A-V-G+P binary，softmax argmax | `checkpoints/Track1/A-V-G+P/binary/best_model_2026-04-30-09.48.21.pth` |
| `ternary_pred` | 官方 A-V-G+P ternary，softmax argmax | `checkpoints/Track1/A-V-G+P/ternary/best_model_2026-04-30-09.27.07.pth` |
| `phq9_pred` | warm-start 后全集微调的 ternary 双头模型**回归支路** | `checkpoints/mainline/.../ternary_ws_reg3_ep50/best_model_2026-05-31-18.42.24.pth` |

测试阶段用 `build_hybrid.py` 按样本 ID 合并三份 CSV，再打包为 `submission.zip`（`binary.csv` + `ternary.csv`）。

## 目录结构

```
cand_final_repro/
  README.md                 # 本文件
  MANIFEST.md               # 数据与权重路径清单
  requirements.txt
  reproduce.ps1             # Windows 一键复现
  reproduce.sh              # Linux/macOS 一键复现
  verify_reproduction.py    # 与 reference 逐列比对
  reference/
    submission/             # 已提交的  CSV
    scores.txt              # CodaBench 实测分项（仅作对照）
  code/                     # 最小源码
  work/                     # 运行后生成（可删）
```

## 环境

- Python 3.10+，CUDA 建议可用  
- 依赖见 `requirements.txt`  
- 环境：`conda create -n mpdd python=3.10` 后 `pip install -r cand_final_repro/requirements.txt`

## 前置资源

将 MPDD-AVG2026 **测试集**与 **3 个 checkpoint** 按 `MANIFEST.md` 放入**仓库根目录**（`cand_final_repro` 的上一级）。

## 复现

在**仓库根目录**执行：

**Windows（PowerShell）：**

```powershell
.\cand_final_repro\reproduce.ps1
```

**Linux / macOS：**

```bash
bash cand_final_repro/reproduce.sh
```

成功时终端输出 `PASSED`，结果为：

`cand_final_repro/work/submission/submission.zip`



主仓库；**仅本包 + MANIFEST 资源即可复现提交文件**。

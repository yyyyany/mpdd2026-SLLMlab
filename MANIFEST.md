# cand_final 资源清单

## 概述

`cand_final` 为**模块化拼装提交**：

| 列 | 来源 | 说明 |
|---|---|---|
| `binary_pred` | 官方 binary checkpoint，argmax | mfcc + resnet |
| `ternary_pred` | 官方 ternary checkpoint，argmax | wav2vec + openface |
| `phq9_pred` | 自训 `ternary_ws_reg3_ep50` 回归头 | warm-start 官方 ternary，全集 50 epoch |

CodaBench 实测 Score：**0.5226**（见 `reference/scores.txt`）。

## 目录（相对仓库根目录）

### 测试集特征

```
MPDD-AVG2026/MPDD-AVG2026-test/Elder/
  split_labels_test.csv
  ...（音频/视频/步态特征，与 baseline 一致）
```

### Trainval 人格向量（test 推理时加载）

```
MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/descriptions_embeddings_with_ids.npy
```

### Checkpoints（3 个）

```
checkpoints/Track1/A-V-G+P/binary/best_model_2026-04-30-09.48.21.pth
checkpoints/Track1/A-V-G+P/ternary/best_model_2026-04-30-09.27.07.pth
checkpoints/mainline/Track1/A-V-G+P/ternary/ternary_ws_reg3_ep50/best_model_2026-05-31-18.42.24.pth
```


训练配置见 `README.md` 附录 A。

## 不含

- 原始数据集本体（体积过大，需自行放置）
- checkpoint 权重文件（已放置）


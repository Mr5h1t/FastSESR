# FastSESR 保覆盖平面重叠修复

日期：2026-10-10（Asia/Shanghai）。这轮先测试了几何自适应置信度，开发集未通过约束，保留了负结果；随后新增一个独立的可选网格后处理：对有重复覆盖的平面区域重新三角化，检查覆盖面积和接缝，而不是单纯删掉低置信度面。

## 正式结果

固定参数后共测试 40 个 ABC 文件。结束后按点云与 GT 的 SHA-256 排查，发现 2 个文件虽然 CAD 文档名不同，却与开发或此前测试样本字节相同；按与指标无关的数据重复规则排除，正式结论使用剩余 **38 个独立文件**。三个方案都在同一组数据、三个采样种子下重新计算。原始 40 个样本数据和统计完整保留。

| 指标 | 原方案 | 固定 Top-2=0.6 | 保覆盖平面修复 |
|---|---:|---:|---:|
| 采样点 CD1 ×100 ↓ | 0.280570 | 0.280546 | 0.280562 |
| 采样点 CD2 ×10⁵ ↓ | 0.532275 | 0.532301 | 0.532289 |
| 点到连续表面 CD1 ×100 ↓ | 0.000464 | 0.000492 | 0.000469 |
| 点到连续表面 CD2 ×10⁵ ↓ | 0.001283 | 0.001360 | 0.001300 |
| Precision ↑ | 95.23991% | 95.22989% | 95.22835% |
| Recall ↑ | 95.15891% | 95.15229% | 95.15211% |
| F1 @0.003 ↑ | 95.19901% | 95.19067% | 95.18982% |
| 法线一致性 ↑ | 98.54880% | 98.54352% | 98.55249% |
| 平均三角面数 | 22817.342105 | 22799.631579 | 22799.947368 |
| 平均非流形边数 ↓ | 68.894737 | 42.789474 | 34.421053 |
| 平均边界边数 | 55.368421 | 57.526316 | 46.947368 |
| 完整重建时间（秒） | 0.366938 | 0.366947 | 1.006834 |
| 额外峰值显存（MiB） | 2985.307322 | 2985.307052 | 2985.307322 |

相对原方案，采样 CD1 变化 **-0.00292%**，CD2 变化 **+0.00260%**；平均非流形边变化 **-50.04%**，平均边界边变化 **-15.21%**。负值代表下降。

采样 CD1 的形状配对平均差异 95% 百分位区间为 [-0.00010416, +0.00009091]。CD1 区间包含零，不能认为采样 CD 有稳定提升。

38 个独立样本中有 36 个实际修复了平面片。完整重建耗时相对变化为 **+174.39%**。新增过程在 CPU 上进行几何集合和三角化运算，具有时间代价；不能宣称本轮再次加速。默认关闭，用户按需开启。

连续表面 CD1 相对变化为 **+0.926%**，其配对 95% 区间为 [2.250922903331667e-07, 1.1567654168124978e-05]，显示轻微退化。F1 变化为 **-0.00919 个百分点**。因此，这是一种改善网格连接质量的取舍，不能称为提高重建精度。

## 为何改变方向

上轮统一提高阈值减少了非流形边，却增加了边界边。检查 CD1 变差案例，被删面的中心距离 GT 表面很近，可能误删有用面。

新增的几何自适应阈值使用三角形形状质量调整 Top-2 门槛。16 个开发样本、11 个方案、三个 100000 点采样种子的搜索选回了原方案，主要原因是新规则未通过连续表面误差约束。它仍是实验选项，默认权重为 0；详见 [负结果报告](../adaptive_quality/REPORT.md)。该连续距离在部分干净开发样本接近浮点数值分辨率，相对约束很保守，不能把这一次筛选当作所有自适应规则都无效的证明。

这些干净 ABC 样本中，已有网格非常贴近 GT；项目的采样点 CD 还包含有限点采样间距，细微变化容易受采样影响。新修复主要尝试改善重复覆盖和连接质量，同时测量 CD 是否受益。

## 算法与保护条件

1. 根据共享边和近似平行的面法线建立组件；对整个组件检查顶点到一个共同平面的最大距离。只修复平面区域，曲面不通过检查时保持原样。
2. 将组件投影到二维，计算原三角面的集合并集。仅有明显重复覆盖的组件进入处理。
3. 对原有顶点做二维 Delaunay 三角化，只保留落在原区域内的三角面。
4. 检查新旧区域并集的对称差面积不超过容差，保留所有原顶点，并保留与其他组件相连的接缝边，接缝关联面数不得增加。不通过时恢复原组件。
5. 最终再检查整个网格的索引边计数：非流形边和边界边均不得增加，否则恢复整个原网格。

顶点坐标不移动，也不生成新顶点。三维组件只是容差内共面，因此二维覆盖保持不能解释为严格的三维曲面完全相同；三维影响还通过连续表面距离检查。方法也不保证所有自交、顶点非流形或所有曲面缺陷被解决。

组件分组使用一次排序，避免每个组件扫描所有面的二次复杂度。优化前实现快照已保留，另外在七个真实测试样本上比较旧分组与新分组，面索引和修复统计完全一致（`grouping_equivalence.json`）。

二维区域操作使用 Shapely 的 [union_all](https://shapely.readthedocs.io/en/stable/reference/shapely.union_all.html) 和 [covers](https://shapely.readthedocs.io/en/stable/reference/shapely.covers.html)，三角化使用 SciPy Delaunay。可选依赖安装在 `requirements-geometry.txt`，默认推理不需要 Shapely。

## 实验方案

- 开发样本：之前的 16 个 ABC 开发形状。初始测试名单按 CAD 文档名排除此前所有实验样本；结束后的文件内容去重发现 2 个跨名单重复，独立汇总包含 38 个文件。排除依据及名单在 `verification.json`，不是按表现筛选。字节哈希只能识别字节相同的数据，不保证所有重新编码或变换后的相同几何被识别。
- 平面容差固定为包围盒对角线的 1e-6。数据、参数和保护条件在新测试前固定；未根据测试结果调参。
- 采样点 CD/F1/NC 与原项目定义一致，分别对 GT/预测做包围盒对角线归一化，每个网格每个种子采样 100000 点；三个种子先按形状求均值。
- 点到连续表面 CD 使用 20000 个表面查询点到三角网格的最近距离，是额外诊断，不能混用为项目原 CD。它仍是有限查询近似，不是精确曲面积分。
- 三个方案共享输入、预训练权重及神经预测；固定阈值对照使用 Top-2=0.6。GT 只用于评价，不进入修复规则。
- 完整计时包含神经推理、面提取、CPU 网格组装和新后处理，每形状每方案 3 次，交替顺序。排除读盘、预处理、权重加载和写盘。
- 额外显存是 PyTorch 分配增量，不覆盖新增 CPU 内存；不能凭该字段声称总内存不变。
- 配对区间对 38 个独立形状的多种子均值进行 10000 次重采样。样本来自固定点数分布，不代表全量 ABC。

## 使用

```bash
pip install -r requirements-geometry.txt

python S2_reconstruct.py \
  --use_loon_unet --loon_unet_ckpt runs/S2_train/model_best.pth \
  --dataset ABC_test --data_root Data \
  --test_list benchmarks/results/2026-10-10/planar_repair/test_list.txt \
  --out_dir results/ABC_planar_repaired \
  --confidence_top1 0.8 --confidence_top2 0.5 --face_angle 120 \
  --repair_planar_overlaps --planar_tolerance 1e-6
```

修复统计保存为 `planar_repair_stats.json`。保持原结果时不传 `--repair_planar_overlaps`。几何阈值自适应仍默认关闭。

## 复现与证据

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python benchmarks/compare_planar_repair.py \
  --protocol benchmarks/results/2026-10-10/planar_repair/protocol.json \
  --out benchmarks/results/2026-10-10/planar_repair_reproduction

python benchmarks/audit_planar_results.py \
  --out benchmarks/results/2026-10-10/planar_repair_reproduction --skip-cli-check

python benchmarks/summarize_planar_repair.py \
  --out benchmarks/results/2026-10-10/planar_repair_reproduction
```

服务器 Python 为 `/home/syx/data/miniconda3/envs/dcx/bin/python`。换机器时修改 protocol 副本的数据路径，保留原始 protocol 和 manifest。

- `development_raw.csv`：144 行开发指标。
- `test_raw.csv`：360 行正式指标；连续距离每个形状每方案计算一次，复用在三个种子行中。
- `test_per_shape.csv`：120 行形状均值；`timing_raw.csv`：360 次完整推理。
- `repair_per_shape.csv`：56 个形状的修复、覆盖、接缝和全局回退记录。
- `fixed_before_test.json`：测试前保存的固定参数与开发结果。
- `manifest.json`：源码、权重、输入哈希、依赖版本和模型配置。
- `heldout_comparison.csv`、`heldout_comparison.md`、`heldout_comparison.png`、`heldout_comparison.svg`、`heldout_summary.json`、`heldout_paired_analysis.json`：38 个独立文件的正式汇总。
- `comparison.csv`、`comparison.md`、`comparison.png`、`comparison.svg`、`paired_analysis.json`：原始 40 个文件的附加汇总，不作为独立测试结论。
- `verification.json`：源码权重、原始指标、输出拓扑、真实 CLI 字节一致和数据去重检查。
- `baseline/`、`strict_0.6/`、`planar_repaired/`：输出网格留在服务器。

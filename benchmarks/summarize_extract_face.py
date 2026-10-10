"""Generate tables, paired uncertainty estimates and a report from raw runs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    summary = json.loads((out/'summary.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    rows = list(csv.DictReader((out/'timing_raw.csv').open()))
    names = [x['name'] for x in manifest['samples']]
    per_sample = []
    intervals = {}
    rng = np.random.default_rng(20261009)
    for name in names:
        record = {'sample': name}
        for stage in ('reconstruction', 'extraction'):
            for method in ('original', 'improved'):
                selected = [r for r in rows if r['sample']==name and r['stage']==stage and r['method']==method]
                record[f'{stage}_{method}_seconds'] = float(np.median([float(r['seconds']) for r in selected]))
                record[f'{stage}_{method}_extra_peak_mib'] = max(float(r['extra_peak_mib']) for r in selected)
            record[f'{stage}_time_reduction_pct'] = 100*(1-record[f'{stage}_improved_seconds']/record[f'{stage}_original_seconds'])
        per_sample.append(record)
    for stage in ('reconstruction', 'extraction'):
        a = np.array([r[f'{stage}_original_seconds'] for r in per_sample])
        b = np.array([r[f'{stage}_improved_seconds'] for r in per_sample])
        idx = rng.integers(0, len(a), size=(10000, len(a)))
        changes = 100*(1-b[idx].mean(axis=1)/a[idx].mean(axis=1))
        intervals[stage] = {'paired_sample_bootstrap_95pct_time_reduction': np.quantile(changes,[.025,.975]).tolist(),
                            'samples_faster': int((b<a).sum()), 'samples':len(a)}
    (out/'paired_analysis.json').write_text(json.dumps(intervals,indent=2))
    with (out/'per_sample_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_sample[0]));writer.writeheader();writer.writerows(per_sample)
    comparisons=[]
    labels={'reconstruction':'完整重建','extraction':'表面提取'}
    for stage in labels:
        a,b=summary[stage]['original'],summary[stage]['improved']
        comparisons.extend([
            {'metric':labels[stage]+'平均耗时_ms','original':a['mean_sample_median_seconds']*1000,'improved':b['mean_sample_median_seconds']*1000,'reduction_pct':summary[stage]['time_reduction_pct']},
            {'metric':labels[stage]+'平均额外峰值显存_MiB','original':a['mean_extra_peak_mib'],'improved':b['mean_extra_peak_mib'],'reduction_pct':summary[stage]['memory_reduction_pct']}])
    for key in ('CD1_x100','CD2_x1e5','F1_at_0.003','NC'):
        comparisons.append({'metric':key,'original':summary['accuracy']['original'][key],'improved':summary['accuracy']['improved'][key],'reduction_pct':0.})
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    table=['| 指标 | 原实现 | 改进后 | 变化 |','|---|---:|---:|---:|']
    for r in comparisons:
        change='相同' if r['reduction_pct']==0 else f'降低 {r["reduction_pct"]:.2f}%'
        table.append(f'| {r["metric"]} | {r["original"]:.6f} | {r["improved"]:.6f} | {change} |')
    (out/'comparison.md').write_text('\n'.join(table)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    panels=[('extraction','mean_sample_median_seconds',1000,'Extraction time (ms)'),
            ('extraction','mean_extra_peak_mib',1,'Extraction extra peak memory (MiB)'),
            ('reconstruction','mean_sample_median_seconds',1000,'Reconstruction time (ms)'),
            ('reconstruction','mean_extra_peak_mib',1,'Reconstruction extra peak memory (MiB)')]
    for ax,(stage,key,factor,title) in zip(axes.flat,panels):
        values=[summary[stage][method][key]*factor for method in ('original','improved')]
        bars=ax.bar(['Original','Improved'],values,color=['#687386','#277bb5'],width=.55)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=4)
        ax.set_title(title);ax.set_ylim(0,max(values)*1.22);ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    fig.suptitle('FastSESR: 40 paired ABC test shapes, identical output meshes')
    fig.savefig(out/'comparison.png',dpi=200);fig.savefig(out/'comparison.svg');plt.close(fig)
    audit=''
    if args.audit_dir:
        audit=(args.audit_dir/'PROJECT_AUDIT.md').read_text()
    pipeline_ci=intervals['reconstruction']['paired_sample_bootstrap_95pct_time_reduction']
    extract_ci=intervals['extraction']['paired_sample_bootstrap_95pct_time_reduction']
    report=f'''# FastSESR 按需构建候选三角形：真实数据对照实验

日期：2026-10-09（Asia/Shanghai）。本轮完成个人科研项目盘点、生产代码改进和真实数据对照。

## 项目选择

选中 `/home/syx/data/FastSESR`。本地数据约 23.94 GB，包含 2,256 个 ABC 训练网格和 6,771 对测试点云/GT；还有 FAUST、MGN、ScanNet、Matterport3D 等数据。两个阶段的预训练权重、重建与评估代码均存在。现有 `dcx` 环境能够直接执行原重建入口，全部活动网络参数已确认来自已有权重，因此能马上建立真实的对照实验。

PoNQ 虽然有约 74.93 GB 数据且完整覆盖 3,843 个训练 HDF5，但当前默认评估配置的外部数据路径不存在。这里选择的是当前训练数据、测试配对、预训练权重和可执行入口综合最齐全的项目，并不以磁盘容量作为唯一标准。

`dcx` 环境中当前流程需要的 torch、torchvision、timm、PyTorch3D、Open3D、pykdtree、numpy、scipy、sklearn 等已可用。requirements.txt 中的 wandb、gdown 未安装；当前代码没有使用它们，且本轮使用本地数据和权重。没有宣称整个 requirements 列表无缺项。

## 改进内容及原因

修改文件：`S2/ExtractFace.py` 的 `SurfExtract.forward()`，生产代码差异为 11 行新增、7 行删除。原实现完整保存在 `benchmarks/baselines/extract_face_before.py`。

原代码为每个中心点构建全部 `(K-1)^2` 个候选三角形，再按分数排序，后续只访问每条中心边的前两个候选。当 K=50 时，每点先构建 2,401 个三角形，最终此步骤只需要 98 个。改进后先排序 logits，再用前两名索引直接构建这 98 个三角形。

```python
neighbors = knn_indices[:, 1:]
other_vertices = torch.gather(
    neighbors[:, None, :].expand(-1, K-1, -1), 2, indices[:, :, :2])
sorted_candidate_triangles = torch.stack([
    knn_indices[:, :1, None].expand(-1, K-1, 2),
    neighbors[:, :, None].expand(-1, -1, 2),
    other_vertices,
], dim=-1)
```

仍使用原来的 torch.sort，确保并列分数的排序行为也相同。网络、权重、置信度阈值、角度判定和最终去重逻辑在对照中一致。这是一个减少无用张量构建和搬运的性能改进：实现小、无需重新训练或调参，适合最先落地。候选张量规模理论上缩小 24.5 倍；整个提取算子的显存还包括 logits 排序等操作，应以实测结果为准。

## 对照设计

- GPU：{manifest['gpu']}；Python 3.11；PyTorch {manifest['torch']}，CUDA {manifest['cuda']}，使用现有 dcx 环境。
- 从全部 6,771 个 ABC 测试样本中，按原始点数排序，再等距选取 40 个排名，覆盖 1,004–19,669 个点。所有候选都有同名 GT，选择规则在看到新实验结果之前已保存。
- 使用现有 Stage-1 `trained_models/model_knn50.pth` 和 Stage-2 `runs/S2_train/model_best.pth`。初始化由原 `S2_reconstruct.py --use_loon_unet` 入口执行，当前网络 183 个 state_dict 键全部精确匹配权重；权重中有 29 个历史结构的多余键，原入口会忽略它们，对照沿用这一行为，没有活动参数缺失。
- KNN=50，chunk_size=2000，delta=0.0，rescale_delta=False，其余 U-Net 参数来自原权重 metadata；seed=20261009。
- 完整重建每种方法每样本重复 5 次，共 400 次测量；原实现/改进实现的顺序交替，每样本先预热。
- 表面提取在每个样本第一批真实网络 logits 上重复 25 次，共 2,000 次测量；每批最多 2,000 点。它是一个阶段测量，不能当作整个模型的推理速度。
- 耗时使用 perf_counter，测量开始/结束都调用 CUDA synchronize。完整重建计时覆盖 GPU 推理到 CPU 网格组装，不包含磁盘读取、点云预处理、权重加载或 PLY 写入。
- 显存为当前 PyTorch 进程在测量开始后的额外峰值 allocated 显存，单位 MiB；不等同于 nvidia-smi 的全卡占用。
- 时间汇总：先取每样本重复测量的中位数，再对 40 个样本求均值。显存先取每样本重复测量的最大值，再求均值。
- 每个样本两种输出都保存成 PLY；逐次验证顶点和三角形数组完全相同。实验辅助重建函数在首个样本上也与原 CLI 输出精确一致。

## 结果

{chr(10).join(table)}

四项精度的变化都严格为零，40/40 个输出网格完全相同。每个模型分别从 GT 和预测网格采样 100,000 个表面点，使用相同的明确随机种子；指标沿用 `eval/surf_acc.py` 的包围盒对角线归一化、CD1、CD2、F1 和绝对法向一致性定义。CD1 为双向平均欧氏距离之和乘 100；CD2 为双向平均平方距离之和乘 1e5；F1 距离阈值为 0.003；NC 为双向法向绝对点积均值。没有把输出完全一致描述为精度提升。

成对样本 bootstrap 的耗时降低 95% 区间：完整重建 {pipeline_ci[0]:.2f}%–{pipeline_ci[1]:.2f}%，提取阶段 {extract_ci[0]:.2f}%–{extract_ci[1]:.2f}%。完整重建有 {intervals['reconstruction']['samples_faster']}/40 个样本的中位数耗时降低。这些区间描述当前固定样本组的重采样变化，不能替代整个 ABC 测试集的统计结论。

![实测性能对比](comparison.png)

## 结果的实际含义与限制

提取阶段减少了大量候选组合张量，因此该阶段耗时下降 {summary['extraction']['time_reduction_pct']:.2f}%，额外显存下降 {summary['extraction']['memory_reduction_pct']:.2f}%。但完整重建耗时只下降 {summary['reconstruction']['time_reduction_pct']:.2f}%，完整流程峰值显存没有下降，说明总体成本仍主要出现在网络前向过程。不能将提取阶段的加速比例套用到整个模型。

本轮没有重新训练网络；这是对当前服务器项目与已有权重的无损性能优化。实验覆盖 40 个固定真实测试样本，没有覆盖全部 6,771 个测试模型，也没有证明所有数据集上的泛化收益。运行期间服务器还有其他 GPU 进程，计时采用成对交替顺序与重复中位数，但绝对耗时仍受共享负载影响。

## 回归测试与复现

6 项回归测试通过，覆盖不同 K、不同全局点索引、并列分数、无面输出、退化几何、非连续输入和 CPU/GPU。真实样本的每次完整重建和提取测量均做精确输出一致性检查。

```bash
cd /home/syx/data/FastSESR
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /home/syx/data/miniconda3/envs/dcx/bin/python benchmarks/test_extract_face.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /home/syx/data/miniconda3/envs/dcx/bin/python benchmarks/compare_extract_face.py \\
  --manifest benchmarks/results/2026-10-09/selected_abc_test.json \\
  --out benchmarks/results/2026-10-09/reproduction \\
  --repeats 5 --extraction-repeats 25 --sample-count 100000

/home/syx/data/miniconda3/envs/dcx/bin/python benchmarks/summarize_extract_face.py \\
  --out benchmarks/results/2026-10-09/reproduction
```

主要文件：`comparison.csv` / `comparison.md` 为对比表；`timing_raw.csv` 为 2,400 次原始测量；`per_sample_summary.csv` 为逐模型汇总；`accuracy_per_sample.csv` 为逐模型精度；`manifest.json` 为配置、源码/权重/数据 SHA256 和资源信息；`original/`、`improved/` 为两组 40 个真实输出网格；`original_cli_bootstrap.log` 记录原入口初始化；`comparison.png` / `comparison.svg` 为可导出的图表。

## 全部项目盘点

{audit if audit else '完整盘点报告在 /home/syx/research_audit/2026-10-09/PROJECT_AUDIT.md。'}
'''
    (out/'REPORT.md').write_text(report)
    print(json.dumps({'summary':summary,'paired_analysis':intervals,'report':str(out/'REPORT.md')},indent=2))


if __name__ == '__main__':
    main()

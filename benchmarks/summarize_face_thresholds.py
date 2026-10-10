"""Generate an evidence-grounded report and plots for the threshold experiment."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    summary = json.loads((out/'summary.json').read_text())
    selected = json.loads((out/'selected_thresholds.json').read_text())
    protocol = json.loads((out/'protocol.json').read_text())
    paired = json.loads((out/'paired_analysis.json').read_text())
    development = json.loads((out/'development_summary.json').read_text())
    a, b = summary['baseline'], summary['selected']
    lines = ['| 指标 | 原阈值 | 开发集选择的阈值 | 变化 |', '|---|---:|---:|---:|']
    labels = {'CD1_x100':'CD1 ×100 ↓', 'CD2_x1e5':'CD2 ×10⁵ ↓', 'Precision':'Precision ↑',
              'Recall':'Recall ↑', 'F1_at_0_003':'F1 @0.003 ↑', 'NC':'法线一致性 ↑',
              'triangles':'平均三角面数', 'nonmanifold_edges':'平均非流形边数 ↓',
              'nonmanifold_edge_fraction':'平均非流形边比例 ↓', 'boundary_edges':'平均边界边数',
              'boundary_edge_fraction':'平均边界边比例', 'mean_sample_median_seconds':'重建时间（秒）',
              'mean_extra_peak_mib':'额外峰值显存（MiB）'}
    for key, label in labels.items():
        if key in ['Precision','Recall','F1_at_0_003','NC','nonmanifold_edge_fraction','boundary_edge_fraction']:
            av, bv = f'{a[key]*100:.5f}%', f'{b[key]*100:.5f}%'
            delta = f'{(b[key]-a[key])*100:+.5f} 个百分点'
        else:
            av, bv, delta = f'{a[key]:.6f}', f'{b[key]:.6f}', f'{b[key]-a[key]:+.6f}'
        lines.append(f'| {label} | {av} | {bv} | {delta} |')
    table = '\n'.join(lines)
    (out/'comparison.md').write_text(table+'\n')

    test_rows = list(csv.DictReader((out/'test_per_sample.csv').open()))
    names = [s['name'] for s in protocol['test_samples']]
    by_pair = {(r['sample'],r['method']):r for r in test_rows}
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    heat = np.array([[development[f't1_{t1:g}_t2_{t2:g}_angle_120']['F1_at_0_003']*100
                      for t2 in [.4,.5,.6]] for t1 in [.7,.8,.9]])
    im = axes[0,0].imshow(heat,cmap='viridis',aspect='auto')
    axes[0,0].set_xticks(range(3),['0.4','0.5','0.6'])
    axes[0,0].set_yticks(range(3),['0.7','0.8','0.9'])
    axes[0,0].set_xlabel('Top-2 confidence threshold')
    axes[0,0].set_ylabel('Top-1 confidence threshold')
    axes[0,0].set_title('Development F1 (%) at angle=120')
    for i in range(3):
        for j in range(3):
            axes[0,0].text(j,i,f'{heat[i,j]:.3f}',ha='center',va='center',color='white')
    fig.colorbar(im,ax=axes[0,0])
    xs = np.array([float(by_pair[(n,'baseline')]['F1_at_0_003'])*100 for n in names])
    ys = np.array([float(by_pair[(n,'selected')]['F1_at_0_003'])*100 for n in names])
    axes[0,1].scatter(xs,ys,s=23,alpha=.8)
    low,high = min(xs.min(),ys.min())-.2,max(xs.max(),ys.max())+.2
    axes[0,1].plot([low,high],[low,high],color='gray',linestyle='--')
    axes[0,1].set_xlabel('Baseline test F1 (%)')
    axes[0,1].set_ylabel('Selected test F1 (%)')
    axes[0,1].set_title('40 paired test shapes; diagonal = unchanged')
    for ax,key,title in [(axes[1,0],'nonmanifold_edges','Mean non-manifold edges (lower is better)'),
                         (axes[1,1],'boundary_edges','Mean boundary edges (open/defect boundary)')]:
        values = [a[key],b[key]]
        bars=ax.bar(['Baseline','Selected'],values,color=['#64748b','#0d9488'])
        ax.set_title(title)
        ax.set_ylim(0,max(values)*1.2+1)
        ax.bar_label(bars,fmt='%.2f',padding=4)
    fig.suptitle('FastSESR: validation-selected face thresholds',fontsize=15)
    fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=170)
    fig.savefig(out/'comparison.svg')
    plt.close(fig)

    f1_delta=(b['F1_at_0_003']-a['F1_at_0_003'])*100
    lo,hi=np.array(paired['F1_at_0_003']['percentile_95_interval'])*100
    nm_reduction=100*(1-b['nonmanifold_edges']/a['nonmanifold_edges']) if a['nonmanifold_edges'] else 0.
    cd_change=100*(b['CD1_x100']/a['CD1_x100']-1)
    c=selected['selected']
    group=lambda name:name.split('_trimesh_')[0].split('_',1)[1]
    dev_groups={group(s['name']) for s in protocol['development_samples']}
    test_groups={group(s['name']) for s in protocol['test_samples']}
    assert not dev_groups&test_groups
    f1_message = ('配对区间包含零，不能据此声称 F1 有稳定提升。' if lo<=0<=hi else
                  '配对区间不包含零；它只反映本组样本的重采样变化，不能代替完整数据集结论。')
    report=f'''# FastSESR 三角面筛选阈值实验

本轮实现方向 1 的第一步：修正阈值参数未生效的问题，并用独立开发子集选择固定阈值。未实现逐点自适应阈值，也未重新训练网络。之前的按需构建三角面优化保留。

## 实际结果

原方案为 Top-1=0.8、Top-2=0.5、angle=120°。开发子集选出 Top-1={c['confidence_top1']}、Top-2={c['confidence_top2']}、angle={c['face_angle']}°，在 40 个固定测试样本上的结果如下。所有质量数值都在本轮用相同指标及配对随机种子重新计算，不能直接混用上次实验的随机采样数值。

{table}

测试 F1 变化为 {f1_delta:+.5f} 个百分点，按形状配对重采样 10000 次的 95% 百分位区间为 [{lo:+.5f}, {hi:+.5f}] 个百分点。{f1_message}

平均非流形边数减少 {nm_reduction:.2f}%，CD1 的相对变化为 {cd_change:+.4f}%（负值为改善）。需要同时看边界边：删除冲突面可能留下缺口，本次方案不保证网格封闭、无自交或整体拓扑正确。边界边数不能直接解释为孔洞个数。

## 为什么修改

`S2/ExtractFace.py` 的构造器原先接收两个 confidence 参数，但 forward 内部直接比较 0.8 和 0.5，传入新阈值没有效果。修正为读取实例参数，加入范围检查，并在 `S2_reconstruct.py` 接入 `--confidence_top1`、`--confidence_top2`、`--face_angle`。LOON 和 LOON-UNet 分支均接线，其他分支显式拒绝不支持的自定义值。推理输出保存 `face_extraction_config.json`。

默认阈值仍为 0.8/0.5/120°，避免把小样本实验结论自动变成所有数据集的默认值。默认方案在全部 40 个测试样本上的三角面索引与上次原始实现输出一致。顶点、预训练权重、邻域与神经网络输出均保持相同；变化只来自筛选规则。

提高 Top-2 阈值要求第二个候选也有较高置信度，可能减少不可靠连接，但也可能丢失应保留的面。它是筛选强度的调整，不是置信度校准或新的网络结构。

## 预先固定的方案

- 数据：服务器现有 ABC 点云及对应 GT。16 个开发样本，40 个测试样本；两者 CAD 文档哈希无交集。测试样本涉及 {len(test_groups)} 个 CAD 文档。
- 两组都取自官方 ABC_test，本轮的 16 个样本用于开发，不能再计入独立测试成绩。40 个测试样本此前用于输出完全一致的速度实验，本轮不使用它们的质量结果选阈值。
- 开发样本按点数分布预先选取。比较 13 组参数：Top-1∈{{0.7,0.8,0.9}} × Top-2∈{{0.4,0.5,0.6}}，angle=120°；另比较原置信度下 angle∈{{90,105,135,150}}。
- 选择目标：平均开发 F1 最大，同时 Precision 不低于基线 0.002，CD1 不高于基线的 1.01 倍，非流形边比例不高于基线 0.001。若没有更好的可行方案，则保留基线。
- 参数在测试前写入 `selected_thresholds.json`，随后不再依据测试结果改动。
- 开发指标每个网格表面采样 20000 点，正式测试每个网格表面采样 100000 点；开发较低采样量用于降低搜索开销，其指标值不能直接与测试指标对比，阈值搜索的小差异也可能受采样波动影响。
- 配对采样种子固定，CD/F1/NC 的定义与项目 `eval/surf_acc.py` 相同：各自包围盒对角线归一化，F1 距离阈值 0.003；新增 Precision、Recall 和索引边关联统计。
- 所有阈值共享同一组真实神经网络预测。重建计时每方案每形状 3 次，顺序交替，使用完整推理流程而非缓存 logits；汇总各形状中位数的均值。计时排除输入读取、预处理、权重加载和写盘。
- 额外峰值显存是 PyTorch 本进程分配量相对测量起点的增量。速度差异不是本轮主要结论，共享服务器上的计时会受其他任务影响。

## 使用

在项目目录运行，替换列表或输出路径即可：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
/home/syx/data/miniconda3/envs/dcx/bin/python S2_reconstruct.py \\
  --use_loon_unet --loon_unet_ckpt runs/S2_train/model_best.pth \\
  --dataset ABC_test --data_root Data \\
  --test_list benchmarks/results/2026-10-10/thresholds/test_list.txt \\
  --out_dir results/ABC_test_threshold_trial \\
  --confidence_top1 {c['confidence_top1']} --confidence_top2 {c['confidence_top2']} --face_angle {c['face_angle']}
```

## 复现与证据

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
/home/syx/data/miniconda3/envs/dcx/bin/python benchmarks/compare_face_thresholds.py \\
  --protocol benchmarks/results/2026-10-10/thresholds/protocol.json \\
  --out benchmarks/results/2026-10-10/thresholds_reproduction

/home/syx/data/miniconda3/envs/dcx/bin/python benchmarks/summarize_face_thresholds.py \\
  --out benchmarks/results/2026-10-10/thresholds_reproduction
```

- `protocol.json`：搜索之前固定的数据与规则。
- `manifest.json`：源码、权重与输入哈希、模型配置、默认行为校验。
- `source_before/`：本轮修改前的源码快照。
- `development_per_sample.csv`、`development_comparison.csv`：16×13 组开发结果。
- `selected_thresholds.json`：测试之前固定的参数。
- `test_per_sample.csv`：40×2 组正式质量与拓扑结果。
- `timing_raw.csv`：40×2×3 组完整推理计时。
- `paired_analysis.json`：各指标按形状配对的差异与重采样区间。40 个形状来自 39 个 CAD 文档，少量同文档相关性及固定样本选择限制统计解释。
- `baseline/`、`selected/`：本轮的两组输出网格。
- `comparison.csv`、`comparison.md`、`comparison.png`、`comparison.svg`：表格和图。

## 结论边界

本轮能回答简单阈值调整对这组 ABC 样本的影响，不能证明所有数据集都受益，也不足以构成独立的方法创新。若要继续研究逐点自适应阈值，应在新的开发条件中引入密度或几何特征，并与本轮固定阈值方案比较；不能继续用这 40 个测试样本反复选参数。
'''
    (out/'REPORT.md').write_text(report)
    (out/'test_list.txt').write_text('\n'.join(s['name'] for s in protocol['test_samples'])+'\n')
    print(table)


if __name__ == '__main__':
    main()

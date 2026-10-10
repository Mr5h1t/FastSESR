"""Summarize the CD-driven adaptive filtering experiment without overclaiming."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    p=json.loads((out/'protocol.json').read_text())
    selection=json.loads((out/'selected_config.json').read_text())
    s=json.loads((out/'summary.json').read_text())
    paired=json.loads((out/'paired_analysis.json').read_text())
    development=json.loads((out/'development_summary.json').read_text())
    c=selection['selected']
    methods=['baseline','strict_0.6','adaptive_selected']
    labels={'CD1_x100':'采样点 CD1 ×100 ↓','CD2_x1e5':'采样点 CD2 ×10⁵ ↓',
            'SurfaceCD1_x100':'点到连续表面 CD1 ×100 ↓','SurfaceCD2_x1e5':'点到连续表面 CD2 ×10⁵ ↓',
            'Precision':'Precision ↑','Recall':'Recall ↑','F1_at_0_003':'F1 @0.003 ↑','NC':'法线一致性 ↑',
            'triangles':'平均三角面数','nonmanifold_edges':'平均非流形边数 ↓',
            'boundary_edges':'平均边界边数','seconds':'完整重建时间（秒）','extra_peak_mib':'额外峰值显存（MiB）'}
    table=['| 指标 | 原方案 0.8/0.5 | 上轮方案 0.8/0.6 | 本轮选定方案 |','|---|---:|---:|---:|']
    for key,label in labels.items():
        values=[f'{s[m][key]*100:.5f}%' if key in ['Precision','Recall','F1_at_0_003','NC'] else f'{s[m][key]:.6f}' for m in methods]
        table.append('| '+label+' | '+' | '.join(values)+' |')
    table='\n'.join(table)
    (out/'comparison.md').write_text(table+'\n')
    a,b=s['baseline'],s['adaptive_selected']
    cd_change=100*(b['CD1_x100']/a['CD1_x100']-1)
    surface_change=100*(b['SurfaceCD1_x100']/a['SurfaceCD1_x100']-1)
    lo,hi=paired['adaptive_selected']['CD1_x100']['percentile_95_interval']
    interpretation=('差异区间包含零，不能声称稳定改善 CD。' if lo<=0<=hi else
                    '差异区间不包含零，需结合差异方向和连续表面指标判断；它仍只描述这 40 个固定样本。')
    selected_mode=('几何自适应模式' if c['quality_weight'] else
                   ('原固定阈值' if c['confidence_top2']==.5 else '上轮固定阈值'))
    report=f'''# FastSESR 几何质量自适应筛选：以 CD 为目标的实验

日期：2026-10-10（Asia/Shanghai）。本轮新增可选的几何自适应 Top-2 阈值，并改变开发集选参目标，以检验 CD 是否改善。未重新训练网络；此前性能优化和原默认输出保留。

## 本轮结果

开发集选定 **{selected_mode}**：confidence_top2={c['confidence_top2']}，quality_weight={c['quality_weight']}，quality_target={c['quality_target']}，Top-1=0.8，angle=120°。参数固定后，对按 CAD 文档名排除旧样本的 40 个文件进行了对比。文件名不同不保证几何独立；后续字节哈希复核发现重复文件，见 `verification.json`。这些均值保留为原始 40 文件的历史记录，不应视为 40 个完全独立的测试形状。下面的三个方案都在这组新数据、相同采样种子下重新计算，不能直接与上轮不同样本上的均值相减。

{table}

本轮选定方案相对原方案，采样点 CD1 的相对变化为 **{cd_change:+.5f}%**（负值为改善），连续表面 CD1 的相对变化为 **{surface_change:+.5f}%**。采样点 CD1 的形状配对平均差异 95% 百分位区间为 [{lo:+.8f}, {hi:+.8f}]。{interpretation}

若开发集最终选回固定阈值，表示本轮搜索范围内的新规则未显示更好的开发表现，不应把新增接口宣称为已验证的质量提升。即使 CD 改善，也要同时看 Precision、Recall、非流形边和边界边，避免用错误填面换取距离下降。

## 为什么不继续统一提高阈值

检查上轮 CD1 变差最多的三个样本，删除三角面的中心到 GT 连续表面的中位距离接近零（见 `diagnosis_previous_thresholds.json`）。这是“低置信度不必然等于错误几何”的证据，但只检查了面中心，不能证明整个面都正确，也不能完全分离旧 CD 差异中的随机采样波动。

提高 Top-2 阈值会同时删掉冲突面和有用面。新规则尝试利用输入几何区分筛选强度，推理中不读取 GT。

## 实现

对候选三角面计算无量纲形状质量：

`q = 4 * sqrt(3) * area / (edge_a² + edge_b² + edge_c²)`。

q 接近 1 代表接近等边三角形，接近 0 代表细长或退化的三角形。使用候选对中较低的质量值，调整第二个候选的阈值：

`threshold_2 = clamp(base_threshold_2 + quality_weight * (quality_target - pair_quality), 0, 1)`。

质量较高时放宽阈值，质量较低时收紧阈值。Top-1 条件、角度条件、原始点坐标和网络预测保持相同。q 只表示面形状，不表示它一定在正确表面上，也无法保证薄壁不被跨接。

新增参数 `--face_quality_weight` 和 `--face_quality_target`。weight 默认 0，此时沿用原固定阈值，跳过额外几何计算。保持默认比自动启用一个小样本选出的规则更合适。新参数范围为 [0,1]，LOON/LOON-UNet 分支接线并保存到输出配置。

对照脚本增加了旧提取器接口适配，只允许质量权重为零；原始提取器文件未修改。

## 固定的数据与选参规则

- 复用上轮的 16 个开发样本，新增 40 个测试样本。新测试样本分别来自 40 个 CAD 文档，排除以前开发和测试中出现的所有 CAD 文档。
- 搜索 11 个方案：原 0.8/0.5、上轮 0.8/0.6，以及 quality_weight∈{{0.1,0.2,0.4}} × quality_target∈{{0.1,0.2,0.4}} 的 9 组自适应规则。
- 开发目标为 **最小化采样点 CD1**。每个网格表面每次采样 100000 点，使用 3 个采样种子取均值；正式测试同样使用 3 个种子，不再使用上轮低分辨率开发指标。
- 约束：开发 F1 和 Precision 不低于基线 0.0005，边界边比例不高于基线 0.0002，连续表面 CD1 不高于基线的 1.01 倍加 1e-7。原方案参与选择，平局优先原方案。
- `selected_config.json` 在任何新测试样本评估前写入。本轮不根据测试结果调整参数。
- 输入点云、邻域和神经网络预测在开发比较中共享。所有活动参数来自既有预训练权重，官方 CLI 初始化后校验。
- 每种方案、每个测试形状执行 3 次完整推理计时，交替顺序，以各形状中位数的均值汇总。排除读盘、预处理、加载权重和写盘。
- 配对区间对 40 个形状的种子均值重采样 10000 次。它反映固定样本组的变化，不能当作完整 ABC 数据集的结论。

## 两种 CD 的含义

“采样点 CD”保留项目 `eval/surf_acc.py` 的定义：GT 和预测网格分别按包围盒对角线归一化，采样点之间用最近邻距离计算双向均值。不同采样种子的波动会影响小差异。

“点到连续表面 CD”是额外诊断，使用 20000 个表面查询点到目标三角网格的最近距离（Open3D RaycastingScene），而不是到目标的有限采样点。它减少了目标侧采样间距造成的距离下限，但仍是有限查询的近似，不是精确曲面积分。它也不能与项目原 CD 数值混用。

边界边统计包括真实开放边界和缺陷边界，不等于孔洞数量；非流形边数减少也不保证无自交或顶点流形。

## 使用与复现

项目根目录、已配置依赖及 ABC 数据时，可以给原重建命令追加：

```bash
--confidence_top1 0.8 --confidence_top2 {c['confidence_top2']} --face_angle 120 \\
--face_quality_weight {c['quality_weight']} --face_quality_target {c['quality_target']}
```

复现实验：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
python benchmarks/compare_adaptive_quality.py \\
  --protocol benchmarks/results/2026-10-10/adaptive_quality/protocol.json \\
  --out benchmarks/results/2026-10-10/adaptive_quality_reproduction

python benchmarks/summarize_adaptive_quality.py \\
  --out benchmarks/results/2026-10-10/adaptive_quality_reproduction
```

服务器使用 `/home/syx/data/miniconda3/envs/dcx/bin/python`。换机器时，在 protocol 的副本中修改本地点云和 GT 路径，保留原 protocol 与 manifest 作为历史证据。

## 文件

- `protocol.json`：搜索开始前确定的划分、参数网格和选择规则。
- `development_raw.csv`：16×11×3=528 行开发指标。
- `development_comparison.csv`、`selected_config.json`：开发均值和测试前固定的方案。
- `test_raw.csv`：40×3×3=360 行正式指标；连续表面指标每个形状每个方案计算一次，复用在三个采样种子行中。
- `test_per_shape.csv`：120 行形状均值，避免把采样种子当成独立形状。
- `timing_raw.csv`：360 次真实完整推理。
- `manifest.json`：源码、权重、输入哈希及模型配置。
- `paired_analysis.json`、`comparison.csv`、`comparison.md`、`comparison.png`、`comparison.svg`：配对统计、对比表和图。
- `baseline/`、`strict_0.6/`、`adaptive_selected/`：三组输出网格，留在服务器，不上传数据集或新权重。
'''
    (out/'REPORT.md').write_text(report)
    (out/'test_list.txt').write_text('\n'.join(x['name'] for x in p['test_samples'])+'\n')
    rows=list(csv.DictReader((out/'test_per_shape.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    names=['Original','Fixed 0.6','Selected']
    for ax,key,title in [(axes[0,0],'CD1_x100','Sampled-point CD1 x100 (3-seed mean)'),
                         (axes[0,1],'SurfaceCD1_x100','Point-to-surface CD1 x100'),
                         (axes[1,0],'nonmanifold_edges','Mean non-manifold edges'),
                         (axes[1,1],'boundary_edges','Mean boundary edges')]:
        values=[s[m][key] for m in methods]
        bars=ax.bar(names,values,color=['#64748b','#d97706','#0d9488'])
        fmt='%.6f' if 'CD' in key else '%.2f'
        ax.bar_label(bars,fmt=fmt,padding=3)
        ax.set_ylim(0,max(values)*1.2+1e-12)
        ax.set_title(title)
    fig.suptitle('FastSESR: geometry-adaptive filtering, 40 fresh test shapes')
    fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=170)
    fig.savefig(out/'comparison.svg')
    plt.close(fig)
    print(table)
    print('Selected mode:',selected_mode,c)


if __name__=='__main__':
    main()

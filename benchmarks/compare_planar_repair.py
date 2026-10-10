"""Evaluate conservative planar-overlap repair with fixed parameters."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'benchmarks'))
import numpy as np
import open3d as o3d
import torch
import shapely
from torch.utils.data import DataLoader
from dataset.pc_recon import PCReconSet
from S2.PlanarRepair import repair_planar_overlaps
from compare_extract_face import load_original_models,reconstruct,measure,seed_all,sha256,gpu_snapshot
from compare_adaptive_quality import shared_predictions,quality_for_shape,mean_metrics,KEYS,cad,extractor
from compare_face_thresholds import write_csv,mesh_from_arrays


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--chunk-size',type=int,default=2000)
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    p=json.loads(args.protocol.read_text())
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;seed_all(p['seed'])
    assert not {cad(s['name']) for s in p['development_samples']}&{cad(s['name']) for s in p['test_samples']}
    assert len({cad(s['name']) for s in p['test_samples']})==40
    print('INITIALIZE original pretrained models',flush=True)
    model,unet,config,bootstrap=load_original_models(args,out,p['development_samples'])
    manifest=dict(protocol_sha256=sha256(args.protocol),protocol=p,config=config,
                  sources={rel:sha256(ROOT/rel) for rel in ['S2/ExtractFace.py','S2/PlanarRepair.py','S2_reconstruct.py','benchmarks/compare_planar_repair.py','benchmarks/compare_adaptive_quality.py','benchmarks/compare_extract_face.py','benchmarks/compare_face_thresholds.py']},
                  weights={rel:sha256(ROOT/rel) for rel in ['trained_models/model_knn50.pth','runs/S2_train/model_best.pth']},
                  shapely=shapely.__version__,gpu_before=gpu_snapshot())
    manifest['input_sha256']={s['name']:dict(pointcloud=sha256(s['pointcloud']),ground_truth=sha256(s['ground_truth'])) for s in p['development_samples']+p['test_samples']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    neural_configs=[dict(name='baseline',quality_weight=0.,quality_target=.2,confidence_top2=.5),
                    dict(name='strict_0.6',quality_weight=0.,quality_target=.2,confidence_top2=.6)]
    repairs=[];timings=[];all_rows={}
    for stage in ['development','test']:
        samples=p[f'{stage}_samples']
        data_set=PCReconSet([s['pointcloud'] for s in samples],knn=50,delta=config['delta'],rescale_delta=config['rescale_delta'])
        rows=[]
        for i,batch in enumerate(DataLoader(data_set,batch_size=1,shuffle=False,num_workers=0)):
            data=[x[0] for x in batch]
            vertices,faces=shared_predictions(data,model,unet,args.chunk_size,neural_configs)
            repaired,stats=repair_planar_overlaps(vertices,faces['baseline'],p['plane_tolerance'])
            faces['planar_repaired']=repaired
            repairs.append(dict(stage=stage,sample=samples[i]['name'],**stats))
            if stage=='development' and i==0:
                assert np.array_equal(vertices,bootstrap[0]) and np.array_equal(faces['baseline'],bootstrap[1])
                manifest['default_matches_original_cli']=True
            spec={**p,'sampling_repeats':3}
            rows.extend(quality_for_shape(samples[i],vertices,faces,p['configs'],spec,p['seed']+(10000 if stage=='test' else 0)+2*i))
            write_csv(out/f'{stage}_raw.csv',rows)
            write_csv(out/'repair_per_shape.csv',repairs)
            if stage=='test':
                for name in faces:
                    (out/name).mkdir(exist_ok=True)
                    assert o3d.io.write_triangle_mesh(str(out/name/samples[i]['name']),mesh_from_arrays(vertices,faces[name]))
                def full(method):
                    cfg=neural_configs[1] if method=='strict_0.6' else neural_configs[0]
                    mesh=reconstruct(data,extractor(cfg),model,unet,args.chunk_size)
                    if method=='planar_repaired':
                        f,_=repair_planar_overlaps(np.asarray(mesh.vertices),np.asarray(mesh.triangles),p['plane_tolerance'])
                        mesh.triangles=o3d.utility.Vector3iVector(f)
                    return mesh
                for repeat in range(p['timing_repeats']):
                    order=list(faces) if (i+repeat)%2==0 else list(reversed(faces))
                    for method in order:
                        mesh,measurement=measure(lambda:full(method))
                        assert np.array_equal(np.asarray(mesh.triangles),faces[method])
                        timings.append(dict(sample=samples[i]['name'],method=method,repeat=repeat,**measurement))
                write_csv(out/'timing_raw.csv',timings)
            print(f'{stage.upper()} {i+1}/{len(samples)} repaired_patches={stats["repaired_patches"]}',flush=True)
        all_rows[stage]=rows
        summary={c['name']:mean_metrics([r for r in rows if r['config']==c['name']]) for c in p['configs']}
        (out/f'{stage}_summary.json').write_text(json.dumps(summary,indent=2))
        if stage=='development':
            (out/'fixed_before_test.json').write_text(json.dumps(dict(plane_tolerance=p['plane_tolerance'],test_samples_evaluated=0,development_summary=summary),indent=2))
            print('FIXED PARAMETERS; development comparison',json.dumps(summary),flush=True)
    per_shape=[]
    for sample in p['test_samples']:
        for c in p['configs']:
            method=c['name']
            row=mean_metrics([r for r in all_rows['test'] if r['sample']==sample['name'] and r['config']==method])
            row['seconds']=float(np.median([r['seconds'] for r in timings if r['sample']==sample['name'] and r['method']==method]))
            row['extra_peak_mib']=max(r['extra_peak_mib'] for r in timings if r['sample']==sample['name'] and r['method']==method)
            per_shape.append(dict(sample=sample['name'],method=method,**row))
    write_csv(out/'test_per_shape.csv',per_shape)
    summary={c['name']:{key:float(np.mean([r[key] for r in per_shape if r['method']==c['name']])) for key in (*KEYS,'seconds','extra_peak_mib')} for c in p['configs']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    write_csv(out/'comparison.csv',[dict(metric=key,**{m:s[key] for m,s in summary.items()}) for key in summary['baseline']])
    rng=np.random.default_rng(p['seed']);indices=rng.integers(0,40,size=(10000,40));paired={}
    for method in ['strict_0.6','planar_repaired']:
        paired[method]={}
        for key in KEYS:
            a=np.array([r[key] for r in per_shape if r['method']=='baseline'])
            b=np.array([r[key] for r in per_shape if r['method']==method]);d=b-a
            paired[method][key]=dict(mean_difference=float(d.mean()),percentile_95_interval=np.percentile(d[indices].mean(axis=1),[2.5,97.5]).tolist(),lower=int((d<0).sum()),higher=int((d>0).sum()),equal=int((d==0).sum()))
    (out/'paired_analysis.json').write_text(json.dumps(paired,indent=2))
    manifest.update(parameters_fixed_before_test=True,gpu_after=gpu_snapshot())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()

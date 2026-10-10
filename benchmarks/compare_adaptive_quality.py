"""CD-driven development of geometry-adaptive filtering with fresh evaluation."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'benchmarks'))
import numpy as np
import open3d as o3d
import torch
import trimesh
from torch.utils.data import DataLoader
from dataset.pc_recon import PCReconSet
from S2.ExtractFace import SurfExtract
from compare_extract_face import load_original_models, measure, reconstruct, seed_all, sha256, gpu_snapshot
from compare_face_thresholds import PairedEvaluator, write_csv, mesh_from_arrays, METRICS


def cad(name):
    return name.split('_trimesh_')[0].split('_', 1)[1]


def extractor(config):
    return SurfExtract(.8, config['confidence_top2'], 120,
                       config['quality_weight'], config['quality_target'])


@torch.no_grad()
def shared_predictions(data, model, unet, chunk_size, configs):
    points, indices, center, scale = data
    points = points.cuda()
    indices = indices.cuda()
    offsets = unet(points.t().unsqueeze(0).contiguous()).squeeze(0).t().contiguous()
    methods = {c['name']: extractor(c) for c in configs}
    faces = {key: [] for key in methods}
    for start in range(0, len(points), chunk_size):
        sl = slice(start, min(start+chunk_size, len(points)))
        features = points[indices[sl]]-points[sl,None,:]
        off = offsets[indices[sl]]-offsets[indices[sl,0],None,:]
        logits = model(features, off)
        for key, method in methods.items():
            selected = method(points, logits, indices[sl])
            if selected.numel():
                faces[key].append(selected)
    result = {}
    for key, chunks in faces.items():
        result[key] = (torch.unique(torch.sort(torch.cat(chunks), dim=-1).values,
                                   dim=0).cpu().numpy() if chunks else
                       np.empty((0,3), dtype=np.int64))
    vertices = (points.cpu()*scale+center).numpy()
    return vertices, result


def scene_for(mesh):
    vertices = o3d.core.Tensor(np.asarray(mesh.vertices).astype('float32'))
    faces = o3d.core.Tensor(np.asarray(mesh.faces).astype('uint32'))
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(vertices, faces)
    return scene


class SurfaceEvaluator:
    """Extra diagnostic: sampled queries to continuous triangle surfaces.

    This is a separate metric from the project's sampled-point CD. It removes
    nearest-sample spacing from the target side of each directed distance.
    It still uses finitely many queries, and is not an exact surface integral.
    """
    def __init__(self, evaluator, count):
        self.evaluator = evaluator
        self.count = count
        self.gt_points = evaluator.gt_points[:count].astype('float32')
        self.gt_scene = scene_for(evaluator.gt)

    def evaluate(self, vertices, faces):
        if not len(faces):
            return dict(SurfaceCD1_x100=float('inf'), SurfaceCD2_x1e5=float('inf'))
        pred = self.evaluator.normalizer.normalize_diagonal(trimesh.Trimesh(vertices=vertices, faces=faces))
        pred_points, _ = trimesh.sample.sample_surface(pred, self.count, seed=self.evaluator.seed+777)
        d_gt = scene_for(pred).compute_distance(o3d.core.Tensor(self.gt_points), nthreads=4).numpy()
        d_pred = self.gt_scene.compute_distance(o3d.core.Tensor(pred_points.astype('float32')), nthreads=4).numpy()
        return dict(SurfaceCD1_x100=float((d_gt.mean()+d_pred.mean())*100),
                    SurfaceCD2_x1e5=float(((d_gt*d_gt).mean()+(d_pred*d_pred).mean())*1e5))


KEYS = (*METRICS, 'SurfaceCD1_x100', 'SurfaceCD2_x1e5')


def mean_metrics(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in KEYS}


def choose(rows, configs):
    summary = {c['name']:dict(config=c, **mean_metrics([r for r in rows if r['config']==c['name']])) for c in configs}
    base = summary['baseline']
    eligible = []
    for name, stats in summary.items():
        ok = (stats['F1_at_0_003'] >= base['F1_at_0_003']-.0005 and
              stats['Precision'] >= base['Precision']-.0005 and
              stats['boundary_edge_fraction'] <= base['boundary_edge_fraction']+.0002 and
              stats['SurfaceCD1_x100'] <= base['SurfaceCD1_x100']*1.01+1e-7)
        stats['eligible'] = bool(ok)
        if ok:
            eligible.append(name)
    selected = min(eligible, key=lambda n:(summary[n]['CD1_x100'], n!='baseline'))
    return summary[selected]['config'], summary


def quality_for_shape(sample, vertices, faces, configs, protocol, shape_seed):
    rows = []
    evaluator = PairedEvaluator(sample['ground_truth'], protocol['sample_count'], shape_seed)
    continuous = SurfaceEvaluator(evaluator, protocol['surface_distance_sample_count'])
    surface = {c['name']:continuous.evaluate(vertices,faces[c['name']]) for c in configs}
    for sampling_repeat in range(protocol['sampling_repeats']):
        if sampling_repeat:
            evaluator = PairedEvaluator(sample['ground_truth'], protocol['sample_count'], shape_seed+100*sampling_repeat)
        for c in configs:
            name = c['name']
            metrics = evaluator.evaluate(vertices, faces[name])
            rows.append(dict(sample=sample['name'], config=name, sampling_repeat=sampling_repeat,
                             **metrics, **surface[name]))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=2000)
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    p=json.loads(args.protocol.read_text())
    dev,test=p['development_samples'],p['test_samples']
    assert not {cad(s['name']) for s in dev}&{cad(s['name']) for s in test}
    assert len({cad(s['name']) for s in test})==len(test)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    seed_all(p['seed'])
    print('INITIALIZE original pretrained models', flush=True)
    model,unet,config,bootstrap=load_original_models(args,out,dev)
    manifest=dict(protocol_sha256=sha256(args.protocol),config=config,protocol=p,
                  source_sha256={rel:sha256(ROOT/rel) for rel in ['S2/ExtractFace.py','S2_reconstruct.py','benchmarks/compare_adaptive_quality.py','benchmarks/compare_extract_face.py','benchmarks/compare_face_thresholds.py']},
                  weight_sha256={rel:sha256(ROOT/rel) for rel in ['trained_models/model_knn50.pth','runs/S2_train/model_best.pth']},
                  gpu_before=gpu_snapshot(),torch=torch.__version__,cuda=torch.version.cuda)
    manifest['input_sha256']={s['name']:dict(pointcloud=sha256(s['pointcloud']),ground_truth=sha256(s['ground_truth'])) for s in dev+test}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def loader(samples):
        dataset=PCReconSet([s['pointcloud'] for s in samples],knn=50,delta=config['delta'],rescale_delta=config['rescale_delta'])
        return DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    development=[]
    for i,batch in enumerate(loader(dev)):
        data=[x[0] for x in batch]
        vertices,faces=shared_predictions(data,model,unet,args.chunk_size,p['configs'])
        if i==0:
            assert np.array_equal(vertices,bootstrap[0])
            assert np.array_equal(faces['baseline'],bootstrap[1])
            manifest['default_matches_original_cli']=True
        spec={**p,'sampling_repeats':p['development_sampling_seeds']}
        development.extend(quality_for_shape(dev[i],vertices,faces,p['configs'],spec,p['seed']+2*i))
        write_csv(out/'development_raw.csv',development)
        print(f'DEVELOPMENT {i+1}/{len(dev)}',flush=True)
    selected,stats=choose(development,p['configs'])
    (out/'development_summary.json').write_text(json.dumps(stats,indent=2))
    write_csv(out/'development_comparison.csv',[dict(name=n,**{k:v for k,v in s.items() if k!='config'}) for n,s in stats.items()])
    selection=dict(selected=selected,selection_rule=p['selection_rule'],test_samples_evaluated=0)
    (out/'selected_config.json').write_text(json.dumps(selection,indent=2))
    print('FROZEN SELECTION',json.dumps(selection),flush=True)
    methods={'baseline':p['configs'][0],'strict_0.6':p['configs'][1],'adaptive_selected':selected}
    test_configs=[dict(c,name=method) for method,c in methods.items()]
    for method in methods:
        (out/method).mkdir(exist_ok=True)
    results,times=[],[]
    for i,batch in enumerate(loader(test)):
        data=[x[0] for x in batch]
        vertices,faces=shared_predictions(data,model,unet,args.chunk_size,test_configs)
        spec={**p,'sampling_repeats':p['test_sampling_seeds']}
        results.extend(quality_for_shape(test[i],vertices,faces,test_configs,spec,p['seed']+10000+2*i))
        for method in methods:
            mesh=mesh_from_arrays(vertices,faces[method])
            assert o3d.io.write_triangle_mesh(str(out/method/test[i]['name']),mesh)
        for repeat in range(p['timing_repeats']):
            order=list(methods) if (i+repeat)%2==0 else list(reversed(methods))
            for method in order:
                mesh,timing=measure(lambda:reconstruct(data,extractor(methods[method]),model,unet,args.chunk_size))
                assert np.array_equal(np.asarray(mesh.triangles),faces[method])
                times.append(dict(sample=test[i]['name'],method=method,repeat=repeat,**timing))
        write_csv(out/'test_raw.csv',results)
        write_csv(out/'timing_raw.csv',times)
        print(f'TEST {i+1}/{len(test)}',flush=True)
    per_shape=[]
    for sample in test:
        for method in methods:
            rows=[r for r in results if r['sample']==sample['name'] and r['config']==method]
            m=mean_metrics(rows)
            m['seconds']=float(np.median([r['seconds'] for r in times if r['sample']==sample['name'] and r['method']==method]))
            m['extra_peak_mib']=max(r['extra_peak_mib'] for r in times if r['sample']==sample['name'] and r['method']==method)
            per_shape.append(dict(sample=sample['name'],method=method,**m))
    write_csv(out/'test_per_shape.csv',per_shape)
    summary={method:{key:float(np.mean([r[key] for r in per_shape if r['method']==method])) for key in (*KEYS,'seconds','extra_peak_mib')} for method in methods}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    rng=np.random.default_rng(p['seed'])
    indices=rng.integers(0,len(test),size=(10000,len(test)))
    paired={}
    for method in ['strict_0.6','adaptive_selected']:
        paired[method]={}
        for key in KEYS:
            a=np.array([r[key] for r in per_shape if r['method']=='baseline'])
            b=np.array([r[key] for r in per_shape if r['method']==method])
            d=b-a
            paired[method][key]=dict(mean_difference=float(d.mean()),percentile_95_interval=np.percentile(d[indices].mean(axis=1),[2.5,97.5]).tolist(),lower=int((d<0).sum()),higher=int((d>0).sum()),equal=int((d==0).sum()))
    (out/'paired_analysis.json').write_text(json.dumps(paired,indent=2))
    write_csv(out/'comparison.csv',[dict(metric=key,**{m:summary[m][key] for m in methods}) for key in summary['baseline']])
    manifest.update(selected=selected,selection_frozen_before_test=True,gpu_after=gpu_snapshot())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()

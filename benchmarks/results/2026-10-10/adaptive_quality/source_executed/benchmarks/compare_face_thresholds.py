"""Development-only threshold selection followed by frozen paired evaluation.

Use the original pretrained CLI to initialize the models. All threshold variants
share exactly the same logits, point positions, and neighbor indices. Quality
metrics follow eval/surf_acc.py, with fixed paired sampling seeds. Timing runs
execute the complete stock LOON-UNet reconstruction helper, not cached logits.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))
import numpy as np
import open3d as o3d
import torch
import trimesh
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from dataset.pc_recon import PCReconSet
from S2.ExtractFace import SurfExtract
from eval.surf_acc import MeshEvaluator
from compare_extract_face import (load_original_models, measure, reconstruct,
                                 seed_all, sha256, gpu_snapshot)


def identifier(config):
    return 't1_%g_t2_%g_angle_%g' % (config['confidence_top1'],
                                    config['confidence_top2'], config['face_angle'])


def extractor(config):
    return SurfExtract(config['confidence_top1'], config['confidence_top2'], config['face_angle'])


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mesh_from_arrays(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


@torch.no_grad()
def shared_predictions(data, model, unet, chunk_size, configs):
    """Evaluate all settings on one neural prediction; no training takes place."""
    points, indices, center, scale = data
    points_dev = points.cuda()
    offsets = unet(points_dev.t().unsqueeze(0).contiguous()).squeeze(0).t().contiguous()
    indices_dev = indices.cuda()
    methods = {identifier(c): extractor(c) for c in configs}
    faces = {key: [] for key in methods}
    for start in range(0, len(points), chunk_size):
        sl = slice(start, min(start + chunk_size, len(points)))
        features = points_dev[indices_dev[sl]] - points_dev[sl, None, :]
        off = offsets[indices_dev[sl]] - offsets[indices_dev[sl, 0], None, :]
        logits = model(features, off)
        for key, method in methods.items():
            selected = method(points_dev, logits, indices_dev[sl])
            if selected.numel():
                faces[key].append(selected)
    arrays = {}
    for key, chunks in faces.items():
        arrays[key] = (torch.unique(torch.sort(torch.cat(chunks), dim=-1).values,
                                    dim=0).cpu().numpy() if chunks else
                       np.empty((0, 3), dtype=np.int64))
    vertices = (points_dev.cpu() * scale + center).numpy()
    return vertices, arrays


def topology(vertices, faces):
    """Count indexed undirected edge incidence; no repair or welding."""
    if not len(faces):
        return dict(triangles=0, edges=0, boundary_edges=0, boundary_edge_fraction=0.,
                    nonmanifold_edges=0, nonmanifold_edge_fraction=0., degenerate_faces=0)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                     vertices[faces[:, 2]] - vertices[faces[:, 0]])
    return dict(triangles=len(faces), edges=len(unique), boundary_edges=int((counts == 1).sum()),
                boundary_edge_fraction=float((counts == 1).mean()),
                nonmanifold_edges=int((counts > 2).sum()),
                nonmanifold_edge_fraction=float((counts > 2).mean()),
                degenerate_faces=int((np.linalg.norm(cross, axis=1) == 0).sum()))


class PairedEvaluator:
    def __init__(self, gt_path, sample_count, seed):
        self.normalizer = MeshEvaluator(sample_num=sample_count)
        self.gt = self.normalizer.normalize_diagonal(trimesh.load_mesh(gt_path))
        self.gt_points, gt_faces = trimesh.sample.sample_surface(self.gt, sample_count, seed=seed)
        self.gt_normals = self.gt.face_normals[gt_faces]
        self.gt_tree = KDTree(self.gt_points)
        self.count, self.seed = sample_count, seed

    def evaluate(self, vertices, faces):
        stats = topology(vertices, faces)
        if not len(faces):
            return dict(CD1_x100=float('inf'), CD2_x1e5=float('inf'), Precision=0.,
                        Recall=0., F1_at_0_003=0., NC=0., **stats)
        # Preserve the project's independent bbox-diagonal normalization.
        pred = self.normalizer.normalize_diagonal(trimesh.Trimesh(vertices=vertices, faces=faces))
        pred_points, pred_faces = trimesh.sample.sample_surface(pred, self.count, seed=self.seed + 1)
        pred_normals = pred.face_normals[pred_faces]
        d_gt, i_gt = KDTree(pred_points).query(self.gt_points, k=1)
        d_pred, i_pred = self.gt_tree.query(pred_points, k=1)
        precision, recall = float((d_pred < .003).mean()), float((d_gt < .003).mean())
        nc_gt = np.abs(np.sum(self.gt_normals * pred_normals[i_gt[:, 0]], axis=1)).mean()
        nc_pred = np.abs(np.sum(pred_normals * self.gt_normals[i_pred[:, 0]], axis=1)).mean()
        return dict(CD1_x100=float((d_gt.mean() + d_pred.mean()) * 100),
                    CD2_x1e5=float(((d_gt*d_gt).mean() + (d_pred*d_pred).mean()) * 1e5),
                    Precision=precision, Recall=recall,
                    F1_at_0_003=2*precision*recall/(precision+recall) if precision+recall else 0.,
                    NC=float((nc_gt + nc_pred)/2), **stats)


METRICS = ('CD1_x100', 'CD2_x1e5', 'Precision', 'Recall', 'F1_at_0_003', 'NC',
           'triangles', 'boundary_edges', 'boundary_edge_fraction',
           'nonmanifold_edges', 'nonmanifold_edge_fraction', 'degenerate_faces')


def aggregate(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in METRICS}


def choose(rows, configs, baseline):
    summary = {identifier(c): dict(config=c, **aggregate([r for r in rows if r['config'] == identifier(c)]))
               for c in configs}
    base = summary[identifier(baseline)]
    eligible = []
    for key, stats in summary.items():
        ok = (stats['Precision'] >= base['Precision'] - .002 and
              stats['CD1_x100'] <= base['CD1_x100'] * 1.01 and
              stats['nonmanifold_edge_fraction'] <= base['nonmanifold_edge_fraction'] + .001)
        stats['eligible'] = bool(ok)
        if ok:
            eligible.append(key)
    def rank(key):
        c = summary[key]['config']
        distance = (abs(c['confidence_top1']-.8) + abs(c['confidence_top2']-.5) +
                    abs(c['face_angle']-120)/180)
        return summary[key]['F1_at_0_003'], key == identifier(baseline), -distance
    selected_key = max(eligible, key=rank)
    return summary[selected_key]['config'], summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=2000)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    seed_all(protocol['seed'])
    dev, test = protocol['development_samples'], protocol['test_samples']
    assert not {s['name'].split('_trimesh_')[0] for s in dev} & {s['name'].split('_trimesh_')[0] for s in test}
    print('LOAD pretrained original CLI', flush=True)
    model, unet, config, bootstrap = load_original_models(args, out, dev)
    manifest = dict(protocol_sha256=sha256(args.protocol), protocol=protocol, model_config=config,
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=gpu_snapshot(),
                    weights={p: sha256(ROOT/p) for p in ['trained_models/model_knn50.pth', 'runs/S2_train/model_best.pth']},
                    sources={p: sha256(ROOT/p) for p in ['S2/ExtractFace.py', 'S2_reconstruct.py', 'benchmarks/compare_face_thresholds.py']})
    manifest['input_sha256'] = {s['name']:dict(pointcloud=sha256(s['pointcloud']), ground_truth=sha256(s['ground_truth'])) for s in dev+test}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    dataset = PCReconSet([s['pointcloud'] for s in dev], knn=50, delta=config['delta'], rescale_delta=config['rescale_delta'])
    for i, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)):
        data = [x[0] for x in batch]
        vertices, faces = shared_predictions(data, model, unet, args.chunk_size, protocol['configs'])
        if i == 0:
            assert np.array_equal(vertices, bootstrap[0])
            assert np.array_equal(faces[identifier(protocol['baseline'])], bootstrap[1])
            manifest['default_extraction_matches_original_cli'] = True
        evaluator = PairedEvaluator(dev[i]['ground_truth'], protocol['development_surface_samples'], protocol['seed']+2*i)
        for c in protocol['configs']:
            key = identifier(c)
            stats = evaluator.evaluate(vertices, faces[key])
            rows.append(dict(sample=dev[i]['name'], config=key, **stats))
        write_csv(out/'development_per_sample.csv', rows)
        print(f'DEVELOPMENT {i+1}/{len(dev)} {dev[i]["name"]}', flush=True)
    selected, development = choose(rows, protocol['configs'], protocol['baseline'])
    (out/'development_summary.json').write_text(json.dumps(development, indent=2))
    write_csv(out/'development_comparison.csv', [dict(config=key, **{k:v for k,v in value.items() if k != 'config'}) for key,value in development.items()])
    # This file is persisted before evaluating any of the 40 evaluation cases.
    selection = dict(selected=selected, selected_identifier=identifier(selected),
                     baseline=protocol['baseline'], selection_rule=protocol['selection_rule'],
                     evaluated_test_samples_at_selection=0)
    (out/'selected_thresholds.json').write_text(json.dumps(selection, indent=2))
    print('FROZEN SELECTED', json.dumps(selection), flush=True)
    for method in ['baseline', 'selected']:
        (out/method).mkdir(exist_ok=True)
    test_rows, times = [], []
    configs = [protocol['baseline']]
    if selected != protocol['baseline']:
        configs.append(selected)
    dataset = PCReconSet([s['pointcloud'] for s in test], knn=50, delta=config['delta'], rescale_delta=config['rescale_delta'])
    for i, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)):
        data = [x[0] for x in batch]
        vertices, faces = shared_predictions(data, model, unet, args.chunk_size, configs)
        evaluator = PairedEvaluator(test[i]['ground_truth'], protocol['test_surface_samples'], protocol['seed']+10000+2*i)
        methods = dict(baseline=protocol['baseline'], selected=selected)
        reference = ROOT/'benchmarks/results/2026-10-09/full/original'/test[i]['name']
        previous = o3d.io.read_triangle_mesh(str(reference))
        assert np.array_equal(np.asarray(previous.triangles), faces[identifier(protocol['baseline'])]), 'Default faces changed versus original experiment'
        for method, c in methods.items():
            f = faces[identifier(c)]
            mesh = mesh_from_arrays(vertices, f)
            assert o3d.io.write_triangle_mesh(str(out/method/test[i]['name']), mesh)
            stats = evaluator.evaluate(vertices, f)
            test_rows.append(dict(sample=test[i]['name'], method=method, **stats))
        for repeat in range(protocol['full_pipeline_timing_repeats']):
            order = list(methods) if (i+repeat)%2 == 0 else list(reversed(methods))
            for method in order:
                mesh, timing = measure(lambda: reconstruct(data, extractor(methods[method]), model, unet, args.chunk_size))
                assert np.array_equal(np.asarray(mesh.triangles), faces[identifier(methods[method])])
                times.append(dict(sample=test[i]['name'], method=method, repeat=repeat, **timing))
        write_csv(out/'test_per_sample.csv', test_rows)
        write_csv(out/'timing_raw.csv', times)
        print(f'TEST {i+1}/{len(test)} {test[i]["name"]}', flush=True)
    summary = {m: aggregate([r for r in test_rows if r['method'] == m]) for m in ['baseline', 'selected']}
    for m in summary:
        summary[m]['mean_sample_median_seconds'] = float(np.mean([np.median([r['seconds'] for r in times if r['method']==m and r['sample']==s['name']]) for s in test]))
        summary[m]['mean_extra_peak_mib'] = float(np.mean([max(r['extra_peak_mib'] for r in times if r['method']==m and r['sample']==s['name']) for s in test]))
    # Shape-level paired bootstrap describes this fixed sample group only.
    rng = np.random.default_rng(protocol['seed'])
    indices = rng.integers(0, len(test), size=(10000, len(test)))
    paired = {}
    for metric in METRICS:
        a = np.array([r[metric] for r in test_rows if r['method']=='baseline'])
        b = np.array([r[metric] for r in test_rows if r['method']=='selected'])
        delta = b-a
        paired[metric] = dict(mean_difference=float(delta.mean()),
                             percentile_95_interval=np.percentile(delta[indices].mean(axis=1), [2.5,97.5]).tolist(),
                             selected_higher=int((delta>0).sum()), selected_lower=int((delta<0).sum()), equal=int((delta==0).sum()))
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    (out/'paired_analysis.json').write_text(json.dumps(paired, indent=2))
    write_csv(out/'comparison.csv', [dict(metric=metric, baseline=summary['baseline'][metric], selected=summary['selected'][metric], difference=summary['selected'][metric]-summary['baseline'][metric]) for metric in summary['baseline']])
    manifest.update(selected=selected, selection_frozen_before_test=True, default_faces_match_previous_original_on_all_40=True, gpu_after=gpu_snapshot())
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

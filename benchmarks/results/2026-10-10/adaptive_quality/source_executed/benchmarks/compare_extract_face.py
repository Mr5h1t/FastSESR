"""Paired real-data benchmark of the original and on-demand face extractors.

The stock reconstruction entry point initializes the pretrained models. The
reconstruction helper mirrors its LOON-UNet branch and is checked against that
entry point before any measured comparisons. Inputs and GT come from ABC test.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import open3d as o3d
import torch
import trimesh
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from dataset.pc_recon import PCReconSet
import S2.ExtractFace as extract_module
from eval.surf_acc import MeshEvaluator

spec = importlib.util.spec_from_file_location('extract_face_before', ROOT / 'benchmarks/baselines/extract_face_before.py')
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
    return result.stdout.strip()


def load_original_models(args, out, samples):
    """Use exactly the project's current CLI weight/config loading behavior."""
    warm_list = out / 'bootstrap_test_list.txt'
    warm_list.write_text(samples[0]['name'] + '\n')
    previous_argv = sys.argv
    current_extractor = extract_module.SurfExtract
    class OriginalCLIExtractor(before.SurfExtract):
        def __init__(self, confidence_top1=0.8, confidence_top2=0.5, angle=120,
                     quality_weight=0.0, quality_target=0.2):
            if quality_weight != 0:
                raise ValueError("The original baseline does not support adaptive quality.")
            super().__init__(confidence_top1, confidence_top2, angle)

    extract_module.SurfExtract = OriginalCLIExtractor
    sys.argv = [str(ROOT / 'S2_reconstruct.py'), '--use_loon_unet',
                '--loon_unet_ckpt', str(ROOT / 'runs/S2_train/model_best.pth'),
                '--dataset', 'ABC_test', '--data_root', str(ROOT / 'Data'),
                '--test_list', str(warm_list), '--out_dir', str(out / 'original_cli_bootstrap'),
                '--chunk_size', str(args.chunk_size)]
    try:
        with (out / 'original_cli_bootstrap.log').open('w') as log, contextlib.redirect_stdout(log):
            ns = runpy.run_path(str(ROOT / 'S2_reconstruct.py'), run_name='__main__')
    finally:
        sys.argv = previous_argv
        extract_module.SurfExtract = current_extractor
    model, unet, opt = ns['model'], ns['loon_unet'], ns['opt']
    assert unet is not None and not model.training and not unet.training
    bootstrap_mesh = ns['recon_mesh']
    bootstrap_arrays = (np.asarray(bootstrap_mesh.vertices).copy(), np.asarray(bootstrap_mesh.triangles).copy())
    ck = torch.load(ROOT / 'runs/S2_train/model_best.pth', map_location='cpu')
    state = ck['model_state_dict']
    active = unet.state_dict()
    missing = sorted(set(active) - set(state))
    unused = sorted(set(state) - set(active))
    assert not missing, f'Active pretrained weights missing: {missing}'
    assert all(torch.equal(value.detach().cpu(), state[key]) for key, value in active.items()), 'Active weights do not exactly match checkpoint'
    config = {'delta': opt.delta, 'rescale_delta': opt.rescale_delta,
              'chunk_size': args.chunk_size, 'knn': 50, 'pretrained_active_keys': len(active),
              'pretrained_missing_active_keys': missing, 'unused_checkpoint_keys': unused,
              'checkpoint_args': ck.get('args', {})}
    del ns, ck, state, active
    return model, unet, config, bootstrap_arrays


@torch.no_grad()
def reconstruct(data, extractor, model, unet, chunk_size):
    points, indices, center, scale = data
    points_dev = points.to('cuda')
    final_offsets = unet(points_dev.t().unsqueeze(0).contiguous()).squeeze(0).t().contiguous()
    indices_dev = indices.to('cuda')
    triangles = []
    for start in range(0, len(points), chunk_size):
        sl = slice(start, min(start + chunk_size, len(points)))
        features = points_dev[indices_dev[sl]] - points_dev[sl, None, :]
        offset_features = final_offsets[indices_dev[sl]] - final_offsets[indices_dev[sl, 0], None, :]
        logits = model(features, offset_features)
        faces = extractor(points_dev, logits, indices_dev[sl])
        if faces.numel() > 0:
            triangles.append(faces)
    if triangles:
        faces = torch.unique(torch.sort(torch.cat(triangles), dim=-1).values, dim=0)
        faces_np = faces.cpu().numpy()
    else:
        faces_np = np.empty((0, 3), dtype=np.int64)
    vertices_np = (points_dev.detach().cpu() * scale + center).numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    return mesh


@torch.no_grad()
def prediction_chunk(data, model, unet, chunk_size):
    points, indices, _, _ = data
    points_dev = points.to('cuda')
    offsets = unet(points_dev.t().unsqueeze(0).contiguous()).squeeze(0).t().contiguous()
    indices_dev = indices[:chunk_size].to('cuda')
    features = points_dev[indices_dev] - points_dev[:len(indices_dev), None, :]
    offset_features = offsets[indices_dev] - offsets[indices_dev[:, 0], None, :]
    logits = model(features, offset_features)
    return points_dev, logits, indices_dev


def measure(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    return result, {'seconds': elapsed, 'extra_peak_mib': (peak - base) / 2**20,
                    'total_peak_mib': peak / 2**20}


def same_mesh(a, b):
    return (np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices)) and
            np.array_equal(np.asarray(a.triangles), np.asarray(b.triangles)))


def accuracy(gt_path, mesh, seed, sample_count):
    """Same normalization and CD/F1/NC definitions as eval/surf_acc.py.

    Explicit sampling seeds make the baseline/improved comparison reproducible.
    Edge metrics are not computed because this change preserves every face.
    """
    evaluator = MeshEvaluator(sample_num=sample_count)
    gt = evaluator.normalize_diagonal(trimesh.load_mesh(gt_path))
    pred = evaluator.normalize_diagonal(trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)))
    gt_points, gt_faces = trimesh.sample.sample_surface(gt, sample_count, seed=seed)
    pred_points, pred_faces = trimesh.sample.sample_surface(pred, sample_count, seed=seed + 1)
    gt_normals, pred_normals = gt.face_normals[gt_faces], pred.face_normals[pred_faces]
    d_gt, i_gt = KDTree(pred_points).query(gt_points, k=1)
    d_pred, i_pred = KDTree(gt_points).query(pred_points, k=1)
    recall, precision = np.mean(d_gt < .003), np.mean(d_pred < .003)
    nc_gt = np.mean(np.abs(np.sum(gt_normals * pred_normals[i_gt[:, 0]], axis=1)))
    nc_pred = np.mean(np.abs(np.sum(pred_normals * gt_normals[i_pred[:, 0]], axis=1)))
    return {'CD1_x100': float((d_gt.mean() + d_pred.mean()) * 100),
            'CD2_x1e5': float(((d_gt*d_gt).mean() + (d_pred*d_pred).mean()) * 1e5),
            'F1_at_0.003': float(2*precision*recall/(precision+recall)) if precision+recall else 0.,
            'NC': float((nc_gt + nc_pred) / 2), 'triangles': len(mesh.triangles)}


def summarize(rows, metrics):
    result = {}
    for stage in ('reconstruction', 'extraction'):
        result[stage] = {}
        sample_names = sorted({r['sample'] for r in rows if r['stage'] == stage})
        for method in ('original', 'improved'):
            sample_times = [np.median([r['seconds'] for r in rows if r['stage'] == stage and r['method'] == method and r['sample'] == name]) for name in sample_names]
            peaks = [max(r['extra_peak_mib'] for r in rows if r['stage'] == stage and r['method'] == method and r['sample'] == name) for name in sample_names]
            result[stage][method] = {'mean_sample_median_seconds': float(np.mean(sample_times)),
                                    'median_sample_median_seconds': float(np.median(sample_times)),
                                    'mean_extra_peak_mib': float(np.mean(peaks)), 'max_extra_peak_mib': float(max(peaks))}
        a, b = result[stage]['original'], result[stage]['improved']
        result[stage]['time_reduction_pct'] = 100 * (1 - b['mean_sample_median_seconds'] / a['mean_sample_median_seconds'])
        result[stage]['memory_reduction_pct'] = 100 * (1 - b['mean_extra_peak_mib'] / a['mean_extra_peak_mib'])
    result['accuracy'] = {method: {key: float(np.mean([r[key] for r in metrics if r['method'] == method])) for key in ('CD1_x100', 'CD2_x1e5', 'F1_at_0.003', 'NC', 'triangles')} for method in ('original', 'improved')}
    result['all_meshes_exactly_equal'] = True
    result['sample_count'] = len(metrics)//2
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--extraction-repeats', type=int, default=15)
    parser.add_argument('--chunk-size', type=int, default=2000)
    parser.add_argument('--sample-count', type=int, default=100000)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ('original', 'improved'):
        (out/name).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    seed_all(20261009)
    samples = json.loads(args.manifest.read_text())['samples']
    if args.limit:
        samples = samples[:args.limit]
    print('INITIALIZE stock pretrained pipeline', flush=True)
    model, unet, config, bootstrap_arrays = load_original_models(args, out, samples)
    manifest = {'sample_selection_manifest': str(args.manifest.resolve()), 'sample_selection_sha256': sha256(args.manifest),
                'config': config, 'seed': 20261009, 'reconstruction_repeats': args.repeats,
                'extraction_repeats': args.extraction_repeats, 'surface_samples': args.sample_count,
                'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0), 'gpu_before': gpu_snapshot(),
                'timing_scope': 'GPU reconstruction through CPU mesh assembly; excludes disk reads, PC preprocessing, weight loading and mesh writes',
                'extraction_scope': 'first up-to-2000-point chunk of real logits for each shape',
                'pair_order': 'alternates original/improved order across repeats and shapes',
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.rglob('*.py') if '.git' not in p.parts},
                'weight_sha256': {p: sha256(ROOT/p) for p in ('trained_models/model_knn50.pth', 'runs/S2_train/model_best.pth', 'S1_training/model_k50/best_model')},
                'samples': samples}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    dataset = PCReconSet([s['pointcloud'] for s in samples], knn=50, delta=config['delta'], rescale_delta=config['rescale_delta'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    extractors = {'original': before.SurfExtract(), 'improved': extract_module.SurfExtract()}
    rows, metrics = [], []
    fields = ['sample', 'num_points', 'stage', 'method', 'repeat', 'seconds', 'extra_peak_mib', 'total_peak_mib']
    with (out/'timing_raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, batch in enumerate(loader):
            data = [item[0] for item in batch]
            sample = samples[index]
            num_points = len(data[0])
            print(f'SAMPLE {index+1}/{len(samples)} {sample["name"]} points={num_points}', flush=True)
            manifest['samples'][index]['pointcloud_sha256'] = sha256(sample['pointcloud'])
            manifest['samples'][index]['ground_truth_sha256'] = sha256(sample['ground_truth'])
            meshes = {}
            # Warm both paths on each actual shape before timed measurements.
            for method in ('original', 'improved'):
                meshes[method] = reconstruct(data, extractors[method], model, unet, args.chunk_size)
            assert same_mesh(meshes['original'], meshes['improved']), 'Warm-up face mismatch'
            if index == 0:
                assert np.array_equal(np.asarray(meshes['original'].vertices), bootstrap_arrays[0])
                assert np.array_equal(np.asarray(meshes['original'].triangles), bootstrap_arrays[1])
                manifest['benchmark_matches_original_cli_exactly'] = True
            assert len(meshes['original'].triangles) > 0, 'No surface extracted'
            for repeat in range(args.repeats):
                order = ('original', 'improved') if (repeat+index)%2 == 0 else ('improved', 'original')
                for method in order:
                    seed_all(20261009 + index)
                    mesh, stats = measure(lambda: reconstruct(data, extractors[method], model, unet, args.chunk_size))
                    assert same_mesh(mesh, meshes['original']), 'Repeated full-pipeline face mismatch'
                    row = dict(sample=sample['name'], num_points=num_points, stage='reconstruction', method=method, repeat=repeat, **stats)
                    rows.append(row); writer.writerow(row); stream.flush()
                    del mesh
            chunk = prediction_chunk(data, model, unet, args.chunk_size)
            expected = extractors['original'](*chunk).cpu()
            for method in ('original', 'improved'):
                for _ in range(3):
                    extractors[method](*chunk)
            for repeat in range(args.extraction_repeats):
                order = ('original', 'improved') if (repeat+index)%2 == 0 else ('improved', 'original')
                for method in order:
                    faces, stats = measure(lambda: extractors[method](*chunk))
                    assert torch.equal(faces.cpu(), expected), 'Real-logit extraction mismatch'
                    row = dict(sample=sample['name'], num_points=len(chunk[2]), stage='extraction', method=method, repeat=repeat, **stats)
                    rows.append(row); writer.writerow(row); stream.flush()
                    del faces
            del chunk, expected
            quality = {}
            for method in ('original', 'improved'):
                mesh = meshes[method]
                path = out/method/sample['name']
                assert o3d.io.write_triangle_mesh(str(path), mesh)
                quality[method] = accuracy(sample['ground_truth'], mesh, 20261009 + 2*index, args.sample_count)
                metrics.append(dict(sample=sample['name'], method=method, **quality[method]))
            assert quality['original'] == quality['improved'], 'Deterministic surface-metric mismatch'
            print('  exact mesh match; medians(seconds)', {method: round(float(np.median([r['seconds'] for r in rows if r['stage']=='reconstruction' and r['method']==method and r['sample']==sample['name']])), 4) for method in extractors}, flush=True)
            (out/'accuracy_per_sample.json').write_text(json.dumps(metrics, indent=2))
            (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (out/'accuracy_per_sample.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader();writer.writerows(metrics)
    manifest['gpu_after'] = gpu_snapshot()
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    summary = summarize(rows, metrics)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

"""Audit recorded outputs and summarize a byte-deduplicated held-out subset."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from S2.PlanarRepair import edge_table


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--skip-cli-check', action='store_true', help='Audit a reproduction without the separately generated CLI smoke output')
    args = parser.parse_args()
    out = args.out.resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    protocol = manifest['protocol']
    checks = {}
    checks['sources_match'] = all(digest(ROOT / rel) == value for rel, value in manifest['sources'].items())
    checks['weights_match'] = all(digest(ROOT / rel) == value for rel, value in manifest['weights'].items())
    assert checks['sources_match'] and checks['weights_match']
    expected = {'development_raw.csv': 144, 'test_raw.csv': 360,
                'test_per_shape.csv': 120, 'timing_raw.csv': 360,
                'repair_per_shape.csv': 56}
    for file, count in expected.items():
        rows = read_rows(out / file)
        assert len(rows) == count, (file, len(rows))
    checks['row_counts'] = expected
    metrics = read_rows(out / 'test_raw.csv')
    shapes = read_rows(out / 'test_per_shape.csv')
    summary = json.loads((out / 'summary.json').read_text())
    methods = list(summary)
    keys = list(summary['baseline'])
    for row in shapes:
        raw = [r for r in metrics if r['sample'] == row['sample'] and r['config'] == row['method']]
        assert len(raw) == 3
        for key in keys:
            assert np.isfinite(float(row[key]))
            if key not in ['seconds', 'extra_peak_mib']:
                assert np.isclose(float(row[key]), np.mean([float(r[key]) for r in raw]), rtol=1e-12, atol=1e-12)
    for method in methods:
        for key in keys:
            assert np.isclose(summary[method][key], np.mean([float(r[key]) for r in shapes if r['method'] == method]), rtol=1e-12, atol=1e-12)
    checks['raw_and_summary_agree'] = True
    for sample in protocol['test_samples']:
        meshes = {method: o3d.io.read_triangle_mesh(str(out / method / sample['name'])) for method in methods}
        for method, mesh in meshes.items():
            np.testing.assert_array_equal(np.asarray(mesh.vertices), np.asarray(meshes['baseline'].vertices))
            faces = np.asarray(mesh.triangles)
            _, counts, _, _ = edge_table(faces)
            row = next(r for r in shapes if r['sample'] == sample['name'] and r['method'] == method)
            assert int((counts > 2).sum()) == int(float(row['nonmanifold_edges']))
            assert int((counts == 1).sum()) == int(float(row['boundary_edges']))
        a = next(r for r in shapes if r['sample'] == sample['name'] and r['method'] == 'baseline')
        b = next(r for r in shapes if r['sample'] == sample['name'] and r['method'] == 'planar_repaired')
        assert float(b['boundary_edges']) <= float(a['boundary_edges'])
        assert float(b['nonmanifold_edges']) <= float(a['nonmanifold_edges'])
    checks['all_40_vertex_arrays_unchanged_and_topology_guards_hold'] = True
    if not args.skip_cli_check:
        smoke = (out / 'cli_smoke_list.txt').read_text().strip()
        cli_mesh = out / 'cli_selected_smoke' / smoke
        assert digest(cli_mesh) == digest(out / 'planar_repaired' / smoke)
        cli_config = json.loads((out / 'cli_selected_smoke/face_extraction_config.json').read_text())
        assert cli_config['repair_planar_overlaps'] and cli_config['planar_tolerance'] == protocol['plane_tolerance']
        checks['real_cli_matches_experiment_byte_for_byte'] = True
    else:
        checks['real_cli_matches_experiment_byte_for_byte'] = 'not checked in this reproduction'
    # Audit against every earlier evaluated input, including same geometry saved
    # under a different CAD document name. Do not filter by measured quality.
    history = []
    for rel in ['2026-10-09/full/manifest.json', '2026-10-10/thresholds/protocol.json',
                '2026-10-10/adaptive_quality/protocol.json']:
        data = json.loads((ROOT / 'benchmarks/results' / rel).read_text())
        history.extend(data.get('samples', []))
        history.extend(data.get('development_samples', []) + data.get('test_samples', []))
    history.extend(protocol['development_samples'])
    known = {kind: {} for kind in ['pointcloud', 'ground_truth']}
    for sample in history:
        for kind in known:
            known[kind].setdefault(digest(sample[kind]), set()).add(sample['name'])
    excluded = []
    included = []
    for sample in protocol['test_samples']:
        matches = set()
        for kind in known:
            value = digest(sample[kind])
            assert value == manifest['input_sha256'][sample['name']][kind]
            matches.update(known[kind].get(value, set()))
        if matches:
            excluded.append({'sample': sample['name'], 'duplicates': sorted(matches)})
        else:
            included.append(sample['name'])
        for kind in known:
            known[kind].setdefault(digest(sample[kind]), set()).add(sample['name'])
    clean_rows = [row for row in shapes if row['sample'] in included]
    clean = {method: {key: float(np.mean([float(row[key]) for row in clean_rows if row['method'] == method]))
                      for key in keys} for method in methods}
    rng = np.random.default_rng(protocol['seed'])
    indices = rng.integers(0, len(included), size=(10000, len(included)))
    paired = {}
    for method in methods[1:]:
        paired[method] = {}
        for key in keys:
            a = np.array([float(r[key]) for r in clean_rows if r['method'] == 'baseline'])
            b = np.array([float(r[key]) for r in clean_rows if r['method'] == method])
            delta = b - a
            paired[method][key] = {'mean_difference': float(delta.mean()),
                                  'percentile_95_interval': np.percentile(delta[indices].mean(axis=1), [2.5, 97.5]).tolist()}
    (out / 'heldout_summary.json').write_text(json.dumps(clean, indent=2))
    (out / 'heldout_paired_analysis.json').write_text(json.dumps(paired, indent=2))
    with (out / 'heldout_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['metric'] + methods)
        writer.writeheader()
        writer.writerows({'metric': key, **{m: clean[m][key] for m in methods}} for key in keys)
    checks.update(heldout_count=len(included), heldout_samples=included,
                  excluded_byte_duplicates=excluded,
                  deduplication_note='Post-run byte-hash audit; no metric-dependent exclusion. Original 40-case records are retained.')
    (out / 'verification.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps({'checks': checks, 'heldout_summary': clean}, indent=2))


if __name__ == '__main__':
    main()

"""Differential regression tests against the exact pre-change implementation."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from S2.ExtractFace import SurfExtract
spec = importlib.util.spec_from_file_location('extract_face_before', ROOT / 'benchmarks/baselines/extract_face_before.py')
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)
torch.set_num_threads(4)


class ExtractionEquivalence(unittest.TestCase):
    def compare(self, points, logits, indices):
        expected = before.SurfExtract()(points, logits, indices)
        actual = SurfExtract()(points, logits, indices)
        self.assertTrue(torch.equal(expected, actual))

    def make_case(self, k, device='cpu', seed=123):
        generator = torch.Generator().manual_seed(seed)
        points = torch.rand(157, 3, generator=generator)
        indices = torch.stack([torch.randperm(157, generator=generator)[:k] for _ in range(23)])
        logits = torch.randn(23, (k-1)**2, generator=generator) * 4
        return points.to(device), logits.to(device), indices.to(device)

    def test_varied_neighborhood_sizes_and_global_indices(self):
        for k in (3, 4, 8, 50, 100):
            for seed in (0, 1, 2):
                with self.subTest(k=k, seed=seed):
                    self.compare(*self.make_case(k, seed=seed))

    def test_equal_logits_preserve_original_tie_order(self):
        p, scores, idx = self.make_case(50)
        for value in (-5., 0., 5.):
            scores.fill_(value)
            self.compare(p, scores, idx)
        scores = scores.round()
        self.compare(p, scores, idx)

    def test_no_faces_below_threshold(self):
        p, scores, idx = self.make_case(50)
        scores.fill_(-20.)
        self.compare(p, scores, idx)
        self.assertEqual(SurfExtract()(p, scores, idx).shape, (0, 3))

    def test_degenerate_geometry_and_duplicate_indices(self):
        p, scores, idx = self.make_case(8)
        p.zero_()
        idx[:, 2] = idx[:, 1]
        self.compare(p, scores, idx)

    def test_noncontiguous_logits_and_indices(self):
        p, scores, idx = self.make_case(50)
        scores = scores.reshape(23, 49, 49).transpose(1, 2)
        idx = idx.repeat_interleave(2, dim=0)[::2]
        self.compare(p, scores, idx)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_random_and_tied_logits(self):
        for k in (3, 8, 50, 100):
            p, scores, idx = self.make_case(k, device='cuda')
            self.compare(p, scores, idx)
            scores.fill_(5.)
            self.compare(p, scores, idx)

    def test_configurable_thresholds_change_selected_faces(self):
        points = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                               [0., 1., 0.], [0., -1., 0.]])
        indices = torch.tensor([[0, 1, 2, 3]])
        scores = torch.full((1, 3, 3), -20.)
        scores[0, 0, 1] = torch.logit(torch.tensor(.75))
        scores[0, 0, 2] = torch.logit(torch.tensor(.55))
        self.assertEqual(len(SurfExtract()(points, scores, indices)), 0)
        self.assertEqual(len(SurfExtract(.7, .5)(points, scores, indices)), 2)
        self.assertEqual(len(SurfExtract(.7, .6)(points, scores, indices)), 0)

    def test_angle_parameter_changes_pair_acceptance(self):
        import math
        radians = math.radians(100)
        points = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                               [0., 1., 0.], [0., math.cos(radians), math.sin(radians)]])
        indices = torch.tensor([[0, 1, 2, 3]])
        scores = torch.full((1, 3, 3), -20.)
        scores[0, 0, 1] = 5.
        scores[0, 0, 2] = 4.
        self.assertEqual(len(SurfExtract(angle=90)(points, scores, indices)), 2)
        self.assertEqual(len(SurfExtract(angle=120)(points, scores, indices)), 0)

    def test_invalid_thresholds_rejected(self):
        for kwargs in ({'confidence_top1': -0.1}, {'confidence_top2': 1.1},
                       {'confidence_top1': float('nan')}, {'angle': 0},
                       {'angle': 180}, {'angle': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SurfExtract(**kwargs)



if __name__ == '__main__':
    unittest.main()

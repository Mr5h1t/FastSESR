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


if __name__ == '__main__':
    unittest.main()

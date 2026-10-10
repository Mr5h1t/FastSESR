import sys
import unittest
from pathlib import Path
import numpy as np
import shapely

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from S2.PlanarRepair import repair_planar_overlaps,edge_table


class PlanarRepairTests(unittest.TestCase):
    def square(self):
        points=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],[.25,.15,0.],[.75,.2,0.]])
        faces=np.array([[0,1,2],[0,2,3],[0,1,4],[0,1,5]])
        return points,faces

    def union(self,points,faces):
        return shapely.union_all(shapely.polygons(points[faces,:2]))

    def test_overlapping_square_keeps_coverage_and_removes_nonmanifold_edge(self):
        p,f=self.square()
        new,stats=repair_planar_overlaps(p,f)
        self.assertGreater(stats['repaired_patches'],0)
        self.assertLess(self.union(p,f).symmetric_difference(self.union(p,new)).area,1e-12)
        _,counts,_,_=edge_table(new)
        self.assertEqual(int((counts>2).sum()),0)
        self.assertEqual(int((counts==1).sum()),4)
        np.testing.assert_array_equal(np.unique(f),np.unique(new))

    def test_crease_seam_and_nonplanar_face_preserved(self):
        p,f=self.square()
        p=np.concatenate([p,[[.5,0.,1.]]])
        f=np.concatenate([f,[[0,1,6]]])
        new,_=repair_planar_overlaps(p,f)
        self.assertIn((0,1,6),map(tuple,np.sort(new,axis=1)))
        edges,_,_,_=edge_table(new)
        self.assertIn((0,1),map(tuple,edges))

    def test_real_hole_is_not_filled(self):
        p=np.array([[0.,0.,0.],[3.,0.,0.],[3.,3.,0.],[0.,3.,0.],[1.,1.,0.],[2.,1.,0.],[2.,2.,0.],[1.,2.,0.]])
        f=np.array([[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7],[0,1,4]])
        new,_=repair_planar_overlaps(p,f)
        domain=self.union(p,new)
        self.assertFalse(domain.covers(shapely.Point(1.5,1.5)))
        self.assertLess(domain.symmetric_difference(self.union(p,f)).area,1e-10)

    def test_curved_and_clean_regions_unchanged(self):
        p=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,.1]])
        f=np.array([[0,1,2],[0,2,3]])
        new,stats=repair_planar_overlaps(p,f)
        np.testing.assert_array_equal(new,f)
        self.assertEqual(stats['repaired_patches'],0)

    def test_invalid_tolerance_rejected(self):
        p,f=self.square()
        for value in [0.,-1.,float('nan'),1.]:
            with self.assertRaises(ValueError):
                repair_planar_overlaps(p,f,value)


if __name__=='__main__':
    unittest.main()

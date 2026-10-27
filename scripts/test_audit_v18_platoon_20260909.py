"""CPU-only tests of the isolated audit analysis, no production changes."""
import unittest
import numpy as np
from audit_v18_platoon_20260909 import bootstrap, clusters

class BootstrapTests(unittest.TestCase):
    def test_identical_models(self):
        x=[[0,1],[5,10],[3,4]]
        r=bootstrap(x,x,1000)
        self.assertEqual(r['ci95_pp'],[0.,0.])
    def test_pooled_denominators(self):
        r=bootstrap([[0,1],[0,9]],[[1,1],[0,9]],1000)
        self.assertEqual(r['delta_pp'],10.)
    def test_height_copy_does_not_double_sample_size(self):
        a=np.array([[0,2],[1,2],[2,2]])
        b=np.array([[1,2],[2,2],[2,2]])
        x=bootstrap(a,b,1000); y=bootstrap(a*2,b*2,1000)
        self.assertEqual(x['ci95_pp'],y['ci95_pp'])
        self.assertEqual(x['clusters'],y['clusters'])
    def test_known_shift(self):
        r=bootstrap([[0,2]]*4,[[1,2]]*4,1000)
        self.assertEqual(r['ci95_pp'],[50.,50.])
    def test_incomplete_pair_rejected(self):
        with self.assertRaises(ValueError):
            clusters([dict(clip_id='a',scenario='s',seed='1',h_copy='1')])
    def test_manifest_order_irrelevant(self):
        rr=[dict(clip_id=f'{s}{h}',scenario='s',seed=str(s),h_copy=str(h)) for s in [2,1] for h in [2,1]]
        self.assertEqual(clusters(rr),clusters(rr[::-1]))
    def test_undefined_denominator_rejected(self):
        with self.assertRaises(ValueError): bootstrap([[0,0]],[[0,0]])
    def test_mismatched_gt_rejected(self):
        with self.assertRaises(ValueError): bootstrap([[0,1]],[[0,2]])
    def test_randomized_models_are_paired(self):
        r=bootstrap([[1,2],[0,4],[4,5]],[[1,2],[0,4],[4,5]],1000)
        self.assertEqual(r['ci95_pp'],[0.,0.])

if __name__=='__main__': unittest.main()

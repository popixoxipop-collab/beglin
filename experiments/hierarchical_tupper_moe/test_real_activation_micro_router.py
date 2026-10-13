import pathlib
import struct
import tempfile
import unittest
import numpy as np
from test_real_checkpoint_reader import make_fixture
from real_activation_micro_router import evaluate,read_route

class TrueActivationTests(unittest.TestCase):
    def test_fake_safetensors_real_activation_format_with_actual_routes(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d)
            index,_=make_fixture(root,expert_count=8)
            cap=root/"trace.bin";routes=root/"routes.txt"
            with cap.open("wb") as f:
                for pos in range(6):
                    h=np.random.default_rng(pos).normal(size=64).astype("<f4")
                    f.write(struct.pack("<iii",pos,1,64))
                    f.write(h.tobytes())
            routes.write_text("".join(f"pos {p} layer 1 experts 0:0.6 1:0.4\n" for p in range(6)))
            result=evaluate(index,cap,routes,1,64,128,widths=(16,64),fractions=(.5,1.))
            self.assertEqual(result["holdout_positions"],3)
            self.assertTrue(result["true_macro_route"])
            self.assertTrue(result["true_activation"])
            self.assertLess(result["results"]["16"]["1.0"]["heldout_macro_output_relative_l2"],1e-10)

    def test_missing_real_routes_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/"route.txt"
            p.write_text("pos 0 layer 1 experts 0:1\n")
            with self.assertRaisesRegex(ValueError,"incomplete"):
                read_route(p,1,{0,1})
if __name__=="__main__":
    unittest.main()

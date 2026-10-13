import pathlib, struct, tempfile, unittest
import numpy as np
from test_real_checkpoint_reader import make_fixture
from trainable_micro_router import (fit_kernel, predict, fit_one_expert,
                                    evaluate, mask_for)

class TrainableMicroTests(unittest.TestCase):
    def test_learning_never_reads_heldout_channel_targets(self):
        rng=np.random.default_rng(22)
        x=rng.normal(size=(12,64))
        c=rng.normal(size=(12,8,64))
        a=fit_one_expert(x[:6],c[:6],x[6:],c[6:],.5)
        shifted=c[6:]+10000
        b=fit_one_expert(x[:6],c[:6],x[6:],shifted,.5)
        np.testing.assert_array_equal(a["dynamic_mask"],b["dynamic_mask"])
        np.testing.assert_array_equal(a["static_mask"],b["static_mask"])

    def test_exact_full_channel_mask_and_compensator(self):
        rng=np.random.default_rng(23)
        x=rng.normal(size=(12,32))
        c=rng.normal(size=(12,8,32))
        a=fit_one_expert(x[:6],c[:6],x[6:],c[6:],1.)
        for name in ("static","static_mean_residual","dynamic",
                     "dynamic_mean_residual","dynamic_ridge_residual"):
            np.testing.assert_allclose(a[name],a["full"],rtol=1e-12,atol=1e-12)

    def test_ridge_fitted_only_on_train(self):
        rng=np.random.default_rng(25)
        x=rng.normal(size=(12,17))
        y=rng.normal(size=(12,6))
        model=fit_kernel(x[:8],y[:8],.25)
        v=predict(model,x[8:])
        y[8:]+=100000.
        np.testing.assert_allclose(v,predict(model,x[8:]))

    def test_reader_real_format_scratch_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            index,_=make_fixture(root,expert_count=8)
            cap=root/"true_h2.bin"
            route=root/"route.txt"
            with cap.open("wb") as fd:
                for pos in range(8):
                    h=np.random.default_rng(pos+3).normal(size=64).astype("<f4")
                    fd.write(struct.pack("<iii",pos,1,64))
                    fd.write(h.tobytes())
            route.write_text("".join(f"pos {p} layer 1 experts 0:0.6 1:0.4\n" for p in range(8)))
            report=evaluate(index,cap,route,hidden=64,intermediate=128,widths=(16,64),fractions=(.5,1.))
            self.assertEqual(report["status"],"EVALUATED_NOT_PROMOTABLE")
            self.assertEqual(report["train_token_positions"],4)
            self.assertTrue(report["results"]["16"]["0.5"]["tupper_roundtrip"])
            for width in ("16","64"):
                values=report["results"][width]["1.0"]["methods"]
                self.assertLess(values["dynamic"]["heldout_macro_output_relative_l2"],1e-10)

    def test_invalid_inputs_block(self):
        with self.assertRaises(ValueError): mask_for(np.ones((1,4)),5)
        with self.assertRaises(ValueError): fit_kernel(np.ones((2,5)),np.ones((2,2)),0)
        with self.assertRaises(ValueError):
            from trainable_micro_router import features
            features(np.array([[float("nan")]]))
if __name__=="__main__":unittest.main()

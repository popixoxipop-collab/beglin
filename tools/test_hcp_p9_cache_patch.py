#!/usr/bin/env python3
"""P9 reversible source and cache safety invariants; no GPU executes here."""
import hashlib,pathlib,tempfile,unittest
from hcp_p9_cache_patch import patch,strip,execute,sha,P8_MLX_SHA,ENV

SOURCE=pathlib.Path(__file__).resolve().parents[1]/"mlx_moe.cpp"
class P9PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=SOURCE.read_bytes()
        cls.changed=patch(cls.raw)
    def test_01_original_blob_pinned(self):
        b=self.raw
        self.assertEqual(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest(),P8_MLX_SHA)
    def test_02_instrumentation_exact_inverse(self):
        self.assertEqual(strip(self.changed),self.raw)
        self.assertNotEqual(self.changed,self.raw)
    def test_03_double_patch_rejected(self):
        with self.assertRaisesRegex(ValueError,"double"):
            patch(self.changed)
    def test_04_missing_marker_cannot_unpatch(self):
        with self.assertRaises(ValueError):strip(self.raw)
    def test_05_modified_source_not_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"src.cpp";q=pathlib.Path(td)/"cache.cpp"
            p.write_bytes(self.raw+b"\n")
            with self.assertRaisesRegex(ValueError,"not the P5"):
                execute(p,q)
            self.assertFalse(q.exists())
    def test_06_no_source_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"src.cpp";q=pathlib.Path(td)/"p9.cpp"
            p.write_bytes(self.raw)
            execute(p,q)
            self.assertEqual(p.read_bytes(),self.raw)
            self.assertEqual(q.read_bytes(),self.changed)
            with self.assertRaisesRegex(ValueError,"new output"):
                execute(p,q)
    def test_07_opt_in_off_by_default(self):
        s=self.changed.decode()
        self.assertIn('std::strcmp(e,"1")==0',s)
        self.assertEqual(s.count(ENV),1)
    def test_08_exact_tensor_whitelist(self):
        s=self.changed.decode()
        self.assertIn('"model.layers.3.mlp.shared_experts.up_proj"',s)
        self.assertIn('"model.layers.26.mlp.shared_experts.down_proj"',s)
        self.assertIn('bits!=5 && bits!=6',s)
    def test_09_bounded_packed_residency(self):
        s=self.changed.decode()
        self.assertIn("P9_MAX_ENTRIES = 4",s)
        self.assertIn("P9_MAX_BYTES = 256ull * 1024ull * 1024ull",s)
        self.assertIn("g_p9_retained_bytes<=P9_MAX_BYTES-reserved",s)
    def test_10_full_byte_and_lifetime_guard(self):
        s=self.changed.decode()
        self.assertIn("s.source_blob==blob",s)
        self.assertIn("s.source_codes.size()!=p9_codes_size",s)
        self.assertIn("std::memcmp(s.source_codes.data()",s)
        self.assertIn("std::memcmp(s.source_scales.data()",s)
        self.assertIn("return 0; // fail CLOSED",s)
    def test_11_snapshot_rollback_unchanged(self):
        base=self.raw.decode();new=self.changed.decode()
        for marker in ("int mlx_gpu_restore_binding_snapshot(uint64_t snapshot_id)",
                       "int mlx_gpu_drop_binding_snapshot(uint64_t snapshot_id)",
                       "int mlx_gpu_reset_runtime_epoch(void)"):
            self.assertEqual(base.count(marker),1)
            self.assertEqual(new.count(marker),1)
        self.assertEqual(strip(self.changed),self.raw)
    def test_12_scales_are_rewrapped_each_hit(self):
        s=self.changed.decode()
        self.assertIn('mx::array scales(scale_ptr,',s)
        self.assertIn('biases = mx::multiply(scales,',s)
        self.assertIn("w=s.packed_w;",s)
    def test_13_cache_no_external_tensor_broadening(self):
        self.assertEqual(self.changed.count(b'P9_PREPARED_CACHE_HIT name='),1)
        self.assertEqual(self.changed.count(b'P9_PREPARED_CACHE_STORE name='),1)
        self.assertEqual(self.changed.count(b'P9_PREPARED_CACHE_SOURCE_MISMATCH'),1)

if __name__=="__main__":unittest.main()

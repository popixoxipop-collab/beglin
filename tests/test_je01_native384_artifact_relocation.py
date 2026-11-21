#!/usr/bin/env python3
"""JE01 actual 512 original int32 artifact relocation fail-closed unit tests."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import je01_native384_relocate_actual_artifact as move


class RelocationGuard(unittest.TestCase):
    def test_verified_native_bytes_unmodified_and_bos(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            ident="f"*64
            file=root/(ident+".bin")
            raw=struct.pack("<4i",100000,549,4345,280)
            file.write_bytes(raw)
            entry={
                "input_id":ident,"token_count_with_bos":4,
                "native_int32_sha256":hashlib.sha256(raw).hexdigest(),
                "native_int32_bytes":len(raw)}
            self.assertEqual(move.verify_file(file,entry,root),16)
            file.write_bytes(raw[:-4]+b"\x00\x00\x00\x00")
            with self.assertRaisesRegex(ValueError,"CONTENT_SHA_CHANGED"):
                move.verify_file(file,entry,root)

    def test_rejects_symlink_and_out_of_range_token(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            ident="a"*64
            bad=struct.pack("<2i",100000,102400)
            file=root/(ident+".bin")
            file.write_bytes(bad)
            item={"input_id":ident,"native_int32_sha256":hashlib.sha256(bad).hexdigest(),
                  "native_int32_bytes":len(bad),"token_count_with_bos":2}
            with self.assertRaisesRegex(ValueError,"TOKENS_INVALID"):
                move.verify_file(file,item,root)
            file.unlink()
            target=root/"other.bin"
            target.write_bytes(struct.pack("<2i",100000,111))
            file.symlink_to(target)
            with self.assertRaisesRegex(ValueError,"SOURCE_PATH_ESCAPE"):
                move.verify_file(file,item,root)

    def test_forged_source_receipt_refused_without_output_write(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            src=base/"source";src.mkdir()
            dest=base/"new_manifest"
            (src/"JE01_NATIVE_384_INPUTS_RECEIPT.json").write_text(
                json.dumps({"receipt_sha256":"e"*64}))
            with self.assertRaisesRegex(ValueError,"SOURCE_RECEIPT_SHA_DRIFT"):
                move.relocate(src,dest)
            self.assertFalse(dest.exists())

    def test_frozen_original_checkpoints_pinned(self):
        self.assertEqual(move.SOURCE_RECEIPT_SHA,
           "2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34")
        self.assertEqual(move.SOURCE_MANIFEST_SHA,
           "a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147")
        self.assertEqual(move.COHORTS,
           {"calibration":256,"same_source_holdout":128,"ood_holdout":128})
        self.assertEqual(move.ORIG_TOKENIZER_SHA,
           "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669")

if __name__=="__main__":
    unittest.main(verbosity=2)

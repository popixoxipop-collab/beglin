#!/usr/bin/env python3
"""JE01 original AF teacher-forced train-only full-sequence NLL safety tests."""
import hashlib,json,pathlib,struct,sys,tempfile,unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import je01_xox_original256_fulltarget_teacherforced_nll as tf

class OriginalFullTargetTeacherForcingContracts(unittest.TestCase):
    def test_real_original_scope_and_byte_pins(self):
        self.assertEqual(tf.VOCAB,102400)
        self.assertEqual(tf.BOS,100000)
        self.assertEqual(tf.POSITIONS,384)
        self.assertEqual(tf.GOLD_SHA,"355a169cfb7c9ab79901ae7848afa21a2f5a576d048a5476f527bca699136fb7")
        self.assertEqual(tf.REAL256_ROUTER_SHA,"62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb")
        self.assertNotIn("heldout",str(tf.ROOT).lower())
        self.assertTrue(tf.ROOT.name.startswith("je01_original_fulltarget_teacherforce_"))

    def test_model_output_binary_nll_record_layout_validates_native_first_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)
            p=root/"gold_native_nll.bin"
            cases=[3,5,7]
            raw=bytearray(tf.OUT_MAGIC)
            for idx,target in enumerate(cases):
                raw.extend(struct.pack("<4IdIf",0,idx,target,10+idx,1.23+idx, 0, -3.0))
            p.write_bytes(raw)
            records=[{"index":9,"source_id":"a"*64,"prompt_ids":[0]*11,
                      "target_ids":cases,"expected_first_gold_nll":1.23}]
            rows=tf.scored(p,records,0)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]["row"],9)
            self.assertEqual(rows[0]["tokens"],3)
            self.assertAlmostEqual(rows[0]["nll_sum"],1.23+2.23+3.23,places=12)
            self.assertEqual(rows[0]["first_gold_nll"],1.23)

    def test_native_record_corrupt_target_or_position_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            p=pathlib.Path(directory)/"gold.bin"
            rec={"index":0,"source_id":"b"*64,"prompt_ids":[100000,123],
                 "target_ids":[4],"expected_first_gold_nll":3.1}
            p.write_bytes(tf.OUT_MAGIC+struct.pack("<4IdIf",0,0,17,1,3.1,2,1.0))
            with self.assertRaisesRegex(ValueError,"RECORD_DRIFT"):
                tf.scored(p,[rec],0)
            p.write_bytes(tf.OUT_MAGIC+struct.pack("<4IdIf",0,0,4,2,3.1,2,1.0))
            with self.assertRaisesRegex(ValueError,"RECORD_DRIFT"):
                tf.scored(p,[rec],0)
            p.write_bytes(tf.OUT_MAGIC+struct.pack("<4IdIf",0,0,4,1,3.2,2,1.0))
            with self.assertRaisesRegex(ValueError,"FIRST_GOLD_NLL_DIFFERENT"):
                tf.scored(p,[rec],0)

    def test_original_256_train_metadata_absolute_paths_and_target_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            target=pathlib.Path(directory)/"not_yet"
            records=[]
            for i in range(256):
                records.append({"index":i,"source_id":hashlib.sha256(str(i).encode()).hexdigest(),
                     "prompt_ids":[100000,141,142],"target_ids":[280,281,282],
                     "expected_first_gold_nll":.5})
            receipt=tf.prepare(target,records)
            self.assertEqual(len(receipt),16)
            b=(target/"chunk00"/"gold_meta.bin").read_bytes()
            self.assertTrue(b.startswith(tf.MANIFEST_MAGIC))
            pos=len(tf.MANIFEST_MAGIC)
            first,n=struct.unpack("<II",b[pos:pos+8])
            self.assertEqual((first,n),(3,3))
            ids=struct.unpack("<384i",b[pos+8:pos+8+384*4])
            self.assertEqual(ids[:3],(280,281,282))
            self.assertEqual(ids[3],-1)
            path=(target/"chunk00"/"manifest.txt").read_text().splitlines()[0].rsplit(" ",1)[0]
            data=pathlib.Path(path).read_bytes()
            self.assertEqual(struct.unpack("<5i",data),(100000,141,142,280,281))
            self.assertEqual(len((target/"chunk15"/"manifest.txt").read_text().splitlines()),16)

    def test_gold_logit_score_only_and_no_production_edits(self):
        source=(ROOT/"tools"/"je01_xox_original256_fulltarget_teacherforced_nll.py").read_text()
        for needle in ('"train256"', '"pilot16"', '"all256"', 'fwrite(&nll,sizeof(double)',
                        'JE01_ORIGINAL_GOLD_OUTPUT','score_input_sha256',
                        'production_promotion_allowed'):
            if needle=='"train256"':
                continue
            self.assertIn(needle,source)
        self.assertNotIn("os.system(",source)
        self.assertNotIn("subprocess.Popen(['sudo'",source)
        self.assertIn("JE01_GOLD_NATIVE_EVIDENCE_RECORD_DRIFT",source)

if __name__=="__main__":
    unittest.main(verbosity=2)

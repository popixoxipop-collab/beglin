#!/usr/bin/env python3
"""JE01 synthetic source anchor contract; no need for real FFN data."""
import importlib.util,unittest
from pathlib import Path
f=Path(__file__).resolve().parents[1]/"tools/je01_eoe_original384_q45_scratch_stage.py"
s=importlib.util.spec_from_file_location("je01_q45",f)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class TestJE01Q45Scratch(unittest.TestCase):
    def setUp(self):
        self.cpp=(m.NATIVE_BIND_ENTRY+"\n"+
                  m.BIAS_PTR_OLD+"\n"+m.BIAS_LOAD_OLD+"\n"+
                  m.NATIVE_BIND_END+"\n"+
                  m.REG_ENTRY+"\n"+m.REGISTER_OLD+"\n"+m.REG_END+"\n"+
                  m.CPP_ENTRY+"\n"+m.BITS_OLD+"\n"+m.COUNT_OLD+"\n"+
                  m.STAT_OLD+"\n"+m.CHECK_OLD+"\n"+m.CPP_END+"\n")
    def test_q4_symmetric_and_original_affine(self):
        a=m.patch(self.cpp)
        self.assertIn(m.CHECK_NEW,a)
        self.assertIn(m.REGISTER_NEW,a)
        self.assertIn("bias_off==-1",a)
        self.assertIn("original affine AF4",a)
        self.assertEqual(a.count("Q4 symmetric"),1)
    def test_duplicate_fails(self):
        with self.assertRaisesRegex(ValueError,"EXPECTED_ONCE"):
            m.patch(self.cpp.replace(m.BIAS_PTR_OLD,m.BIAS_PTR_OLD+'\n'+m.BIAS_PTR_OLD))
    def test_missing_loader_refused(self):
        with self.assertRaises(ValueError):
            m.patch(self.cpp.replace(m.COUNT_OLD,""))
    def test_q4_loader_role_cannot_be_prepatched(self):
        with self.assertRaises(ValueError):
            m.patch(self.cpp.replace(m.BITS_OLD,m.BITS_NEW))
if __name__=="__main__":unittest.main(verbosity=2)

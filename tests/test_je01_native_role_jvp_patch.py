"""Native selector and fail-closed source checks, without claiming MLX execution."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PATCH_PATH = Path(__file__).resolve().parents[1] / "tools/je01_native_role_jvp_patch.py"
SPEC = importlib.util.spec_from_file_location("je01_role_patch_under_test", PATCH_PATH)
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)


class RolePatchContract(unittest.TestCase):
    def test_rejects_modified_base_source(self):
        with self.assertRaisesRegex(ValueError, "FIXED_BASE_SOURCE_SHA_DRIFT"):
            patch.apply("not the pinned C++ source", "not the pinned C source")

    def test_rejects_output_outside_new_research_root(self):
        with self.assertRaisesRegex(ValueError, "SEPARATE_D50_SOURCE_REQUIRED"):
            patch.stage(Path("/tmp/not_the_je01_role_jvp_root/source"))

    def test_actual_cpp_selector_accepts_frozen_map_and_rejects_drift(self):
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "A C++ compiler is required for this native guard test")
        start = patch.HELPERS.index("static bool je01_role_jvp_direction_matches_bank(void)")
        end = patch.HELPERS.index("static void je01_one_jvp_require_contract(void)", start)
        # Compile the exact C++ helper that will be inserted in the native
        # source, not a Python reimplementation of its branch conditions.
        predicate = patch.HELPERS[start:end]
        preamble = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
static int g_je01_direction_layer=-1,g_je01_direction_expert=-1,g_je01_direction_role=-1;
static std::string g_je01_direction_alternative;
'''
        program = r'''
static void configure(int selected,const char *tier) {
    const int slots[8][2]={{12,13},{13,22},{13,28},{14,63},{20,17},{21,61},{24,26},{24,60}};
    const char *roles[]={"gate_proj","up_proj","down_proj"};
    setenv("JE01_DIRECTION_ROLE",std::to_string(selected).c_str(),1);
    setenv("JE01_DIRECTION_TIER",tier,1);
    g_je01_direction_layer=slots[selected/3][0];
    g_je01_direction_expert=slots[selected/3][1];
    g_je01_direction_role=selected%3;
    g_je01_direction_alternative="je01.dir.L"+std::to_string(g_je01_direction_layer)+
        ".E"+std::to_string(g_je01_direction_expert)+"."+roles[selected%3]+"."+tier;
}
int main() {
    int valid=0,invalid=0;
    for(int selected=0;selected<24;++selected)for(const char *tier:{"Q4","Q5"}) {
        configure(selected,tier);
        if(!je01_role_jvp_direction_matches_bank())return 10;
        ++valid;
    }
    for(const char *value:{"-1","24","99999999999999999999999999","","00","+0"," 0","0x0","0x","0 "}) {
        configure(0,"Q4");setenv("JE01_DIRECTION_ROLE",value,1);
        if(je01_role_jvp_direction_matches_bank())return 11;
        ++invalid;
    }
    for(const char *value:{"Q3","Q6","q4","","Q4 "}) {
        configure(0,"Q4");setenv("JE01_DIRECTION_TIER",value,1);
        if(je01_role_jvp_direction_matches_bank())return 12;
        ++invalid;
    }
    configure(0,"Q4");unsetenv("JE01_DIRECTION_ROLE");
    if(je01_role_jvp_direction_matches_bank())return 13;++invalid;
    configure(0,"Q4");unsetenv("JE01_DIRECTION_TIER");
    if(je01_role_jvp_direction_matches_bank())return 14;++invalid;
    configure(0,"Q4");g_je01_direction_role=1;
    if(je01_role_jvp_direction_matches_bank())return 15;++invalid;
    configure(0,"Q4");g_je01_direction_layer=13;
    if(je01_role_jvp_direction_matches_bank())return 16;++invalid;
    configure(0,"Q4");g_je01_direction_expert=22;
    if(je01_role_jvp_direction_matches_bank())return 17;++invalid;
    configure(0,"Q4");g_je01_direction_alternative="je01.dir.L12.E13.gate_proj.Q5";
    if(je01_role_jvp_direction_matches_bank())return 18;++invalid;
    configure(1,"Q4");g_je01_direction_alternative="je01.dir.L12.E13.gate_proj.Q4";
    if(je01_role_jvp_direction_matches_bank())return 19;++invalid;
    configure(0,"Q4");setenv("JE01_DIRECTION_ROLE","3",1);
    if(je01_role_jvp_direction_matches_bank())return 20;++invalid;
    std::printf("VALID=%d INVALID=%d\n",valid,invalid);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="je01_role_guard_") as folder:
            source, binary = Path(folder) / "guard.cpp", Path(folder) / "guard"
            source.write_text(preamble + predicate + program)
            compiled = subprocess.run([compiler, "-std=c++17", "-O1", str(source), "-o", str(binary)],
                                      text=True, capture_output=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            checked = subprocess.run([str(binary)], text=True, capture_output=True, timeout=10)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(checked.stdout.strip(), "VALID=48 INVALID=23")


if __name__ == "__main__":
    unittest.main()

import sys, unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "oml_acs"))
from hf_adapter import ModelSpec, AdapterContractError, blocks, PinnedHF, PILOT

class HFContractTests(unittest.TestCase):
    def test_models_and_layers_match_paper_small_scale(self):
        self.assertEqual(PILOT["Qwen/Qwen2.5-1.5B-Instruct"],(28,14))
        self.assertEqual(PILOT["meta-llama/Llama-3.2-3B-Instruct"],(28,14))
        self.assertEqual(PILOT["microsoft/Phi-4-mini-instruct"],(32,16))
    def test_revision_mandatory(self):
        with self.assertRaises(AdapterContractError):
            ModelSpec("Qwen/Qwen2.5-1.5B-Instruct","main",28,14).validate()
    def test_valid_pinned_revision(self):
        self.assertIsNotNone(ModelSpec("Qwen/Qwen2.5-1.5B-Instruct","a"*40,28,14).validate())
    def test_layer_drifts_rejected(self):
        with self.assertRaises(AdapterContractError):
            ModelSpec("Qwen/Qwen2.5-1.5B-Instruct","a"*40,28,13).validate()
    def test_unsupported_model_rejected(self):
        with self.assertRaises(AdapterContractError):
            ModelSpec("other/model","a"*40,28,14).validate()
    def test_unknown_decoder_block_fails_closed(self):
        with self.assertRaises(AdapterContractError):
            blocks(SimpleNamespace(model=SimpleNamespace(decoder=[])))
    def test_no_implicit_checkpoint_download(self):
        with self.assertRaises(AdapterContractError):
            PinnedHF(ModelSpec("Qwen/Qwen2.5-1.5B-Instruct","a"*40,28,14),
                     local_files_only=False,explicitly_allow_download=False)
if __name__=="__main__":
    unittest.main(verbosity=2)

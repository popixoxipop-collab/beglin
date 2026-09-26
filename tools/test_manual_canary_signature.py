#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

import manual_canary_contract as mc
import manual_canary_signature as ms
import manual_canary_production_intent as mpi


SSH_KEYGEN=Path("/usr/bin/ssh-keygen")
NOW=datetime(2026,9,26,4,30,tzinfo=timezone.utc)
BASELINE=[{"role":"shared_up_proj","layer":3,"n":5}]
CANDIDATE=[{"role":"shared_up_proj","layer":3,"n":6}]


def h(ch): return ch*64


def proposal(**overrides):
    p={
      "mode":"dry_run","production_write_allowed":False,
      "proposal_id":"p-real","proposer":"planner-agent",
      "environment_id":"xox-vdsp-gpu-precision",
      "model_revision":"deepseek-v2-lite","backend":"mlx_metal","architecture":"mla",
      "source_commit":"d9401d774691cfcc4b2f74b11f9098c96b22c59b",
      "binary_sha256":"6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392",
      "checkpoint_sha256":"1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898",
      "baseline_policy_hash":mc.sha256_json(BASELINE),
      "candidate_policy_hash":mc.sha256_json(CANDIDATE),
      "single_target":{"role":"shared_up_proj","layer":3,"before_n":5,"after_n":6},
      "evidence_refs":[
        {"kind":"G4_A_B_R","run_id":"f-real","sha256":h("a")},
        {"kind":"G6_RESTART_CANARY","run_id":"f-real","sha256":h("b")},
      ],
      "budget":{"max_requests":50,"max_tokens":1000,"max_duration_ms":10000,"max_memory_bytes":1024*1024},
      "expected_epoch":7,"restart_instance_id":"worker-pre",
      "kill_switch_scope":"single-target","rollback_plan":"restore exact runtime baseline preimage",
    }
    p.update(overrides); return p


def metadata(**overrides):
    m={
      "approval_id":"human-a1","issuer":"human-reviewer","principal":"human-reviewer",
      "issued_at":NOW.isoformat(),"expires_at":(NOW+timedelta(minutes=10)).isoformat(),
      "nonce":"nonce-human-1","mode":"dry_run","production_write_allowed":False,
    }
    m.update(overrides); return m


def create_signer(root:Path):
    if not SSH_KEYGEN.is_file():
        raise unittest.SkipTest("/usr/bin/ssh-keygen unavailable")
    key=root/"id_ed25519"
    subprocess.run([str(SSH_KEYGEN),"-q","-t","ed25519","-N","","-f",str(key)],check=True)
    pub=key.with_suffix(".pub").read_text().split()
    allowed=root/"allowed_signers"
    allowed.write_text(f"human-reviewer {pub[0]} {pub[1]}\n")
    return key,allowed


def sign(root:Path,key:Path,payload:bytes):
    msg=root/"approval.json"
    msg.write_bytes(payload)
    subprocess.run([str(SSH_KEYGEN),"-Y","sign","-f",str(key),"-n",ms.SSH_NAMESPACE,str(msg)],check=True,capture_output=True)
    return (root/"approval.json.sig").read_text()


class SignatureTests(unittest.TestCase):
    def test_real_openssh_signature_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=proposal(); m=metadata(); key,allowed=create_signer(root)
            sig=sign(root,key,ms.signing_payload(p,m))
            got=ms.verify_human_signature(proposal=p,approval_metadata=m,signature_text=sig,allowed_signers_path=allowed,now=NOW)
            self.assertEqual(got["status"],"VERIFIED")
            self.assertFalse(got["production_write_allowed"])
            self.assertEqual(got["principal"],"human-reviewer")

    def test_tampered_proposal_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=proposal(); m=metadata(); key,allowed=create_signer(root)
            sig=sign(root,key,ms.signing_payload(p,m))
            p2=proposal(binary_sha256=h("f"))
            with self.assertRaises(ms.HumanSignatureError):
                ms.verify_human_signature(proposal=p2,approval_metadata=m,signature_text=sig,allowed_signers_path=allowed,now=NOW)

    def test_wrong_principal_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=proposal(); m=metadata(); key,allowed=create_signer(root)
            sig=sign(root,key,ms.signing_payload(p,m))
            m2=metadata(issuer="other",principal="other")
            with self.assertRaises(ms.HumanSignatureError):
                ms.verify_human_signature(proposal=p,approval_metadata=m2,signature_text=sig,allowed_signers_path=allowed,now=NOW)

    def test_expired_signature_metadata_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=proposal(); m=metadata(expires_at=(NOW+timedelta(seconds=1)).isoformat()); key,allowed=create_signer(root)
            sig=sign(root,key,ms.signing_payload(p,m))
            with self.assertRaises(ms.HumanSignatureError):
                ms.verify_human_signature(proposal=p,approval_metadata=m,signature_text=sig,allowed_signers_path=allowed,now=NOW+timedelta(seconds=1))

    def test_nonce_reuse_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=proposal(); m=metadata(); key,allowed=create_signer(root)
            sig=sign(root,key,ms.signing_payload(p,m))
            with self.assertRaises(ms.HumanSignatureError):
                ms.verify_human_signature(proposal=p,approval_metadata=m,signature_text=sig,allowed_signers_path=allowed,now=NOW,consumed_nonces={m["nonce"]})

    def test_self_approval_fails_before_verification(self):
        p=proposal(proposer="human-reviewer")
        with self.assertRaises(ms.HumanSignatureError):
            ms.signing_payload(p,metadata())

    def test_symlink_allowed_signers_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); real=root/"real"; real.write_text("x\n"); link=root/"link"; link.symlink_to(real)
            with self.assertRaises(ms.HumanSignatureError):
                ms._validate_allowed_signers(link)


class IntentTests(unittest.TestCase):
    def verified(self,p=None):
        p=p or proposal()
        return {
          "schema":"manual-canary-human-signature-verification-v1","status":"VERIFIED",
          "production_write_allowed":False,"proposal_digest":mc.proposal_digest(p),
          "approval_id":"a","principal":"human-reviewer",
          "signed_payload_sha256":h("c"),"signature_sha256":h("d"),"allowed_signers_sha256":h("e"),
        }

    def runtime(self,**overrides):
        r={"active_policy":BASELINE,"active_policy_hash":mc.sha256_json(BASELINE),
           "weight_epoch":7,"ack_sha256":h("f"),"worker_instance_id":"worker-pre"}
        r.update(overrides); return r

    def test_seal_ready_for_external_review_but_not_executable(self):
        p=proposal()
        got=mpi.seal_production_intent(proposal=p,verified_signature=self.verified(p),runtime_preimage=self.runtime(),candidate_policy=CANDIDATE)
        self.assertEqual(got["status"],"READY_FOR_EXTERNAL_REVIEW")
        self.assertFalse(got["execution_enabled"])
        self.assertFalse(got["production_write_allowed"])
        self.assertEqual(len(got["intent_sha256"]),64)

    def test_signature_digest_mismatch_fails(self):
        p=proposal(); sig=self.verified(p); sig["proposal_digest"]=h("0")
        with self.assertRaises(mpi.ProductionIntentError):
            mpi.seal_production_intent(proposal=p,verified_signature=sig,runtime_preimage=self.runtime(),candidate_policy=CANDIDATE)

    def test_runtime_policy_hash_mismatch_fails(self):
        p=proposal()
        with self.assertRaises(mpi.ProductionIntentError):
            mpi.seal_production_intent(proposal=p,verified_signature=self.verified(p),runtime_preimage=self.runtime(active_policy_hash=h("0")),candidate_policy=CANDIDATE)

    def test_runtime_epoch_mismatch_fails(self):
        p=proposal()
        with self.assertRaises(mpi.ProductionIntentError):
            mpi.seal_production_intent(proposal=p,verified_signature=self.verified(p),runtime_preimage=self.runtime(weight_epoch=8),candidate_policy=CANDIDATE)

    def test_runtime_instance_mismatch_fails(self):
        p=proposal()
        with self.assertRaises(mpi.ProductionIntentError):
            mpi.seal_production_intent(proposal=p,verified_signature=self.verified(p),runtime_preimage=self.runtime(worker_instance_id="other"),candidate_policy=CANDIDATE)

    def test_candidate_multitarget_fails(self):
        p=proposal()
        candidate=CANDIDATE+[{"role":"shared_down_proj","layer":4,"n":6}]
        with self.assertRaises(mpi.ProductionIntentError):
            mpi.seal_production_intent(proposal=p,verified_signature=self.verified(p),runtime_preimage=self.runtime(),candidate_policy=candidate)

    def test_execute_is_unconditionally_disabled(self):
        with self.assertRaises(mpi.ProductionBridgeDisabled):
            mpi.execute_production_intent({"anything":"ignored"})


if __name__=="__main__":
    unittest.main()

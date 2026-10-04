#!/usr/bin/env python3
import unittest

import model_capability as mc
import model_capability_bridge as mcb
import precision_p10_canary_gate as p10
import precision_p11_production_gate as p11
import precision_p9_shadow_cert as p9
import precision_policy_refresh as p8


CURRENT=[
    {"role":"shared_up_proj","layer":3,"n":6},
    {"role":"shared_down_proj","layer":26,"n":5},
]
def cand(role,layer,n,feasible=True):
    return {
        "role":role,"layer":layer,"n":n,"feasible":feasible,
        "conditionally_feasible":False,"conditional_recoveries":[],
        "blocked_reasons":[] if feasible else ["FAIL"],
        "real_pass_events":10,"real_fail_events":0,"real_event_count":10,
        "real_pass_event_keys":[],"real_fail_event_keys":[],
        "g4_context_hash":f"g4-{role}-{layer}-{n}",
        "g6_context_hash":f"g6-{role}-{layer}-{n}",
        "tensor_count":1,"numel":1000,"effective_bpw":float(n),
        "estimated_bytes":float(n*1000),"persistent_p50_ms":None,
        "persistent_p95_ms":None,"persistent_rss_bytes":None,"benchmark_pid":None,
    }

CANDS=[
    cand("shared_up_proj",3,5),
    cand("shared_up_proj",3,6),
    cand("shared_down_proj",26,5),
    cand("shared_down_proj",26,6),
]
SUMMARY={"admissions":600,"finite_logits_rate":1.0,"lineage_head_sha256":"a"*64}
P7={"status":"PASS","production_touched":False,"result_sha256":"b"*64}


def capability_bundle():
    target="synthetic/L3/shared_up"
    bundle={
        "schema":"beglin-model-capability-bundle-v1",
        "model_id":"synthetic",
        "checkpoint_identity_sha256":"c"*64,
        "production_write_allowed":False,
        "automatic_live_promotion":False,
        "tensor_role_graph":{
            "nodes":[{
                "role":"SHARED_UP","layer":3,"expert_id":None,
                "canonical_target_key":target,
            }]
        },
        "backend_capability_matrix":{
            "rows":[{
                "target_key":target,"backend":"mlx_metal",
                "inference_status":"IMPLEMENTED_UNVERIFIED",
                "qng64_status":"IMPLEMENTED_UNVERIFIED",
                "supported_n":[5,6],"validation_required":True,
            }]
        },
        "runtime_mutation_matrix":{
            "rows":[{
                "target_key":target,"backend":"mlx_metal",
                "mutation_mode":"IMPLEMENTED_UNVERIFIED",
                "allowed_target_precisions":[5,6],
            }]
        },
        "p8_p11_eligibility":{
            "status":"PARTIAL","p8_allowed":True,"p9_allowed":True,
            "p10_allowed":True,"p11_allowed":False,
            "automatic_live_promotion":False,
        },
    }
    bundle["bundle_sha256"]=mc.stable_identity_sha256(bundle)
    return bundle


class CapabilityBridgeTests(unittest.TestCase):
    def test_valid_binding_canonicalizes_engine_role(self):
        bundle=capability_bundle()
        got=mcb.validate_p8_target(
            bundle=bundle,role="shared_up_proj",layer=3,target_n=5,
            backend="mlx_metal")
        self.assertEqual(got["canonical_role"],"SHARED_UP")
        self.assertEqual(got["capability_target_key"],"synthetic/L3/shared_up")
        self.assertEqual(got["model_capability_bundle_sha256"],bundle["bundle_sha256"])

    def test_unverified_mutation_forces_validation_even_when_backend_is_verified(self):
        bundle=capability_bundle()
        bundle["backend_capability_matrix"]["rows"][0].update({
            "inference_status":"VERIFIED",
            "qng64_status":"VERIFIED",
            "validation_required":False,
        })
        bundle["bundle_sha256"]=mc.stable_identity_sha256(bundle)
        got=mcb.validate_p8_target(
            bundle=bundle,role="shared_up_proj",layer=3,target_n=5,
            backend="mlx_metal")
        self.assertEqual(got["mutation_mode"],"IMPLEMENTED_UNVERIFIED")
        self.assertTrue(got["requires_validation"])

    def test_tampered_bundle_is_rejected(self):
        bundle=capability_bundle()
        bundle["model_id"]="tampered"
        with self.assertRaisesRegex(mcb.CapabilityBridgeError,"SHA mismatch"):
            mcb.validate_p8_target(
                bundle=bundle,role="shared_up_proj",layer=3,target_n=5)

    def test_unknown_target_or_precision_is_rejected(self):
        bundle=capability_bundle()
        with self.assertRaisesRegex(mcb.CapabilityBridgeError,"exactly one"):
            mcb.validate_p8_target(
                bundle=bundle,role="shared_down_proj",layer=26,target_n=5)
        with self.assertRaisesRegex(mcb.CapabilityBridgeError,"unsupported"):
            mcb.validate_p8_target(
                bundle=bundle,role="shared_up_proj",layer=3,target_n=13)

    def test_p8_to_p10_lineage_is_stable(self):
        bundle=capability_bundle()
        proposal=p8.propose(
            candidates=CANDS,current_policy=CURRENT,lineage_summary=SUMMARY,
            p7_certification=P7,model_capability_bundle=bundle)
        self.assertEqual(
            proposal["model_capability_bundle_sha256"],bundle["bundle_sha256"])
        self.assertEqual(proposal["capability_target_key"],"synthetic/L3/shared_up")
        self.assertEqual(proposal["capability_backend"],"mlx_metal")

        waiting=p9.queue_item(proposal=proposal,provenance=None)
        self.assertEqual(
            waiting["model_capability_bundle_sha256"],bundle["bundle_sha256"])
        self.assertEqual(waiting["capability_target_key"],proposal["capability_target_key"])

        target=proposal["selected_shadow_target"]
        admitted={
            "shadow_status":"SHADOW_ADMITTED","production_write_allowed":False,
            "candidate":{
                "role":target["role"],"layer":target["layer"],"n":target["new_n"]
            },
            "result_sha256":"d"*64,
        }
        cert=p8.certification_candidate(proposal=proposal,shadow_result=admitted)
        self.assertEqual(cert["model_capability_bundle_sha256"],bundle["bundle_sha256"])

        final=p10.final_gate(
            p9_bundle=cert,
            canary_pass={"state":"CANARY_PASS_REVIEW_REQUIRED"},
            rollback_drill={"state":"ROLLBACK_VERIFIED"},
        )
        self.assertEqual(final["model_capability_bundle_sha256"],bundle["bundle_sha256"])
        self.assertEqual(final["capability_target_key"],"synthetic/L3/shared_up")

    def test_p9_certification_rejects_shadow_from_other_capability_bundle(self):
        bundle=capability_bundle()
        proposal=p8.propose(
            candidates=CANDS,current_policy=CURRENT,lineage_summary=SUMMARY,
            p7_certification=P7,model_capability_bundle=bundle)
        candidate=p8.shadow_candidate(proposal)
        repeated={
            "shadow_status":"SHADOW_ADMITTED",
            "production_write_allowed":False,
            "automatic_live_promotion":False,
            "candidate":candidate,
            "result_sha256":"d"*64,
            "repeats":2,
            "statuses":["SHADOW_ADMITTED","SHADOW_ADMITTED"],
            "model_capability_bundle_sha256":"f"*64,
            "capability_target_key":proposal["capability_target_key"],
            "capability_backend":proposal["capability_backend"],
        }
        with self.assertRaisesRegex(p9.P9Error,"capability lineage mismatch"):
            p9.certification_bundle(
                proposal=proposal,repeated_shadow=repeated)

    def test_p11_approval_request_carries_optional_lineage(self):
        bundle=capability_bundle()
        plan={
            "schema":"beglin-precision-p11-production-cutover-plan-v1",
            "status":"AWAITING_TRUSTED_PRODUCTION_APPROVAL",
            "preimage_sha256":"1"*64,
            "p10_result_sha256":"2"*64,
            "executor_source_sha256":"3"*64,
            "production_binary_rebind_compat_sha256":"4"*64,
            "expected_live_preimage":{
                "route_generation":6,
                "route_manifest_sha256":"5"*64,
                "worker_pid":99,
                "weight_epoch":7,
                "ack_sha256":"6"*64,
            },
            "target":{"runtime_policy_hash":"7"*64},
            "model_capability_bundle_sha256":bundle["bundle_sha256"],
            "capability_target_key":"synthetic/L3/shared_up",
            "capability_backend":"mlx_metal",
        }
        plan["cutover_plan_sha256"]=p11.sha256_json(plan)
        req=p11.build_approval_request(plan)
        self.assertEqual(
            req["model_capability_bundle_sha256"],bundle["bundle_sha256"])
        self.assertEqual(req["capability_target_key"],"synthetic/L3/shared_up")
        self.assertEqual(req["capability_backend"],"mlx_metal")

    def test_legacy_paths_remain_without_p12_fields(self):
        p9_bundle={
            "status":"MANUAL_REVIEW_CANDIDATE",
            "production_write_allowed":False,
            "automatic_live_promotion":False,
        }
        out=p10.final_gate(
            p9_bundle=p9_bundle,
            canary_pass={"state":"CANARY_PASS_REVIEW_REQUIRED"},
            rollback_drill={"state":"ROLLBACK_VERIFIED"},
        )
        self.assertNotIn("model_capability_bundle_sha256",out)

        plan={
            "schema":"beglin-precision-p11-production-cutover-plan-v1",
            "status":"AWAITING_TRUSTED_PRODUCTION_APPROVAL",
            "preimage_sha256":"1"*64,
            "p10_result_sha256":"2"*64,
            "executor_source_sha256":"3"*64,
            "production_binary_rebind_compat_sha256":"4"*64,
            "expected_live_preimage":{
                "route_generation":6,
                "route_manifest_sha256":"5"*64,
                "worker_pid":99,
                "weight_epoch":7,
                "ack_sha256":"6"*64,
            },
            "target":{"runtime_policy_hash":"7"*64},
        }
        plan["cutover_plan_sha256"]=p11.sha256_json(plan)
        req=p11.build_approval_request(plan)
        self.assertNotIn("model_capability_bundle_sha256",req)


if __name__=="__main__":
    unittest.main()

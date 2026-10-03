"""P6 journal integrity, bounded hot path and complete admission coverage."""
import copy
import json
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import precision_observability as po
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
import precision_closed_loop as pcl
from test_gpu_precision_observability import decision, result


def steady():
    r = result(transitioned=False)
    r['precision_epoch'].update(before_policy_hash='2'*64, after_policy_hash='2'*64,
                                before_epoch=3, after_epoch=3)
    return r


class JournalGuards(unittest.TestCase):
    def test_discontinuity_rejected_without_poisoning_journal(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'lineage.jsonl'
            obs = po.PrecisionObservability(lineage_path=p)
            obs.record(admission_id='a', worker_pid=7, request_count=1, decision=decision(), result=result())
            saved = p.read_bytes()
            with self.assertRaises(po.PrecisionObservabilityError):
                obs.record(admission_id='bad', worker_pid=7, request_count=1,
                           decision=decision(), result=result())
            self.assertEqual(p.read_bytes(), saved)
            obs.record(admission_id='b', worker_pid=7, request_count=1,
                       decision=decision(), result=steady())
            self.assertEqual(obs.summary()['admissions'], 2)

    def test_normal_append_does_not_reread_entire_journal(self):
        with tempfile.TemporaryDirectory() as td:
            obs = po.PrecisionObservability(lineage_path=Path(td)/'l.jsonl')
            with patch.object(obs, 'read_records', side_effect=AssertionError('hot path reread')):
                for i in range(25):
                    obs.record(admission_id=str(i), worker_pid=1, request_count=1,
                               decision=decision(False), result=steady())
                self.assertEqual(obs.summary(verify=False)['admissions'], 25)
            self.assertEqual(obs.summary(), po.summarize(obs.read_records()))

    def test_two_cooperating_writers_share_one_valid_chain(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'l.jsonl'
            observers = [po.PrecisionObservability(lineage_path=p, identity={'worker_instance_id': 'same'})
                         for _ in range(2)]
            errors = []
            def run(worker, prefix):
                try:
                    for i in range(8):
                        worker.record(admission_id=f'{prefix}-{i}', worker_pid=1, request_count=1,
                                      decision=decision(False), result=steady())
                except Exception as exc:
                    errors.append(exc)
            threads = [threading.Thread(target=run, args=(obs, n)) for n,obs in enumerate(observers)]
            for t in threads: t.start()
            for t in threads: t.join(5)
            self.assertFalse(any(t.is_alive() for t in threads))
            self.assertEqual(errors, [])
            rows = observers[0].read_records()
            po.verify_records(rows)
            self.assertEqual(len(rows), 16)

    def test_snapshot_cannot_overwrite_lineage(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'l.jsonl'
            with self.assertRaises(po.PrecisionObservabilityError):
                po.PrecisionObservability(lineage_path=p, snapshot_path=p)

    def test_snapshot_failure_does_not_discard_committed_record(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'l.jsonl'
            obs=po.PrecisionObservability(lineage_path=p)
            with patch.object(obs, '_write_snapshot', side_effect=OSError('disk error')):
                got=obs.record(admission_id='a',worker_pid=1,request_count=1,
                               decision=decision(),result=result())
            self.assertEqual(got['status'], 'RECORDED')
            self.assertEqual(got['snapshot_status'], 'ERROR')
            self.assertEqual(obs.sink_error_count, 1)
            restored=po.PrecisionObservability(lineage_path=p)
            self.assertEqual(restored.summary()['admissions'], 1)

    def test_unknown_cost_is_not_measured_zero(self):
        r=po.build_record(admission_id='x',worker_pid=1,request_count=1,decision={},
                          result={},prev_record_sha256=None)
        summary=po.summarize([r])
        self.assertIsNone(r['actual_cost']['roundtrip_ms'])
        self.assertIsNone(summary['actual_roundtrip_ms_mean'])
        self.assertIsNone(summary['cache_hit_rate'])
        self.assertEqual(summary['missing_measurement_counts']['cache_hits'], 1)

    def test_arbitrary_signal_payload_is_not_persisted(self):
        d=decision()
        d['signal']['raw']={'prompt':'PRIVATE_CONTENT'}
        d['signal']['prompt']='PRIVATE_CONTENT'
        r=po.build_record(admission_id='x',worker_pid=1,request_count=1,decision=d,
                          result=result(),prev_record_sha256=None)
        self.assertNotIn('PRIVATE_CONTENT',json.dumps(r))
        self.assertNotIn('responses',r['outcome'])

    def test_invalid_numbers_rejected_before_append(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'l.jsonl'; obs=po.PrecisionObservability(lineage_path=p)
            for value in [float('nan'),float('inf'),-1,True,1.2]:
                r=result(); r['precision_epoch']['inference_passes']=value
                with self.subTest(value=value), self.assertRaises(po.PrecisionObservabilityError):
                    obs.record(admission_id='x',worker_pid=1,request_count=1,decision=decision(),result=r)
            self.assertFalse(p.exists())

    def test_incomplete_tail_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'l.jsonl'; p.write_text('{"schema":')
            with self.assertRaises(po.PrecisionObservabilityError):
                po.PrecisionObservability(lineage_path=p)


class AdmissionCoverage(unittest.TestCase):
    def worker(self, td):
        worker=ps.PersistentRouteWorker(route=base.candidate_route(),root=Path(td)/'worker')
        worker.ack_path.parent.mkdir(parents=True)
        worker.ack_path.write_text(json.dumps({
            'schema':'gpu-precision-applied-v1','backend':'mlx_metal','correction_mode':'off',
            'status':'PROMOTION_APPLIED','weight_epoch':1,
            'active_policy':[{'role':'shared_up_proj','layer':3,'n':6}],
        }))
        worker.precision_observability=po.PrecisionObservability(lineage_path=Path(td)/'l.jsonl')
        return worker

    def test_rejected_decision_logged_without_fake_inference(self):
        with tempfile.TemporaryDirectory() as td:
            w=self.worker(td)
            w.precision_closed_loop_engine=Mock()
            w.precision_closed_loop_engine.decide.side_effect=pcl.PrecisionClosedLoopError('PRIVATE_REASON')
            with self.assertRaisesRegex(pcl.PrecisionClosedLoopError,'PRIVATE_REASON'):
                w.submit_with_closed_loop_precision([([1],1)],signal={'high_entropy':True,'entropy':.3},admission_id='reject')
            rows=w.precision_observability.read_records()
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['outcome']['status'],'REJECTED')
            self.assertEqual(rows[0]['actual_cost']['inference_passes'],0)
            self.assertNotIn('PRIVATE_REASON',json.dumps(rows))
            summary=w.precision_observability.summary()
            self.assertEqual(summary['rejected_admissions'],1)
            self.assertEqual(summary['policy_residency_admissions'],{})
            self.assertIsNone(summary['extra_pass_rate'])

    def test_execution_error_is_not_a_missing_success_row(self):
        with tempfile.TemporaryDirectory() as td:
            w=self.worker(td)
            with self.assertRaises(ps.PersistentSupervisorError):
                w.submit([([1],1)])
            self.assertEqual(w.precision_observability.summary()['error_admissions'],1)

    def test_nested_paths_make_one_record(self):
        with tempfile.TemporaryDirectory() as td:
            w=self.worker(td)
            @ps._precision_observation_guard('direct')
            def child(w, parsed):
                return {'responses':[[2]],'finite_logits':True,'inference_passes':1,
                        'engine_wall_ms':1.,'roundtrip_ms':2.}
            @ps._precision_observation_guard('adaptive')
            def parent(w, parsed):
                return child(w, parsed)
            got=parent(w,[([1],1)])
            self.assertEqual(got['responses'],[[2]])
            summary=w.precision_observability.summary()
            self.assertEqual(summary['admissions'],1)
            self.assertEqual(summary['admission_path_counts'],{'adaptive':1})

    def test_sink_error_does_not_change_returned_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            w=self.worker(td)
            @ps._precision_observation_guard('direct')
            def child(w, parsed):
                return {'responses':[[123]],'finite_logits':True,'inference_passes':1}
            with patch.object(w.precision_observability,'record',side_effect=OSError('full')):
                got=child(w,[([1],1)])
            self.assertEqual(got['responses'],[[123]])
            self.assertEqual(got['precision_observability']['status'],'ERROR')
            self.assertEqual(w.health()['precision_observability']['sink_error_count'],1)

if __name__=='__main__':
    unittest.main()

class ReadOnlyAPITests(unittest.TestCase):
    def test_unconfigured_pool_reports_not_configured(self):
        pool=ps.PersistentWorkerPool()
        snapshot=pool.observability_snapshot()
        self.assertTrue(snapshot['read_only'])
        self.assertTrue(all(row['status']=='NOT_CONFIGURED' for row in snapshot['workers'].values()))

    def test_handler_reads_snapshot_without_inference(self):
        fake=Mock()
        fake.path='/observability'
        fake.server.pool.observability_snapshot.return_value={'read_only':True}
        ps.PersistentHandler.do_GET(fake)
        fake.server.pool.observability_snapshot.assert_called_once_with()
        fake._send.assert_called_once_with(ps.HTTPStatus.OK,{'read_only':True})
        fake.server.executor.generate_batch.assert_not_called()

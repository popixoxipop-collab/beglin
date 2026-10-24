#!/usr/bin/env python3
"""Read-only fail-closed G6 Node TAP triage; no model/evaluation or release access."""
import argparse,collections,hashlib,json,pathlib,re
SUMMARY_RE=re.compile(r'^\s*(?:ℹ|#)\s*(tests|pass|fail|cancelled|skipped|todo|duration_ms)\s+([0-9]+(?:\.[0-9]+)?)\s*$')
FAILED_RE=re.compile(r'^\s*(?:✖\s+|not ok\s+\d+\s+-\s+)(.+?)(?:\s+\([0-9.]+ms\))?\s*$')
PASSED_RE=re.compile(r'^\s*(?:✔\s+|ok\s+\d+\s+-\s+)(.+?)(?:\s+\([0-9.]+ms\))?\s*$')
def analyze(raw_log:bytes,receipt=None):
    log=raw_log.decode('utf-8',errors='replace')
    totals={};failures=[];passes=[]
    for line in log.splitlines():
        m=SUMMARY_RE.match(line)
        if m:totals[m.group(1)]=float(m.group(2)) if m.group(1)=='duration_ms' else int(m.group(2))
        f=FAILED_RE.match(line);p=PASSED_RE.match(line)
        if f:failures.append(f.group(1).strip())
        if p:passes.append(p.group(1).strip())
    complete_counts=all(k in totals for k in ('tests','pass','fail','cancelled','skipped'))
    consistent=complete_counts and totals['tests']==sum(totals.get(k,0) for k in ('pass','fail','cancelled','skipped','todo'))
    complete=consistent and 'duration_ms' in totals and totals['tests']>0
    observed_errors=bool(failures or totals.get('fail',0) or totals.get('cancelled',0))
    digest=hashlib.sha256(raw_log).hexdigest()
    categories=collections.Counter('file_failure' if n.startswith('/') or '/tests/' in n else 'test_case_failure' for n in failures)
    record={'schema':'BEGLIN-Qv20-isolated-G6-unit-triage/2','log_sha256':digest,
        'totals':totals,'summary_consistent':bool(consistent),'summary_complete':bool(complete),
        'failure_count_visible':len(failures),'pass_count_visible':len(passes),
        'failure_categories':dict(categories),'failure_titles_first_25':failures[:25],
        'failure_signatures':{
            'pending_promise':'Promise resolution is still pending' in log,
            'async_after_test':'asynchronous activity after the test ended' in log,
            'child_startup_timeout':'child startup timeout' in log,
            'assertion_failure':'AssertionError' in log,
            'timeout_like':bool(re.search(r'(?:test|Test).{0,40}timeout|timed out',log,re.I))},
        'receipt_provided':receipt is not None,
        'verdict':'FAIL' if observed_errors else 'LOG_PASS_UNATTESTED' if complete else 'UNKNOWN_INCOMPLETE',
        'no_policy_fit_or_heldout_access':True,'no_release_or_execution_change':True}
    if receipt is not None:
        try:
            ok=(receipt.get('schema')=='tc-development-test-receipt/1'
                and receipt.get('profile')=='unit' and receipt.get('stdout_sha256')==digest)
            record['receipt_state']=receipt.get('state')
            record['receipt_exit_code']=receipt.get('exit_code')
            record['receipt_stdout_hash_matched']=ok
            if not ok:record['verdict']='EVIDENCE_MISMATCH'
            elif receipt.get('state')!='PASS' or receipt.get('exit_code')!=0:record['verdict']='FAIL'
            elif not complete or observed_errors:record['verdict']='EVIDENCE_MISMATCH'
            else:record['verdict']='PASS'
        except (TypeError,AttributeError):record['verdict']='EVIDENCE_MISMATCH'
    return record
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--log',type=pathlib.Path,required=True)
    p.add_argument('--receipt',type=pathlib.Path)
    a=p.parse_args()
    if not a.log.is_file():
        print(json.dumps({'verdict':'UNKNOWN_INCOMPLETE','reason':'missing_log'}));return 3
    raw=a.log.read_bytes();receipt=None
    if a.receipt is not None:
        if not a.receipt.is_file():
            print(json.dumps({'verdict':'EVIDENCE_MISMATCH','reason':'missing_receipt'}));return 3
        try:receipt=json.loads(a.receipt.read_text('utf-8'))
        except (ValueError,OSError):
            print(json.dumps({'verdict':'EVIDENCE_MISMATCH','reason':'invalid_receipt'}));return 3
    result=analyze(raw,receipt)
    print(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2))
    return 0 if result['verdict']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())

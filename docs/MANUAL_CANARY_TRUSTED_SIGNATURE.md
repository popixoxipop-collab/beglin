# Trusted Human Signature + Non-Executable Production Intent

Agent E's next layer verifies a human authorization cryptographically while
keeping production mutation disabled.

## Signature format

The verifier uses OpenSSH detached signatures:

```text
namespace: beglin-manual-canary-v1
trust anchor: OpenSSH allowed_signers file
identity: approval principal
payload: canonical JSON containing proposal_digest + approval metadata
```

The signer's private key is never stored by Beglin, committed to GitHub, or
loaded by the canary controller. A trusted operator signs externally.

Example operator flow:

```bash
ssh-keygen -Y sign -f <operator-private-key> \
  -n beglin-manual-canary-v1 approval-payload.json
```

The service side receives only the armored signature, public allowed-signers
trust anchor, proposal and approval metadata.

## Fail closed

Verification rejects:

- self approval
- proposal tampering after signing
- issuer/principal mismatch
- wrong signer/principal
- expired or not-yet-valid approval
- nonce reuse
- symlinked/invalid trust-anchor path
- non-dry-run or production_write_allowed approvals

A successful verification emits hashes for the proposal, signed payload,
signature and allowed-signers file.

## Production intent sealing

`seal_production_intent()` additionally requires the launch-time runtime ACK:

- active_policy and active_policy_hash
- weight_epoch
- ack_sha256
- worker_instance_id

It rechecks the exact before_n preimage, single-target candidate delta,
candidate policy hash, epoch and restart instance.

The output is:

```text
status = READY_FOR_EXTERNAL_REVIEW
execution_enabled = false
production_write_allowed = false
```

`execute_production_intent()` always raises `ProductionBridgeDisabled`.

This layer therefore proves approval integrity and runtime-context binding
without creating a production mutation path. A real adapter bridge remains a
separate review/deployment.

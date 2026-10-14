"""Pure research decision gate. This module grants no execution/quality authority.

Callers must supply independently verified policy permissions and runtime state.
No network, filesystem, native runtime or production configuration is touched.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import FrozenSet, Optional

SHA = re.compile(r'^[a-f0-9]{64}$')


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()


def require_sha(value: str) -> None:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError('invalid SHA-256 identity')


@dataclass(frozen=True)
class Runtime:
    policy: str
    epoch: int
    worker: str
    context: str

    def __post_init__(self):
        require_sha(self.policy)
        require_sha(self.context)
        if type(self.epoch) is not int or self.epoch < 0 or not self.worker:
            raise ValueError('invalid runtime epoch/worker')


@dataclass(frozen=True)
class Forecast:
    from_policy: str
    to_policy: str
    context: str
    evidence_sha256: str
    gain_lower_ms: float
    switch_upper_ms: float
    planner_upper_ms: float
    horizon: int
    valid_through: int

    def __post_init__(self):
        for value in (self.from_policy, self.to_policy, self.context, self.evidence_sha256):
            require_sha(value)
        for x in (self.gain_lower_ms, self.switch_upper_ms, self.planner_upper_ms):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError('forecast must contain finite measured bounds')
        if min(self.switch_upper_ms, self.planner_upper_ms) < 0:
            raise ValueError('negative costs are forbidden')
        if type(self.horizon) is not int or not 1 <= self.horizon <= 4096:
            raise ValueError('invalid forecast horizon')
        if type(self.valid_through) is not int or self.valid_through < 0:
            raise ValueError('invalid expiry')


@dataclass(frozen=True)
class Decision:
    id: str
    action: str
    reason: str
    admission: int
    before: Runtime
    target: str
    forecast_sha256: Optional[str]
    margin_ms: Optional[float]


class Governor:
    """State advances only on successful admission and verified matching ACK.

    ACK tests use fixtures, not native evidence. The adapter must use the existing
    gpu_runtime_control.verify_terminal_ack before passing an ACK to complete().
    """
    def __init__(self, runtime: Runtime, *, min_hold: int = 8,
                 window: int = 64, max_switches: int = 4):
        if any(type(x) is not int for x in (min_hold, window, max_switches)):
            raise ValueError('settings must be integers')
        if min_hold < 1 or window < min_hold or not 1 <= max_switches <= window:
            raise ValueError('invalid hold/window/rate settings')
        self.runtime = runtime
        self.min_hold, self.window, self.max_switches = min_hold, window, max_switches
        self.admission = 0
        self.last_switch = -1  # startup counts toward the initial hold interval
        self.switches = []
        self.pending = None
        self.dispatched = False

    def plan(self, proposed: str, *, allowed: FrozenSet[str],
             forecast: Optional[Forecast] = None) -> Decision:
        require_sha(proposed)
        if self.pending is not None:
            raise RuntimeError('one admission already pending')
        for p in allowed:
            require_sha(p)
        before = self.runtime
        target, action, reason, margin = before.policy, 'KEEP', 'same_policy', None
        if before.policy not in allowed:
            action, reason = 'BLOCK', 'incumbent_not_authorized'
        elif proposed != before.policy:
            if proposed not in allowed:
                reason = 'candidate_quality_or_authority_not_verified'
            elif forecast is None:
                reason = 'no_cost_evidence'
            elif (forecast.context != before.context or
                  forecast.from_policy != before.policy or forecast.to_policy != proposed):
                reason = 'cost_evidence_identity_mismatch'
            elif self.admission > forecast.valid_through:
                reason = 'cost_evidence_expired'
            elif forecast.gain_lower_ms <= 0:
                reason = 'no_positive_confidence_bound'
            elif forecast.horizon < self.min_hold:
                reason = 'forecast_horizon_too_short'
            elif self.admission - self.last_switch <= self.min_hold:
                reason = 'minimum_hold'
            elif len([i for i in self.switches if i > self.admission - self.window]) >= self.max_switches:
                reason = 'switch_rate_limit'
            else:
                margin = (forecast.gain_lower_ms * forecast.horizon -
                          forecast.switch_upper_ms - forecast.planner_upper_ms)
                if margin <= 0:
                    reason = 'switch_cost_not_amortized'
                else:
                    action, reason, target = 'PROPOSE', 'cost_and_quality_gates_pass', proposed
        payload = {'admission': self.admission, 'before': before.__dict__, 'target': target,
                   'action': action, 'reason': reason,
                   'forecast': forecast.__dict__ if forecast is not None else None}
        dec = Decision(digest(payload), action, reason, self.admission, before, target,
                       forecast.evidence_sha256 if forecast else None, margin)
        self.pending = dec
        self.dispatched = False
        return dec

    def mark_dispatched(self, decision: Decision) -> None:
        self._pending_matches(decision)
        if decision.action == 'BLOCK' or self.dispatched:
            raise RuntimeError('blocked or duplicate dispatch')
        self.dispatched = True

    def abort(self, decision: Decision) -> None:
        """Abort only before dispatch.

        A timeout or unknown native outcome is NOT permission to abort/retry.
        """
        self._pending_matches(decision)
        if self.dispatched:
            raise RuntimeError('dispatched or unknown outcome requires ACK reconciliation')
        self.pending = None

    def _pending_matches(self, decision):
        if decision != self.pending or decision.before != self.runtime or decision.admission != self.admission:
            raise RuntimeError('stale, replayed or foreign decision')

    def complete(self, decision: Decision, ack: dict, *, expected_txn: Optional[str] = None) -> None:
        self._pending_matches(decision)
        if not self.dispatched:
            raise RuntimeError('admission was not dispatched')
        if decision.action == 'BLOCK':
            raise RuntimeError('blocked admission must never be dispatched')
        expected_epoch = self.runtime.epoch + (1 if decision.action == 'PROPOSE' else 0)
        if (ack.get('worker') != self.runtime.worker or ack.get('context') != self.runtime.context or
            ack.get('active_policy_hash') != decision.target or
            type(ack.get('weight_epoch')) is not int or ack['weight_epoch'] != expected_epoch):
            raise RuntimeError('ACK worker/context/policy/epoch mismatch; state unchanged')
        if ack.get('verified_by_native_controller') is not True:
            raise RuntimeError('native ACK verification has not been attested by adapter')
        if decision.action == 'PROPOSE':
            if (not expected_txn or ack.get('txn_id') != expected_txn or
                ack.get('status') != 'REBIND_SET_APPLIED'):
                raise RuntimeError('missing exact REBIND_SET transaction ACK')
        elif ack.get('status') not in ('STARTUP_STATE', 'REBIND_SET_APPLIED'):
            raise RuntimeError('failure or unknown native state cannot complete a hold')
        if decision.action == 'PROPOSE':
            self.switches.append(self.admission)
            self.last_switch = self.admission
        self.runtime = Runtime(decision.target, expected_epoch, self.runtime.worker, self.runtime.context)
        self.admission += 1
        self.pending = None
        self.dispatched = False

#!/usr/bin/env python3
"""Request-scoped P5 risk-signal extraction from real persistent GPU events."""
from __future__ import annotations

class RiskSignalError(RuntimeError):
    pass

def _finite(value, name):
    try:
        value=float(value)
    except (TypeError,ValueError) as exc:
        raise RiskSignalError(f"{name} is invalid") from exc
    if value != value or value in (float("inf"),float("-inf")):
        raise RiskSignalError(f"{name} must be finite")
    return value

def summarize_events(events):
    if not isinstance(events,list):
        raise RiskSignalError("events must be a list")
    margins=[]; entropies=[]; ambiguities=[]; layers=[]
    for i,row in enumerate(events):
        if not isinstance(row,dict):
            raise RiskSignalError(f"events[{i}] must be an object")
        if row.get("margin") is not None:
            margins.append(_finite(row["margin"],f"events[{i}].margin"))
        if row.get("entropy") is not None:
            entropy=_finite(row["entropy"],f"events[{i}].entropy")
            if entropy < 0.0 or entropy > 1.0:
                raise RiskSignalError("normalized entropy must be in [0,1]")
            entropies.append(entropy)
        if row.get("routing_ambiguity_score") is not None:
            score=_finite(row["routing_ambiguity_score"],f"events[{i}].routing_ambiguity_score")
            if score < 0.0 or score > 1.0:
                raise RiskSignalError("routing ambiguity must be in [0,1]")
            ambiguities.append(score)
            if row.get("routing_ambiguity_layer") is not None:
                layers.append(int(row["routing_ambiguity_layer"]))
    return {
        "event_count":len(events),
        "min_margin":min(margins) if margins else None,
        "max_margin":max(margins) if margins else None,
        "max_entropy":max(entropies) if entropies else None,
        "min_entropy":min(entropies) if entropies else None,
        "max_routing_ambiguity_score":max(ambiguities) if ambiguities else None,
        "min_routing_ambiguity_score":min(ambiguities) if ambiguities else None,
        "routing_ambiguity_layers":sorted(set(layers)),
    }

def derive_signal(
    events,
    *,
    near_tie_margin_max,
    high_entropy_min,
    routing_ambiguity_min,
):
    margin_max=_finite(near_tie_margin_max,"near_tie_margin_max")
    entropy_min=_finite(high_entropy_min,"high_entropy_min")
    routing_min=_finite(routing_ambiguity_min,"routing_ambiguity_min")
    if margin_max < 0.0:
        raise RiskSignalError("near_tie_margin_max must be non-negative")
    if not 0.0 <= entropy_min <= 1.0:
        raise RiskSignalError("high_entropy_min must be in [0,1]")
    if not 0.0 <= routing_min <= 1.0:
        raise RiskSignalError("routing_ambiguity_min must be in [0,1]")
    summary=summarize_events(events)
    margin=summary["min_margin"]
    entropy=summary["max_entropy"]
    routing=summary["max_routing_ambiguity_score"]
    return {
        "near_tie": margin is not None and margin <= margin_max,
        "high_entropy": entropy is not None and entropy >= entropy_min,
        "routing_ambiguity": routing is not None and routing >= routing_min,
        "margin": margin,
        "entropy": entropy,
        "routing_ambiguity_score": routing,
        "risk_signal_summary": summary,
    }

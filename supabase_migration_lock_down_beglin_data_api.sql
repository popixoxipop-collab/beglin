-- Lock down Beglin legacy telemetry/research Data API objects.
-- Applied live on 2026-10-03 as migration lock_down_legacy_beglin_data_api.
-- RLS is already enabled on these tables and intentionally has no
-- anon/authenticated policies. Service-role access is retained for the
-- internal Beglin research pipeline.

revoke all privileges on table
  public.moe_attribution_provenance,
  public.moe_autopilot_shadow_decisions,
  public.moe_live_preflight_results,
  public.moe_neartie_events,
  public.moe_quant_sweep_results,
  public.moe_role_precision_state
from anon, authenticated;

revoke all privileges on sequence
  public.moe_attribution_provenance_id_seq,
  public.moe_autopilot_shadow_decisions_id_seq,
  public.moe_live_preflight_results_id_seq,
  public.moe_neartie_events_id_seq,
  public.moe_quant_sweep_results_id_seq
from anon, authenticated;

revoke execute on function
  public.increment_role_precision(text,text,text,integer,double precision)
from anon, authenticated;

grant all privileges on table
  public.moe_attribution_provenance,
  public.moe_autopilot_shadow_decisions,
  public.moe_live_preflight_results,
  public.moe_neartie_events,
  public.moe_quant_sweep_results,
  public.moe_role_precision_state
to service_role;

grant all privileges on sequence
  public.moe_attribution_provenance_id_seq,
  public.moe_autopilot_shadow_decisions_id_seq,
  public.moe_live_preflight_results_id_seq,
  public.moe_neartie_events_id_seq,
  public.moe_quant_sweep_results_id_seq
to service_role;

grant execute on function
  public.increment_role_precision(text,text,text,integer,double precision)
to service_role;

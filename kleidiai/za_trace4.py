import lldb, struct

# ZA_LOOP store point (st1w z24, right where the dequantized 4-column group gets written).
# Fires once per ZA_LOOP iteration (16x for out=64, 4 columns each).
OFF_STORE = None  # resolved at runtime relative to kernel entry

state = {"iter": 0, "first_nan_iter": None}

def store_hit(frame, bp_loc, dict):
    state["iter"] += 1
    z24 = frame.FindRegister("z24")
    data = z24.GetData()
    err = lldb.SBError()
    vals = [data.GetFloat(err, i*4) for i in range(16)]  # 16 lanes of 4 bytes = 64 bytes total
    nan_lanes = [i for i, v in enumerate(vals) if v != v]
    x7 = frame.FindRegister("x7").GetValueAsUnsigned()
    p1 = frame.FindRegister("p1").GetData()
    p1b = [p1.GetUnsignedInt8(err, i) for i in range(8)]
    print("  iter %2d: x7=0x%x nan_lanes=%s p1_bytes=%s z24[0:4]=%s" %
          (state["iter"], x7, nan_lanes, p1b, vals[:4]))
    if nan_lanes and state["first_nan_iter"] is None:
        state["first_nan_iter"] = state["iter"]
        print("  *** FIRST NaN at iteration %d, lanes %s ***" % (state["iter"], nan_lanes))
    return True

def run_trace(debugger, command, result, internal_dict):
    debugger.SetAsync(False)
    target = debugger.GetSelectedTarget()
    kernel_name = "kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa"
    ksyms = target.FindFunctions(kernel_name)
    kernel_addr = ksyms[0].GetSymbol().GetStartAddress()

    bp0 = target.BreakpointCreateBySBAddress(kernel_addr)

    process = target.LaunchSimple(["1"], None, None)
    if process is None or not process.IsValid():
        print("LAUNCH FAILED")
        return

    kernel_load = kernel_addr.GetLoadAddress(target)
    target.BreakpointDelete(bp0.GetID())
    addr = target.ResolveLoadAddress(kernel_load + 0x730)  # placeholder, overwritten below via arg
    import os
    off = int(os.environ.get("STORE_OFF", "0"), 16)
    addr = target.ResolveLoadAddress(kernel_load + off)
    bp1 = target.BreakpointCreateBySBAddress(addr)
    bp1.SetScriptCallbackFunction("za_trace4.store_hit")

    n = 0
    while process.GetState() not in (lldb.eStateExited, lldb.eStateCrashed) and n < 40:
        process.Continue()
        n += 1
        if state["iter"] >= 20:
            break

    print("=== total iterations seen: %d, first NaN at: %s ===" % (state["iter"], state["first_nan_iter"]))
    process.Kill()

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command script add -f za_trace4.run_trace za_trace4')

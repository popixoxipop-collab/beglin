import lldb

OFF_ZERO_ZA = 0xC4
OFF_FIRST_STORE = 0x1D0
OFF_ACC_LOAD = 0x1E4
OFF_SMSTOP = 0x314

state = {"call": 0, "pass_in_call": 0}

def kernel_entry(frame, bp_loc, dict):
    state["call"] += 1
    state["pass_in_call"] = 0
    print("=== CALL %d entered ===" % state["call"])
    return True

def zero_za_hit(frame, bp_loc, dict):
    state["pass_in_call"] += 1
    x10 = frame.FindRegister("x10").GetValueAsUnsigned()
    x3 = frame.FindRegister("x3").GetValueAsUnsigned()
    x4 = frame.FindRegister("x4").GetValueAsUnsigned()
    print("  [call %d pass %d] zero{za}: x10=%d x3=%d x4=%d" %
          (state["call"], state["pass_in_call"], x10, x3, x4))
    return True

def first_store_hit(frame, bp_loc, dict):
    z24 = frame.FindRegister("z24")
    data = z24.GetData()
    err = lldb.SBError()
    vals = [data.GetUnsignedInt32(err, i*4) for i in range(4)]
    print("  [call %d pass %d] FIRST-PASS store: z24[0:4]=%s" %
          (state["call"], state["pass_in_call"], " ".join("%08x" % v for v in vals)))
    return True

def acc_load_hit(frame, bp_loc, dict):
    z8 = frame.FindRegister("z8")
    data = z8.GetData()
    err = lldb.SBError()
    vals = [data.GetUnsignedInt32(err, i*4) for i in range(2)]
    print("  [call %d pass %d] LATER-PASS delta: z8[0:2]=%s" %
          (state["call"], state["pass_in_call"], " ".join("%08x" % v for v in vals)))
    return True

def smstop_hit(frame, bp_loc, dict):
    za = frame.FindRegister("za")
    data = za.GetData()
    err = lldb.SBError()
    vals = [data.GetUnsignedInt32(err, i*4) for i in range(8)]
    print("  [call %d] smstop, total passes=%d, za[0:8]=%s" %
          (state["call"], state["pass_in_call"], " ".join("%08x" % v for v in vals)))
    return True

def run_trace(debugger, command, result, internal_dict):
    debugger.SetAsync(False)
    target = debugger.GetSelectedTarget()
    fn_name = "kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa"
    syms = target.FindFunctions(fn_name)
    if syms.GetSize() == 0:
        print("FUNCTION NOT FOUND")
        return
    entry_addr = syms[0].GetSymbol().GetStartAddress()

    bp0 = target.BreakpointCreateBySBAddress(entry_addr)
    bp0.SetScriptCallbackFunction("za_trace.kernel_entry")

    process = target.LaunchSimple(["256"], None, None)
    if process is None or not process.IsValid():
        print("LAUNCH FAILED")
        return
    print("first stop: state=%s call=%d" % (process.GetState(), state["call"]))

    # NOW the module is loaded -- address-based breakpoints resolve reliably here (confirmed
    # empirically: pre-launch address breakpoints silently never fire on this platform).
    entry_load = entry_addr.GetLoadAddress(target)
    print("entry_load=0x%x" % entry_load)
    for off, cb in [(OFF_ZERO_ZA, "za_trace.zero_za_hit"),
                    (OFF_FIRST_STORE, "za_trace.first_store_hit"),
                    (OFF_ACC_LOAD, "za_trace.acc_load_hit"),
                    (OFF_SMSTOP, "za_trace.smstop_hit")]:
        addr = target.ResolveLoadAddress(entry_load + off)
        bp = target.BreakpointCreateBySBAddress(addr)
        bp.SetScriptCallbackFunction(cb)

    # bp0 no longer needed (module is loaded now); disable it to reduce noise, all subsequent
    # entries are inferred by seeing another zero-za-pass-1 or by explicit re-arm below.
    n = 0
    while process.GetState() not in (lldb.eStateExited, lldb.eStateCrashed) and n < 60:
        process.Continue()
        n += 1

    print("=== final process state:", process.GetState(), "iterations:", n, "===")

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command script add -f za_trace.run_trace za_trace')

import lldb

OFF_ZERO_ZA = 0xC4       # zero{za}, start of each K-pass
OFF_LABEL10 = 0x250      # clamp loop start (0x1000059e0), reads/clamps/writes dst
OFF_LABEL11 = 0x2BC      # right after clamp loop (0x100005a4c)

state = {"call": 0}

def wrapper_entry(frame, bp_loc, dict):
    state["call"] += 1
    x6 = frame.FindRegister("x6").GetValueAsUnsigned()  # y_kai pointer (AAPCS64 7th arg)
    print("=== CALL %d: wrapper entry, y_kai=0x%x ===" % (state["call"], x6))
    state["y_kai"] = x6
    return True

def label10_hit(frame, bp_loc, dict):
    x24 = frame.FindRegister("x24").GetValueAsUnsigned()
    p1 = frame.FindRegister("p1")
    p2 = frame.FindRegister("p2")
    err = lldb.SBError()
    p1b = p1.GetData().GetUnsignedInt8(err, 0)
    p2b = p2.GetData().GetUnsignedInt8(err, 0)
    same_as_ykai = (x24 == state.get("y_kai", -1))
    print("  [call %d] label10 (pre-clamp): x24=0x%x (==y_kai? %s) p1[0]=0x%02x p2[0]=0x%02x" %
          (state["call"], x24, same_as_ykai, p1b, p2b))
    # dump actual dst memory content at x24 for the first 8 floats, raw bytes
    err2 = lldb.SBError()
    data = frame.GetThread().GetProcess().ReadMemory(x24, 32, err2)
    if err2.Success():
        import struct
        floats = struct.unpack('<8f', data)
        print("    dst[0:8] (pre-clamp) = %s" % floats)
    return True

def label11_hit(frame, bp_loc, dict):
    ykai = state.get("y_kai", -1)
    err2 = lldb.SBError()
    data = frame.GetThread().GetProcess().ReadMemory(ykai, 32, err2)
    if err2.Success():
        import struct
        floats = struct.unpack('<8f', data)
        print("  [call %d] label11 (post-clamp): y_kai[0:8] = %s" % (state["call"], floats))
    return True

def run_trace(debugger, command, result, internal_dict):
    debugger.SetAsync(False)
    target = debugger.GetSelectedTarget()

    wrapper_name = "kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa"
    kernel_name = "kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa"

    wsyms = target.FindFunctions(wrapper_name)
    ksyms = target.FindFunctions(kernel_name)
    if wsyms.GetSize() == 0 or ksyms.GetSize() == 0:
        print("FUNCTION NOT FOUND", wsyms.GetSize(), ksyms.GetSize())
        return
    wrapper_addr = wsyms[0].GetSymbol().GetStartAddress()
    kernel_addr = ksyms[0].GetSymbol().GetStartAddress()

    bp0 = target.BreakpointCreateBySBAddress(wrapper_addr)
    bp0.SetScriptCallbackFunction("za_trace2.wrapper_entry")

    process = target.LaunchSimple(["256"], None, None)
    if process is None or not process.IsValid():
        print("LAUNCH FAILED")
        return

    kernel_load = kernel_addr.GetLoadAddress(target)
    for off, cb in [(OFF_LABEL10, "za_trace2.label10_hit"),
                    (OFF_LABEL11, "za_trace2.label11_hit")]:
        addr = target.ResolveLoadAddress(kernel_load + off)
        bp = target.BreakpointCreateBySBAddress(addr)
        bp.SetScriptCallbackFunction(cb)

    n = 0
    while process.GetState() not in (lldb.eStateExited, lldb.eStateCrashed) and n < 30:
        process.Continue()
        n += 1
        if state["call"] >= 3:
            break

    print("=== final state:", process.GetState(), "iterations:", n, "===")
    process.Kill()

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command script add -f za_trace2.run_trace za_trace2')

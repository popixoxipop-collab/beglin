import lldb, struct

OFF_PASS_END = 0x238  # 'subs x10,x10,x4' -- executes once per K_LOOP pass (256x per call),
                       # right after that pass's ZA_LOOP (16 sub-iterations) has fully written
                       # its contribution to the dst/scratch buffer.

state = {"call": 0, "pass_n": 0, "y_kai": None, "history": {1: [], 2: []}}

def wrapper_entry(frame, bp_loc, dict):
    state["call"] += 1
    state["pass_n"] = 0
    x6 = frame.FindRegister("x6").GetValueAsUnsigned()
    state["y_kai"] = x6
    print("=== CALL %d: y_kai=0x%x ===" % (state["call"], x6))
    return True

def pass_end_hit(frame, bp_loc, dict):
    state["pass_n"] += 1
    ykai = state["y_kai"]
    proc = frame.GetThread().GetProcess()
    err = lldb.SBError()
    data = proc.ReadMemory(ykai, 64*4, err)
    if not err.Success():
        return True
    floats = struct.unpack('<64f', data)
    nan_count = sum(1 for f in floats if f != f)
    checksum = sum(floats) if nan_count == 0 else float('nan')
    call = state["call"]
    if call in (1, 2):
        state["history"][call].append((state["pass_n"], nan_count, checksum, floats[0]))
    return True

def run_trace(debugger, command, result, internal_dict):
    debugger.SetAsync(False)
    target = debugger.GetSelectedTarget()

    wrapper_name = "kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa"
    kernel_name = "kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa"
    wsyms = target.FindFunctions(wrapper_name)
    ksyms = target.FindFunctions(kernel_name)
    wrapper_addr = wsyms[0].GetSymbol().GetStartAddress()
    kernel_addr = ksyms[0].GetSymbol().GetStartAddress()

    bp0 = target.BreakpointCreateBySBAddress(wrapper_addr)
    bp0.SetScriptCallbackFunction("za_trace3.wrapper_entry")

    process = target.LaunchSimple(["256"], None, None)
    if process is None or not process.IsValid():
        print("LAUNCH FAILED")
        return

    kernel_load = kernel_addr.GetLoadAddress(target)
    addr = target.ResolveLoadAddress(kernel_load + OFF_PASS_END)
    bp1 = target.BreakpointCreateBySBAddress(addr)
    bp1.SetScriptCallbackFunction("za_trace3.pass_end_hit")

    n = 0
    while process.GetState() not in (lldb.eStateExited, lldb.eStateCrashed) and n < 700:
        process.Continue()
        n += 1
        if state["call"] >= 3:
            break

    process.Kill()

    h1 = state["history"][1]
    h2 = state["history"][2]
    print("call1 passes recorded: %d, call2 passes recorded: %d" % (len(h1), len(h2)))
    first_div = None
    for i in range(min(len(h1), len(h2))):
        p1n, nan1, cs1, f01 = h1[i]
        p2n, nan2, cs2, f02 = h2[i]
        same = (nan1 == nan2) and (nan1 > 0 or abs(cs1 - cs2) < 1e-3) and (f01 == f01) == (f02 == f02)
        if not same and first_div is None:
            first_div = i
            print("FIRST DIVERGENCE at pass index %d (0-based):" % i)
            print("  call1: pass=%d nan_count=%d checksum=%s y_kai[0]=%s" % (p1n, nan1, cs1, f01))
            print("  call2: pass=%d nan_count=%d checksum=%s y_kai[0]=%s" % (p2n, nan2, cs2, f02))
            # print a window around it
            lo = max(0, i-2); hi = min(len(h1), len(h2), i+5)
            for j in range(lo, hi):
                print("  [%d] call1: nan=%d cs=%s y0=%s  |  call2: nan=%d cs=%s y0=%s" %
                      (j, h1[j][1], h1[j][2], h1[j][3], h2[j][1], h2[j][2], h2[j][3]))
            break
    if first_div is None:
        print("NO DIVERGENCE FOUND in recorded range (both identical throughout)")
        print("last few call1:", h1[-5:])
        print("last few call2:", h2[-5:])

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command script add -f za_trace3.run_trace za_trace3')

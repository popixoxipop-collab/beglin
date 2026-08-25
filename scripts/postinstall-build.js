#!/usr/bin/env node
// D-npm-1: this package ships C source and builds locally at install time,
// rather than shipping a prebuilt binary or an N-API addon.
//   WHY: the engine is a single-shot CLI process (mode/arg driven, no stable
//        in-process call boundary), so an N-API layer would mean rearchitecting
//        qwen_infer.c into a persistent-server library -- out of scope here.
//        A prebuilt binary needs a release pipeline (CI matrix, signing) that
//        doesn't exist yet for a first publish; building from source needs only
//        the Xcode Command Line Tools' clang, already present on most Mac dev
//        machines.
//   COST: install requires clang + macOS + Apple Silicon; no Linux/Windows/x86
//        support (the engine's kernels are NEON/SME2, ISA-specific by design).
//   EXIT: swap this script for a prebuild-install/prebuildify fetch once a
//        release pipeline exists; the lib/index.js API surface doesn't change.

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const BIN_DIR = path.join(ROOT, "bin");
const BUILD_DIR = path.join(ROOT, ".build");
const OUT_BINARY = path.join(BIN_DIR, "qwen_infer");

function warnSkip(reason) {
  console.warn(`[vdsp-engine] skipping native build: ${reason}`);
  console.warn(
    "[vdsp-engine] `require('vdsp-engine')` will load, but running the engine will fail " +
      "until a compatible binary is built. See README.md's Build section for the manual steps."
  );
  process.exit(0); // non-fatal: don't break `npm install` on unrelated platforms/CI.
}

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (res.error) throw res.error;
  if (res.status !== 0) {
    throw new Error(`${cmd} ${args.join(" ")} exited with code ${res.status}`);
  }
}

function main() {
  if (process.platform !== "darwin" || process.arch !== "arm64") {
    warnSkip(`unsupported platform ${process.platform}/${process.arch} (requires macOS arm64)`);
    return;
  }

  const clangCheck = spawnSync("clang", ["--version"]);
  if (clangCheck.error || clangCheck.status !== 0) {
    warnSkip("clang not found (install Xcode Command Line Tools: `xcode-select --install`)");
    return;
  }

  fs.mkdirSync(BUILD_DIR, { recursive: true });
  fs.mkdirSync(BIN_DIR, { recursive: true });

  const obj = (name) => path.join(BUILD_DIR, name.replace(/\.(c|S)$/, ".o"));

  console.log("[vdsp-engine] compiling qwen_infer.c (plain -- no SME/SVE arch flag)");
  run("clang", ["-O3", "-w", "-c", path.join(ROOT, "qwen_infer.c"), "-o", obj("qwen_infer.c")]);

  console.log("[vdsp-engine] compiling sme2_kai.c (SME2 dispatch wrapper)");
  run("clang", [
    "-O2",
    "-march=armv9.2-a+sme2",
    "-I",
    ROOT,
    "-c",
    path.join(ROOT, "sme2_kai.c"),
    "-o",
    obj("sme2_kai.c"),
  ]);

  const kernelFiles = [
    "kleidiai/kai_common_sme_asm.S",
    "kleidiai/kai_lhs_pack_f16pmrx2_f32_neon.c",
    "kleidiai/kai_lhs_quant_pack_qsi8d32p_f32_neon.c",
    "kleidiai/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.c",
    "kleidiai/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa_asm.S",
    "kleidiai/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.c",
    "kleidiai/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa_asm.S",
    "kleidiai/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.c",
    "kleidiai/kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.c",
  ];
  console.log("[vdsp-engine] compiling KleidiAI SME2 kernels");
  for (const f of kernelFiles) {
    run("clang", [
      "-O2",
      "-march=armv9.2-a+sme2",
      "-I",
      ROOT,
      "-c",
      path.join(ROOT, f),
      "-o",
      obj(path.basename(f)),
    ]);
  }

  console.log("[vdsp-engine] linking");
  const objs = ["qwen_infer.c", "sme2_kai.c", ...kernelFiles].map((f) => obj(path.basename(f)));
  run("clang", ["-O3", ...objs, "-o", OUT_BINARY, "-framework", "Accelerate", "-lpthread"]);

  // Caller-plain convention check (see RESULTS.md): the plain-compiled caller
  // object must carry zero SVE/SME instructions. This isn't a style nit --
  // it's the difference between "runs on every Apple Silicon chip" and
  // "SIGILLs on anything before M4".
  const otool = spawnSync("otool", ["-tV", obj("qwen_infer.c")]);
  const leaked = /\b(sve|sme|addvl)\b/i.test(otool.stdout ? otool.stdout.toString() : "");
  if (leaked) {
    throw new Error(
      "caller-plain convention violated: qwen_infer.o contains SVE/SME instructions. " +
        "This build is unsafe on non-SME2 hardware -- refusing to install it."
    );
  }

  console.log(`[vdsp-engine] built ${OUT_BINARY} (caller-plain check: OK)`);
}

main();

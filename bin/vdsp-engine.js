#!/usr/bin/env node
"use strict";

// Thin passthrough: `vdsp-engine <mode> [n_gen]` runs the compiled engine
// directly with the same argv the C binary itself expects. All engine
// configuration (weight paths, quantization mode, thread counts, etc.)
// stays env-var driven -- see README.md for the list.

const { binaryPath } = require("../lib/index.js");
const { spawnSync } = require("child_process");

const bin = binaryPath();
if (!bin) {
  console.error(
    "vdsp-engine: no compiled binary found. The native build may have been skipped " +
      "(unsupported platform, or clang missing) -- see README.md's Build section."
  );
  process.exit(1);
}

const res = spawnSync(bin, process.argv.slice(2), { stdio: "inherit" });
process.exit(res.status === null ? 1 : res.status);

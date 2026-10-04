#!/usr/bin/env node
"use strict";

const { spawnSync } = require("child_process");
const path = require("path");

const args = process.argv.slice(2);

// P12 capability front door. Keep this independent of the native engine
// binary so unsupported models can be inspected/fail-closed even when the
// native build is unavailable.
if (args[0] === "inspect-model" || args[0] === "capability-diff") {
  const tool = args[0] === "inspect-model" ? "inspect_model.py" : "capability_diff.py";
  const script = path.resolve(__dirname, "..", "tools", tool);
  const python = process.env.BEGLIN_PYTHON || "python3";
  const res = spawnSync(python, [script, ...args.slice(1)], { stdio: "inherit" });
  if (res.error) {
    console.error(
      "beglin " + args[0] + ": failed to launch Python capability tool: " +
        res.error.message
    );
    process.exit(1);
  }
  process.exit(res.status === null ? 1 : res.status);
}

// Existing native-engine passthrough remains unchanged for every other mode.
const { binaryPath } = require("../lib/index.js");
const bin = binaryPath();
if (!bin) {
  console.error(
    "beglin: no compiled binary found. The native build may have been skipped " +
      "(unsupported platform, or clang missing) -- see README.md's Build section."
  );
  process.exit(1);
}

const res = spawnSync(bin, args, { stdio: "inherit" });
process.exit(res.status === null ? 1 : res.status);

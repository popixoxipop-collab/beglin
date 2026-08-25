#!/usr/bin/env node
// Smoke test: no model weights are shipped in this package (see .gitignore),
// so this can't run real inference. It only proves the binary built, is
// executable, and fails the way the engine is documented to fail (a clear
// FATAL message about missing arch config) rather than crashing.

const { spawnEngine, binaryPath } = require("../lib/index.js");

const bin = binaryPath();
if (!bin) {
  console.error("SMOKE TEST FAIL: no compiled binary at bin/qwen_infer");
  process.exit(1);
}
console.log(`SMOKE TEST: found binary at ${bin}`);

const child = spawnEngine("greedy", 1, {
  env: { QWEN_BASE: "/nonexistent-beglin-smoke-test-path" },
});

let stderr = "";
child.stderr.on("data", (d) => (stderr += d));
child.on("error", (err) => {
  console.error("SMOKE TEST FAIL: could not spawn binary:", err.message);
  process.exit(1);
});
child.on("close", (code) => {
  if (stderr.includes("FATAL") && stderr.includes("arch config")) {
    console.log("SMOKE TEST PASS: binary ran and failed gracefully on missing weights, as documented");
    process.exit(0);
  }
  console.error(`SMOKE TEST FAIL: unexpected exit (code=${code}), stderr:\n${stderr}`);
  process.exit(1);
});

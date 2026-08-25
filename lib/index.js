"use strict";

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const BINARY_PATH = path.join(__dirname, "..", "bin", "qwen_infer");

/** Absolute path to the compiled engine binary (undefined if not yet built). */
function binaryPath() {
  return fs.existsSync(BINARY_PATH) ? BINARY_PATH : undefined;
}

/**
 * Spawn the engine directly. Mirrors the C binary's own CLI: `mode` selects
 * one of the engine's run modes (e.g. "greedy", "bench", "ppl", "spec" --
 * see qwen_infer.c's mode dispatch in main()), `nGen` is the token count for
 * generation modes. Model weights are located via the QWEN_BASE env var
 * (defaults to the engine's own compiled-in default path if unset --
 * override it to point at your own exported/quantized weights directory).
 *
 * Returns the raw ChildProcess so callers get full control over stdio,
 * signals, and streaming stdout -- this wraps a CLI tool, not an in-process
 * API (see README's "Why a subprocess wrapper" note).
 */
function spawnEngine(mode, nGen, opts = {}) {
  const bin = binaryPath();
  if (!bin) {
    throw new Error(
      "vdsp-engine binary not found -- the native build did not run or failed. " +
        "See README.md's Build section, or run `npm run build` in this package."
    );
  }
  const args = [mode];
  if (nGen !== undefined && nGen !== null) args.push(String(nGen));
  return spawn(bin, args, { env: { ...process.env, ...opts.env }, ...opts });
}

/**
 * Promise-based convenience wrapper around spawnEngine: runs to completion
 * and resolves with the collected output. For streaming output (e.g. token
 * generation as it happens), use spawnEngine directly instead.
 */
function runEngine(mode, nGen, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawnEngine(mode, nGen, opts);
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, stdout, stderr }));
  });
}

module.exports = { spawnEngine, runEngine, binaryPath };

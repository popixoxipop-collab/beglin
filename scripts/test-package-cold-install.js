#!/usr/bin/env node
"use strict";

const { spawnSync } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const REQUIRED_INSTALLED_FILES = [
  "bpe_tokenizer.c",
  "bpe_tokenizer.h",
  "unicode_cpt_flags.h",
  "gguf_write.c",
  "gguf_write.h",
  "gguf_write_quants.c",
  "gguf_write_quants.h",
  "scripts/postinstall-build.js",
];

function sha256File(file) {
  const h = crypto.createHash("sha256");
  h.update(fs.readFileSync(file));
  return h.digest("hex");
}

function run(cmd, args, options = {}) {
  const res = spawnSync(cmd, args, {
    encoding: "utf8",
    ...options,
  });
  if (res.error) throw res.error;
  return res;
}

function die(message, extra = "") {
  console.error("[package-cold-install] FAIL:", message);
  if (extra) console.error(extra);
  process.exit(1);
}

function main() {
  if (process.platform !== "darwin" || process.arch !== "arm64") {
    console.log(JSON.stringify({
      schema: "beglin-package-cold-install-v1",
      status: "SKIP_UNSUPPORTED",
      platform: process.platform,
      arch: process.arch,
    }, null, 2));
    return;
  }

  const clang = run("clang", ["--version"]);
  if (clang.status !== 0) {
    die("clang unavailable on supported macOS arm64 test host");
  }

  const root = fs.mkdtempSync(path.join(os.tmpdir(), "beglin-cold-install-"));
  const packDir = path.join(root, "pack");
  const consumer = path.join(root, "consumer");
  fs.mkdirSync(packDir, { recursive: true });
  fs.mkdirSync(consumer, { recursive: true });
  fs.writeFileSync(
    path.join(consumer, "package.json"),
    JSON.stringify({ name: "beglin-cold-consumer", version: "1.0.0", private: true }, null, 2) + "\n"
  );

  const binary = path.join(consumer, "node_modules", "beglin", "bin", "qwen_infer");
  if (fs.existsSync(binary)) {
    die("cold consumer unexpectedly contains native binary before install");
  }

  const packed = run("npm", ["pack", "--json", "--pack-destination", packDir], { cwd: ROOT });
  if (packed.status !== 0) {
    die("npm pack failed", (packed.stdout || "") + (packed.stderr || ""));
  }

  let report;
  try {
    report = JSON.parse(packed.stdout);
  } catch (err) {
    die("could not parse npm pack output", err.message);
  }
  if (!Array.isArray(report) || report.length !== 1 || !report[0].filename) {
    die("unexpected npm pack JSON shape");
  }

  const tarball = path.join(packDir, report[0].filename);
  if (!fs.existsSync(tarball)) {
    die(`npm pack reported missing tarball: ${tarball}`);
  }

  const install = run(
    "npm",
    ["install", tarball, "--foreground-scripts", "--no-audit", "--no-fund"],
    { cwd: consumer }
  );
  const buildLog = (install.stdout || "") + (install.stderr || "");
  if (install.status !== 0) {
    die(`cold npm install failed with code ${install.status}`, buildLog);
  }

  if (!fs.existsSync(binary)) {
    die(
      "npm install returned success but native binary is absent; native build may have silently skipped",
      buildLog
    );
  }

  const installedRoot = path.join(consumer, "node_modules", "beglin");
  for (const file of REQUIRED_INSTALLED_FILES) {
    if (!fs.existsSync(path.join(installedRoot, file))) {
      die(`required packaged file missing after cold install: ${file}`);
    }
  }

  const requireCheck = run(
    process.execPath,
    ["-e", "require('beglin'); process.stdout.write('require-ok\\n')"],
    { cwd: consumer }
  );
  if (requireCheck.status !== 0) {
    die("installed JS API could not be required", (requireCheck.stdout || "") + (requireCheck.stderr || ""));
  }

  const summary = {
    schema: "beglin-package-cold-install-v1",
    status: "PASS",
    platform: process.platform,
    arch: process.arch,
    node: process.version,
    clang: (clang.stdout || "").split(/\r?\n/)[0],
    tarball: path.basename(tarball),
    tarball_sha256: sha256File(tarball),
    binary_sha256: sha256File(binary),
    binary_size: fs.statSync(binary).size,
    installed_root: installedRoot,
    build_log_contains_success_marker: buildLog.includes("[beglin] built "),
    require_check: (requireCheck.stdout || "").trim(),
  };
  console.log(JSON.stringify(summary, null, 2));
}

main();

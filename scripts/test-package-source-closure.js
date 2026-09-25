#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const pkg = require(path.join(ROOT, "package.json"));
const {
  PLAIN_FILES,
  KERNEL_FILES,
  isSupportedPlatform,
} = require("./postinstall-build.js");

function rel(p) {
  return path.relative(ROOT, p).split(path.sep).join("/");
}

function resolveLocalInclude(fromRel, includeName) {
  const base = path.dirname(path.join(ROOT, fromRel));
  const direct = path.resolve(base, includeName);
  if (direct.startsWith(ROOT + path.sep) && fs.existsSync(direct)) return rel(direct);
  const rootRelative = path.resolve(ROOT, includeName);
  if (rootRelative.startsWith(ROOT + path.sep) && fs.existsSync(rootRelative)) return rel(rootRelative);
  return null;
}

function sourceClosure() {
  const queue = ["qwen_infer.c", ...PLAIN_FILES, "sme2_kai.c", ...KERNEL_FILES];
  const seen = new Set();
  while (queue.length) {
    const file = queue.shift();
    if (seen.has(file)) continue;
    seen.add(file);
    const full = path.join(ROOT, file);
    if (!(fs.existsSync(full) && fs.statSync(full).isFile())) throw new Error(`build input missing from checkout: ${file}`);
    const text = fs.readFileSync(full, "utf8");
    for (const match of text.matchAll(/^\s*#\s*include\s*"([^"]+)"/gm)) {
      const resolved = resolveLocalInclude(file, match[1]);
      if (resolved && !seen.has(resolved)) queue.push(resolved);
    }
  }
  return [...seen].sort();
}

function packageCovers(files, closure) {
  const shipped = new Set(files);
  return closure.filter((f) => !shipped.has(f));
}

function main() {
  const closure = sourceClosure();
  const missing = packageCovers(pkg.files || [], closure);
  if (missing.length) {
    console.error("[beglin package closure] missing from package.files:");
    for (const f of missing) console.error("  - " + f);
    process.exit(2);
  }

  // Self-check the detector on the exact historical regression.
  const withoutBpe = (pkg.files || []).filter((f) => f !== "bpe_tokenizer.h");
  const fixtureMissing = packageCovers(withoutBpe, closure);
  if (!fixtureMissing.includes("bpe_tokenizer.h")) {
    throw new Error("closure detector failed its missing-bpe_tokenizer.h fixture");
  }
  if (isSupportedPlatform("linux", "x64")) {
    throw new Error("unsupported-platform fixture classified linux/x64 as supported");
  }
  if (!isSupportedPlatform("darwin", "arm64")) {
    throw new Error("darwin/arm64 fixture must be supported");
  }

  const report = {
    schema: "beglin-package-source-closure-v1",
    status: "PASS",
    compile_sources: ["qwen_infer.c", ...PLAIN_FILES, "sme2_kai.c", ...KERNEL_FILES],
    closure,
    package_files_count: (pkg.files || []).length,
    missing: [],
    negative_fixture: {
      removed: "bpe_tokenizer.h",
      detected: true,
    },
  };
  console.log(JSON.stringify(report, null, 2));
}

if (require.main === module) main();

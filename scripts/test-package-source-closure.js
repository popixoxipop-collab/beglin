#!/usr/bin/env node
"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const PACKAGE_JSON = path.join(ROOT, "package.json");
const POSTINSTALL = path.join(ROOT, "scripts", "postinstall-build.js");

const REQUIRED_BUILD_SOURCES = [
  "bpe_tokenizer.c",
  "gguf_write.c",
  "gguf_write_quants.c",
];

const REQUIRED_ARCHIVE_FILES = [
  "bpe_tokenizer.c",
  "bpe_tokenizer.h",
  "unicode_cpt_flags.h",
  "gguf_write.c",
  "gguf_write.h",
  "gguf_write_quants.c",
  "gguf_write_quants.h",
];

const FORBIDDEN_ARCHIVE_PREFIXES = [
  ".build/",
  ".env",
  "bin/qwen_infer",
];

function fail(message) {
  console.error("[package-source-closure] FAIL:", message);
  process.exit(1);
}

function main() {
  const pkg = JSON.parse(fs.readFileSync(PACKAGE_JSON, "utf8"));
  const postinstall = fs.readFileSync(POSTINSTALL, "utf8");
  const declared = new Set(pkg.files || []);

  for (const source of REQUIRED_BUILD_SOURCES) {
    if (!postinstall.includes(`"${source}"`)) {
      fail(`native build source missing from postinstall-build.js: ${source}`);
    }
  }

  for (const file of REQUIRED_ARCHIVE_FILES) {
    if (!declared.has(file)) {
      fail(`required native source/header missing from package.json files: ${file}`);
    }
    if (!fs.existsSync(path.join(ROOT, file))) {
      fail(`declared native source/header does not exist in checkout: ${file}`);
    }
  }

  const packed = spawnSync("npm", ["pack", "--dry-run", "--json"], {
    cwd: ROOT,
    encoding: "utf8",
    env: process.env,
  });
  if (packed.error) throw packed.error;
  if (packed.status !== 0) {
    process.stderr.write(packed.stdout || "");
    process.stderr.write(packed.stderr || "");
    fail(`npm pack --dry-run failed with code ${packed.status}`);
  }

  let report;
  try {
    report = JSON.parse(packed.stdout);
  } catch (err) {
    fail(`could not parse npm pack JSON: ${err.message}`);
  }
  if (!Array.isArray(report) || report.length !== 1 || !Array.isArray(report[0].files)) {
    fail("unexpected npm pack JSON shape");
  }

  const archiveFiles = new Set(report[0].files.map((row) => row.path));
  for (const file of REQUIRED_ARCHIVE_FILES) {
    if (!archiveFiles.has(file)) {
      fail(`required file absent from actual npm tarball file list: ${file}`);
    }
  }

  for (const forbidden of FORBIDDEN_ARCHIVE_PREFIXES) {
    for (const file of archiveFiles) {
      if (file === forbidden || file.startsWith(forbidden)) {
        fail(`forbidden local/build artifact would be packed: ${file}`);
      }
    }
  }

  const result = {
    schema: "beglin-package-source-closure-v1",
    status: "PASS",
    build_sources: REQUIRED_BUILD_SOURCES,
    archive_required: REQUIRED_ARCHIVE_FILES,
    archive_entry_count: archiveFiles.size,
    package_version: pkg.version,
  };
  console.log(JSON.stringify(result, null, 2));
}

main();

#!/usr/bin/env node
"use strict";

const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const ROOT = path.join(__dirname, "..");

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, {
    cwd: ROOT,
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
    ...opts,
  });
  return res;
}

function sha256File(file) {
  const h = crypto.createHash("sha256");
  h.update(fs.readFileSync(file));
  return h.digest("hex");
}

function machoIdentity(file) {
  const buf = fs.readFileSync(file);
  // 64-bit little-endian Mach-O. The linker emits LC_UUID metadata that is
  // intentionally unique per link and does not change executable code.
  if (buf.length < 32 || buf.readUInt32LE(0) !== 0xfeedfacf) {
    return { uuid: null, normalized_sha256: sha256File(file), normalized: false };
  }
  const copy = Buffer.from(buf);
  const ncmds = copy.readUInt32LE(16);
  let off = 32;
  let uuid = null;
  for (let i = 0; i < ncmds; i++) {
    if (off + 8 > copy.length) throw new Error("truncated Mach-O load command table");
    const cmd = copy.readUInt32LE(off);
    const cmdsize = copy.readUInt32LE(off + 4);
    if (cmdsize < 8 || off + cmdsize > copy.length) {
      throw new Error("invalid Mach-O load command size");
    }
    if (cmd === 0x1b && cmdsize >= 24) {
      const raw = copy.subarray(off + 8, off + 24);
      uuid = [...raw].map((x) => x.toString(16).padStart(2, "0")).join("");
      copy.fill(0, off + 8, off + 24);
    }
    off += cmdsize;
  }
  return {
    uuid,
    normalized_sha256: crypto.createHash("sha256").update(copy).digest("hex"),
    normalized: uuid !== null,
  };
}

function assertNoClangIsClassifiedAsSkipped() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "beglin-no-clang-"));
  try {
    const scripts = path.join(tmp, "scripts");
    fs.mkdirSync(scripts, { recursive: true });
    fs.copyFileSync(path.join(ROOT, "scripts/postinstall-build.js"), path.join(scripts, "postinstall-build.js"));
    const res = spawnSync(process.execPath, [path.join(scripts, "postinstall-build.js")], {
      encoding: "utf8",
      env: { ...process.env, PATH: "/nonexistent" },
    });
    const output = (res.stdout || "") + (res.stderr || "");
    if (res.status !== 0 || !output.includes("skipping native build: clang not found")) {
      throw new Error("clang-missing fixture did not produce the expected explicit SKIP classification");
    }
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

function main() {
  if (process.platform !== "darwin" || process.arch !== "arm64") {
    throw new Error(`cold-install verification requires darwin/arm64, got ${process.platform}/${process.arch}`);
  }

  assertNoClangIsClassifiedAsSkipped();

  const pack = run("npm", ["pack", "--json"]);
  if (pack.status !== 0) throw new Error("npm pack failed:\n" + pack.stderr);
  const packed = JSON.parse(pack.stdout);
  if (!Array.isArray(packed) || packed.length !== 1) throw new Error("unexpected npm pack JSON");
  const tgz = path.join(ROOT, packed[0].filename);
  if (!(fs.existsSync(tgz) && fs.statSync(tgz).isFile())) throw new Error("npm pack did not create " + tgz);

  const packedPaths = new Set((packed[0].files || []).map((f) => f.path));
  const forbiddenExact = [".env", "bin/qwen_infer"];
  for (const bad of forbiddenExact) {
    if (packedPaths.has(bad)) throw new Error("tarball contains forbidden local artifact: " + bad);
  }
  const forbiddenPrefixes = [".build/", "node_modules/"];
  for (const file of packedPaths) {
    if (forbiddenPrefixes.some((prefix) => file.startsWith(prefix))) {
      throw new Error("tarball contains forbidden generated path: " + file);
    }
    if (/\.(safetensors|gguf)$/i.test(file)) {
      throw new Error("tarball unexpectedly contains model weights: " + file);
    }
  }

  const clangVersion = spawnSync("clang", ["--version"], { encoding: "utf8" });
  if (clangVersion.status !== 0) throw new Error("clang disappeared before cold-install verification");
  const npmVersion = spawnSync("npm", ["--version"], { encoding: "utf8" });
  if (npmVersion.status !== 0) throw new Error("npm --version failed");

  const consumer = fs.mkdtempSync(path.join(os.tmpdir(), "beglin-cold-install-"));
  const binary = path.join(consumer, "node_modules", "beglin", "bin", "qwen_infer");
  try {
    if (fs.existsSync(binary)) throw new Error("cold consumer unexpectedly starts with a binary");

    const install = spawnSync("npm", ["install", tgz, "--foreground-scripts"], {
      cwd: consumer,
      encoding: "utf8",
      maxBuffer: 32 * 1024 * 1024,
    });
    const output = (install.stdout || "") + (install.stderr || "");
    if (install.status !== 0) {
      throw new Error(`cold npm install failed (exit=${install.status}):\n${output}`);
    }
    if (output.includes("[beglin] skipping native build:")) {
      throw new Error("cold npm install exited 0 but native build was SKIPPED");
    }
    if (!output.includes("[beglin] built ")) {
      throw new Error("cold npm install did not emit the native build success marker");
    }
    if (!(fs.existsSync(binary) && fs.statSync(binary).isFile()) || fs.statSync(binary).size <= 0) {
      throw new Error("cold npm install did not produce a non-empty qwen_infer binary");
    }

    const api = spawnSync(process.execPath, ["-e", "require('beglin')"], {
      cwd: consumer,
      encoding: "utf8",
    });
    if (api.status !== 0) throw new Error("installed JS API failed to load: " + api.stderr);

    const macho = machoIdentity(binary);
    console.log(JSON.stringify({
      schema: "beglin-package-cold-install-v1",
      status: "PASS",
      platform: process.platform,
      arch: process.arch,
      node: process.version,
      npm: (npmVersion.stdout || "").trim(),
      clang: (clangVersion.stdout || "").split(/\r?\n/, 1)[0],
      npm_pack_filename: packed[0].filename,
      packed_file_count: packedPaths.size,
      tarball_forbidden_artifacts: "NONE",
      tarball_sha256: sha256File(tgz),
      tarball_size: fs.statSync(tgz).size,
      binary_sha256: sha256File(binary),
      binary_normalized_sha256: macho.normalized_sha256,
      macho_uuid: macho.uuid,
      macho_uuid_normalized: macho.normalized,
      binary_size: fs.statSync(binary).size,
      consumer_root: consumer,
      native_build_skipped: false,
      clang_missing_fixture: "SKIP_CLASSIFIED",
    }, null, 2));
  } finally {
    if (process.env.BEGLIN_KEEP_COLD_INSTALL !== "1") {
      fs.rmSync(consumer, { recursive: true, force: true });
    }
  }
}

if (require.main === module) main();

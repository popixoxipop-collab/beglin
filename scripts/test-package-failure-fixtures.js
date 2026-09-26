#!/usr/bin/env node
"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");

function run(cmd, args, options = {}) {
  const res = spawnSync(cmd, args, { encoding: "utf8", ...options });
  if (res.error) throw res.error;
  return res;
}

function fail(message, res) {
  console.error("[package-failure-fixtures] FAIL:", message);
  if (res) {
    if (res.stdout) console.error(res.stdout);
    if (res.stderr) console.error(res.stderr);
  }
  process.exit(1);
}

function copyFixture(dest) {
  fs.cpSync(ROOT, dest, {
    recursive: true,
    filter: (source) => {
      const rel = path.relative(ROOT, source);
      if (!rel) return true;
      const first = rel.split(path.sep)[0];
      if ([".git", ".build", "node_modules"].includes(first)) return false;
      if (rel === path.join("bin", "qwen_infer")) return false;
      return true;
    },
  });
}

function main() {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "beglin-package-fixtures-"));
  const results = {};

  // Fixture 1: source/header closure must fail before packing.
  const missingRoot = path.join(temp, "missing-source");
  copyFixture(missingRoot);
  fs.unlinkSync(path.join(missingRoot, "bpe_tokenizer.h"));
  const missing = run(
    process.execPath,
    [path.join(missingRoot, "scripts", "test-package-source-closure.js")],
    { cwd: missingRoot }
  );
  if (missing.status === 0 || !((missing.stderr || "") + (missing.stdout || "")).includes("does not exist")) {
    fail("missing-source fixture was not rejected", missing);
  }
  results.missing_source = { status: "PASS", observed_exit: missing.status };

  // Fixture 2: on a supported host, no clang must be distinguishable from a native PASS.
  const noClang = run(
    process.execPath,
    [path.join(ROOT, "scripts", "test-package-cold-install.js")],
    { cwd: ROOT, env: { ...process.env, PATH: path.join(temp, "empty-path") } }
  );
  if (noClang.status === 0 || !((noClang.stderr || "") + (noClang.stdout || "")).includes("clang unavailable")) {
    fail("clang-absent fixture did not fail the native verification", noClang);
  }
  results.clang_absent = { status: "PASS", observed_exit: noClang.status };

  // Fixture 3: preserve package policy: unsupported platform skips the postinstall build.
  const unsupportedCode = [
    "Object.defineProperty(process,'platform',{value:'linux'});",
    "Object.defineProperty(process,'arch',{value:'x64'});",
    "require('./scripts/postinstall-build.js');",
  ].join("");
  const unsupported = run(process.execPath, ["-e", unsupportedCode], { cwd: ROOT });
  const unsupportedLog = (unsupported.stdout || "") + (unsupported.stderr || "");
  if (unsupported.status !== 0 || !unsupportedLog.includes("skipping native build: unsupported platform linux/x64")) {
    fail("unsupported-platform fixture did not preserve explicit skip behavior", unsupported);
  }
  results.unsupported_platform = { status: "PASS", observed_exit: unsupported.status };

  // Fixture 4: compile can appear successful while the link step fails; postinstall must fail.
  const fakeBin = path.join(temp, "fake-bin");
  fs.mkdirSync(fakeBin, { recursive: true });
  const fakeClang = path.join(fakeBin, "clang");
  fs.writeFileSync(
    fakeClang,
    `#!/usr/bin/env python3
import pathlib, sys
args=sys.argv[1:]
if args==['--version']:
    print('fixture clang 0')
    raise SystemExit(0)
if '-c' in args:
    if '-o' in args:
        out=pathlib.Path(args[args.index('-o')+1])
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes(b'fixture-object')
    raise SystemExit(0)
print('intentional fixture link failure', file=sys.stderr)
raise SystemExit(42)
`
  );
  fs.chmodSync(fakeClang, 0o755);
  const badLink = run(
    process.execPath,
    [path.join(ROOT, "scripts", "postinstall-build.js")],
    { cwd: ROOT, env: { ...process.env, PATH: fakeBin + path.delimiter + process.env.PATH } }
  );
  const badLinkLog = (badLink.stdout || "") + (badLink.stderr || "");
  if (badLink.status === 0 || !badLinkLog.includes("intentional fixture link failure")) {
    fail("bad-link fixture did not propagate linker failure", badLink);
  }
  results.link_failure = { status: "PASS", observed_exit: badLink.status };

  console.log(JSON.stringify({
    schema: "beglin-package-failure-fixtures-v1",
    status: "PASS",
    fixtures: results,
  }, null, 2));
}

main();

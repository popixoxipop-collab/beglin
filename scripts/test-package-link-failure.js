#!/usr/bin/env node
"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");

function fail(message, res) {
  console.error("[package-link-failure] FAIL:", message);
  if (res?.stdout) console.error(res.stdout);
  if (res?.stderr) console.error(res.stderr);
  process.exit(1);
}

function main() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "beglin-link-failure-"));
  const fakeBin = path.join(tmp, "bin");
  fs.mkdirSync(fakeBin, { recursive: true });

  const fakeClang = path.join(fakeBin, "clang");
  fs.writeFileSync(
    fakeClang,
    `#!/usr/bin/env python3
import pathlib, sys
args=sys.argv[1:]
if args == ['--version']:
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

  const res = spawnSync(
    process.execPath,
    [path.join(ROOT, "scripts", "postinstall-build.js")],
    {
      cwd: ROOT,
      encoding: "utf8",
      env: { ...process.env, PATH: fakeBin + path.delimiter + process.env.PATH },
    }
  );

  const output = (res.stdout || "") + (res.stderr || "");
  if (res.status === 0) fail("link failure was not propagated", res);
  if (!output.includes("intentional fixture link failure")) {
    fail("fixture did not reach the link stage", res);
  }

  console.log(JSON.stringify({
    schema: "beglin-package-link-failure-v1",
    status: "PASS",
    observed_exit: res.status,
    expected_nonzero: true,
  }, null, 2));
}

main();

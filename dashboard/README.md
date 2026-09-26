# beglin-experiment-dashboard

Contract-first observability backend for Beglin local quantization experiments

Scaffolded by `bskel new --stack fastapi`.

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
fastapi dev app/main.py --port 8787     # or: uvicorn app.main:app --reload --port 8787
```

Then check http://127.0.0.1:8787/health

## Database

`--database sqlite` pinned no extra driver -- CPython ships `sqlite3`, which SQLModel/SQLAlchemy drive directly.

No engine, session, or connection code was generated: the connection URL, pooling, migrations
and session lifecycle are decisions about your application, not ones a scaffolder should make
for you. Wire them up yourself before adding models.

## Next steps

This is a local-only git repository with one commit. `bskel preflight` needs a real `origin`
remote with a resolvable default branch, so:

```bash
gh repo create <name> --private --source=. --push   # or push to a remote you already own
git remote set-head origin --auto
bskel preflight
```

From there, `bskel status` / `bskel next` will tell you what to run.

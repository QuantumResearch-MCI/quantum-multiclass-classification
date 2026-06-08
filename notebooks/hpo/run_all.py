"""
Batch runner for every HPO notebook under ``notebooks/hpo/``.

Runs each notebook end-to-end in its own Python subprocess (no Jupyter / no
nbconvert needed). Because every notebook persists its artifacts to disk
(``results/csv``, ``results/figures``, ``results/best``, ``results/checkpoints``)
and is checkpoint-resumable, you can launch the whole batch once and let it run.

Key features
------------
* Resumable at TWO levels: the runner skips notebooks it has already finished
  (recorded in ``results/run_state.json``); and each notebook itself skips
  config combos already in its checkpoint.
* Continue-on-error: a failing notebook is logged and the batch keeps going.
* Per-notebook logs in ``results/run_logs/<relative-path>.log``.
* Cheap models first (classical -> dl -> quantum), then ``results.ipynb`` last.
* Filters + optional parallelism.

Examples
--------
    # run everything, sequentially (safest; quantum is heavy)
    python notebooks/hpo/run_all.py

    # only classical + dl, 4 in parallel (they're fast)
    python notebooks/hpo/run_all.py --category classical dl --jobs 4

    # only one quantum family
    python notebooks/hpo/run_all.py --category quantum --family quadratic

    # just show what would run
    python notebooks/hpo/run_all.py --list

    # re-run from scratch (ignore the "already done" state)
    python notebooks/hpo/run_all.py --force

Notes
-----
* Classical XGBoost uses ``device='cuda'`` and CatBoost may use the GPU; running
  many of those in parallel can exhaust GPU memory. Keep ``--jobs`` low (1-2) if
  you hit OOM. Quantum notebooks are CPU/most-expensive — prefer ``--jobs 1``.
* Inline plots are NOT written back into the .ipynb (we don't use a kernel), but
  every figure is saved as PNG under ``results/figures/`` and the leaderboard is
  in ``results.ipynb`` / ``results/eval_all_final_hpo.csv``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent              # .../notebooks/hpo
PROJECT_ROOT = HERE.parents[1]                       # .../codes
RESULTS = HERE / "results"
RUN_LOGS = RESULTS / "run_logs"
TMP_DIR = RESULTS / "_tmp_run"
STATE_FILE = RESULTS / "run_state.json"

# Prepended to every generated script: headless plotting + a display() shim so
# notebooks written for Jupyter run fine under plain python.
SHIM = (
    "import os, builtins, matplotlib\n"
    "matplotlib.use('Agg')\n"
    "if not hasattr(builtins, 'display'):\n"
    "    builtins.display = lambda *a, **k: [print(repr(x)[:2000]) for x in a]\n"
)


def category_of(path: Path) -> str:
    rel = path.relative_to(HERE).as_posix()
    if rel == "results.ipynb":
        return "results"
    return rel.split("/")[0]            # classical | dl | quantum


def discover(args):
    """Return ordered list of notebooks to run, honouring filters."""
    nbs = [p for p in HERE.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts]
    cats_order = {"classical": 0, "dl": 1, "quantum": 2, "results": 3}

    sel = []
    for p in nbs:
        cat = category_of(p)
        if args.category and cat not in args.category and cat != "results":
            continue
        if args.family and cat == "quantum" and args.family not in p.parent.name:
            continue
        if args.model and args.model not in p.stem:
            continue
        sel.append(p)

    # results.ipynb only if it survived filters AND we're allowed to run it
    if args.no_results:
        sel = [p for p in sel if category_of(p) != "results"]

    sel.sort(key=lambda p: (cats_order[category_of(p)], p.as_posix()))
    return sel


def notebook_to_script(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = [SHIM]
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        # drop any shell (!) / magic (%) lines just in case
        lines = [ln for ln in src.split("\n")
                 if not ln.lstrip().startswith(("!", "%"))]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def run_one(path: Path, timeout):
    """Execute one notebook in a subprocess. Returns (status, seconds)."""
    rel = path.relative_to(HERE).as_posix()
    log_path = RUN_LOGS / (rel.replace("/", "__") + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_py = TMP_DIR / (rel.replace("/", "__") + ".py")
    tmp_py.parent.mkdir(parents=True, exist_ok=True)
    tmp_py.write_text(notebook_to_script(path), encoding="utf-8")

    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# {rel}\n# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        lf.flush()
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp_py)],
                cwd=str(PROJECT_ROOT), env=env,
                stdout=lf, stderr=subprocess.STDOUT,
                timeout=(timeout or None),
            )
            status = "ok" if proc.returncode == 0 else f"fail(rc={proc.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    return status, time.perf_counter() - t0


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Run all HPO notebooks under notebooks/hpo/.")
    ap.add_argument("--category", nargs="*", choices=["classical", "dl", "quantum"],
                    help="Restrict to these categories (results.ipynb always runs last unless --no-results).")
    ap.add_argument("--family", help="Quantum family filter (substring of folder, e.g. quadratic).")
    ap.add_argument("--model", help="Filter by filename stem substring (e.g. qsvc, svc_rbf).")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel notebooks (default 1; keep low for quantum/GPU).")
    ap.add_argument("--timeout", type=int, default=0, help="Per-notebook timeout in seconds (0 = none).")
    ap.add_argument("--force", action="store_true", help="Re-run notebooks already marked done.")
    ap.add_argument("--no-results", action="store_true", help="Do not run results.ipynb at the end.")
    ap.add_argument("--list", action="store_true", help="List the notebooks that would run, then exit.")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    nbs = discover(args)

    state = {} if args.force else load_state()
    pending = [p for p in nbs
               if args.force or state.get(p.relative_to(HERE).as_posix(), {}).get("status") != "ok"]
    done_already = len(nbs) - len(pending)

    print(f"📂 {len(nbs)} notebooks matched | {done_already} already done | {len(pending)} to run | jobs={args.jobs}")
    for p in nbs:
        rel = p.relative_to(HERE).as_posix()
        mark = "·" if p in pending else "✓"
        print(f"   {mark} {rel}")
    if args.list:
        return
    if not pending:
        print("✅ Nothing to do — all matched notebooks already completed (use --force to redo).")
        return

    # results.ipynb (if present) must run AFTER all models -> run it alone last.
    results_nb = [p for p in pending if category_of(p) == "results"]
    model_nbs = [p for p in pending if category_of(p) != "results"]

    print(f"\n▶ Running {len(model_nbs)} model notebooks...\n")
    counts = {"ok": 0, "fail": 0, "timeout": 0}
    t_start = time.perf_counter()

    def _record(p, status, secs):
        rel = p.relative_to(HERE).as_posix()
        state[rel] = {"status": "ok" if status == "ok" else status,
                      "seconds": round(secs, 1),
                      "ts": datetime.now().isoformat(timespec="seconds")}
        save_state(state)
        key = "ok" if status == "ok" else ("timeout" if status == "timeout" else "fail")
        counts[key] += 1
        return rel

    n = len(model_nbs)
    if args.jobs <= 1:
        for i, p in enumerate(model_nbs, 1):
            rel = p.relative_to(HERE).as_posix()
            print(f"[{i}/{n}] {rel} ...", flush=True)
            status, secs = run_one(p, args.timeout)
            _record(p, status, secs)
            print(f"      -> {status}  ({secs:.0f}s)", flush=True)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, p, args.timeout): p for p in model_nbs}
            for fut in as_completed(futs):
                p = futs[fut]
                status, secs = fut.result()
                rel = _record(p, status, secs)
                done += 1
                print(f"[{done}/{n}] {rel} -> {status}  ({secs:.0f}s)", flush=True)

    # Summary table (CSV) for the models.
    summary = RESULTS / "run_summary.csv"
    with open(summary, "w", encoding="utf-8") as f:
        f.write("notebook,status,seconds,ts\n")
        for rel, info in sorted(state.items()):
            f.write(f"{rel},{info['status']},{info.get('seconds','')},{info.get('ts','')}\n")

    elapsed = time.perf_counter() - t_start
    print(f"\n📊 Models done in {elapsed/60:.1f} min — "
          f"ok={counts['ok']} fail={counts['fail']} timeout={counts['timeout']}")
    fails = [rel for rel, info in state.items()
             if info["status"] != "ok" and category_of(HERE / rel) != "results"]
    if fails:
        print("   ⚠️  Failed/incomplete (see results/run_logs/ for tracebacks):")
        for rel in fails:
            print(f"      - {rel}  [{state[rel]['status']}]")

    # Aggregate at the very end.
    if results_nb and counts["fail"] == 0 or (results_nb and not args.no_results):
        rp = results_nb[0]
        print(f"\n▶ Aggregating: {rp.relative_to(HERE).as_posix()}")
        status, secs = run_one(rp, args.timeout)
        _record(rp, status, secs)
        print(f"      -> {status}  ({secs:.0f}s)")
        if status == "ok":
            print(f"      leaderboard: {RESULTS / 'eval_all_final_hpo.csv'}")

    print(f"\n📄 Run log index : {RUN_LOGS}")
    print(f"📄 Run summary   : {summary}")
    print(f"📄 Resume state  : {STATE_FILE}  (delete or --force to redo)")


if __name__ == "__main__":
    main()

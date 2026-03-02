"""
WoundAI — Full Pipeline Launcher
Runs all 4 steps in sequence:
  1. Inspect_Dataset.py
  2. Preprocess.py
  3. prepare_dataset.py
  4. Train_cpu.py

All output is shown live in the console AND saved to results/pipeline_log.txt
Run: python run_pipeline.py
"""
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

BASE   = Path(__file__).parent.resolve()
PY     = BASE / "venv" / "Scripts" / "python.exe"
LOG    = BASE / "results" / "pipeline_log.txt"

STEPS = [
    ("Inspect Dataset",      [str(PY), "-u", str(BASE / "Inspect_Dataset.py"),
                               "--root",       str(BASE),
                               "--output_dir", str(BASE / "inspection")]),
    ("Preprocess Images",    [str(PY), "-u", str(BASE / "Preprocess.py"),
                               "--raw_dir",    str(BASE),
                               "--out_dir",    str(BASE / "datasets_clean")]),
    ("Prepare Datasets",     [str(PY), "-u", str(BASE / "prepare_dataset.py"),
                               "--clean_dir",  str(BASE / "datasets_clean"),
                               "--out_dir",    str(BASE / "datasets")]),
    ("Train Models (GPU)",   [str(PY), "-u", str(BASE / "Train_cpu.py"),
                               "--data_dir",   str(BASE / "datasets"),
                               "--output_dir", str(BASE / "results")]),
]

# ── Create output dirs ────────────────────────────────────────────────────────
for d in ["results", "inspection", "datasets_clean", "datasets"]:
    (BASE / d).mkdir(parents=True, exist_ok=True)

# ── Logging helper ────────────────────────────────────────────────────────────
def log(msg: str, file):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    print(line, file=file, flush=True)

# ── Run ───────────────────────────────────────────────────────────────────────
with open(LOG, "w", encoding="utf-8") as logfile:
    log("=" * 65, logfile)
    log("  WoundAI Full Pipeline", logfile)
    log(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", logfile)
    log(f"  Base    : {BASE}", logfile)
    log(f"  Log     : {LOG}", logfile)
    log("=" * 65, logfile)

    for step_num, (step_name, cmd) in enumerate(STEPS, 1):
        log("", logfile)
        log(f"[STEP {step_num}/{len(STEPS)}] {step_name}", logfile)
        log("-" * 65, logfile)

        t0 = datetime.now()
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(BASE),
        )

        # Stream output line-by-line to screen + log simultaneously
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            print(line, file=logfile, flush=True)

        proc.wait()
        elapsed = round((datetime.now() - t0).total_seconds(), 1)

        if proc.returncode != 0:
            log(f"", logfile)
            log(f"ERROR: Step {step_num} '{step_name}' failed "
                f"(exit code {proc.returncode}) after {elapsed}s", logfile)
            log(f"Pipeline aborted.", logfile)
            log("=" * 65, logfile)
            print("\nPipeline FAILED. Press Enter to close.", flush=True)
            input()
            sys.exit(1)

        log(f"[STEP {step_num}/{len(STEPS)}] DONE in {elapsed}s", logfile)

    log("", logfile)
    log("=" * 65, logfile)
    log("  ALL STEPS COMPLETE", logfile)
    log(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", logfile)
    log(f"  Inspection : {BASE / 'inspection'}", logfile)
    log(f"  Clean data : {BASE / 'datasets_clean'}", logfile)
    log(f"  Datasets   : {BASE / 'datasets'}", logfile)
    log(f"  Results    : {BASE / 'results'}", logfile)
    log("=" * 65, logfile)

print("\nAll done. Press Enter to close.", flush=True)
input()

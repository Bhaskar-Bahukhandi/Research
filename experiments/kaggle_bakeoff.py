"""Write a Kaggle notebook (+ kernel-metadata.json) for one session of the Phase 3 bake-off.

    python experiments/kaggle_bakeoff.py OUT_DIR SESSION BUDGET_HOURS JOBS [PREVIOUS_SLUG]
    kaggle kernels push -p OUT_DIR -t SECONDS --accelerator NvidiaTeslaT4

JOBS is a comma-separated list of CANDIDATE:trials:RANGE or CANDIDATE:seeds, e.g.
"rotate:trials:0-19,quate:trials:0-19" or "rotate:seeds,cayley:seeds,quate:seeds".
Protocol: notes/protocol.md, seventh note of 2026-11-22. As in experiments/kaggle_s10.py:
  * the notebook runs the commit checked out here, which must be clean and pushed;
  * kb.txt is fetched from MetaQA's own Drive folder and checked against saved/metaqa/manifest.json
    (phase3.metaqa.load_data) before anything trains;
  * the preflight runs tests/test_phase3.py and must pass;
  * both GPUs claim jobs from one list (an O_EXCL file each), unfinished runs first;
  * a later session continues from PREVIOUS_SLUG's output (runs are resumable).
"""
import json
import os
import subprocess
import sys

out, session, budget, jobs_arg = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
prev = sys.argv[5] if len(sys.argv) > 5 else None
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
if git("status", "--porcelain"):
    sys.exit("working tree is dirty: commit first")
COMMIT = git("rev-parse", "--short", "HEAD")
if not git("branch", "-r", "--contains", COMMIT):
    sys.exit(f"{COMMIT} is not on any remote branch: push first")
JOBS = []
for item in jobs_arg.split(","):
    cand, kind, *rng = item.split(":")
    if cand not in ("rotate", "cayley", "quate") or kind not in ("trials", "seeds"):
        sys.exit(f"bad job {item!r}")
    if kind == "trials":
        lo, _, hi = rng[0].partition("-")
        JOBS += [[cand, "trials", str(i)] for i in range(int(lo), int(hi or lo) + 1)]
    else:
        JOBS += [[cand, "seeds", str(s)] for s in (1, 2)]
KB_URL = ("https://drive.usercontent.google.com/download?id=0B-36Uca2AvwhMUFHeUpTRWp5eUU"
          "&resourcekey=0-rcKwcmfpRl7uH17ZXU_M5g&export=download")
cell = r'''# RTVM-100M Phase 3 page-rotation bake-off, session SESSION (code at commit COMMIT). Resumable.
import glob, json, os, shutil, subprocess, sys, time
T0 = time.time()
BUDGET_H = BUDGETH
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-3000:], flush=True)
    return r.returncode
sh("git clone -q https://github.com/Bhaskar-Bahukhandi/Research /tmp/repo && git -C /tmp/repo checkout -q COMMIT && git -C /tmp/repo log --oneline -1")
sh("pip install -q pykeen==1.11.1 2>&1 | tail -2; nvidia-smi -L")
os.makedirs("/tmp/repo/data/metaqa", exist_ok=True)
sh("curl -sSL -o /tmp/repo/data/metaqa/kb.txt 'KBURL' && sha256sum /tmp/repo/data/metaqa/kb.txt")
chk = subprocess.run([sys.executable, "-c", "from phase3.metaqa import load_data; d = load_data(); print(len(d.train), len(d.v_tune), len(d.v_select))"],
                     cwd="/tmp/repo", capture_output=True, text=True)
print("data check:", (chk.stdout + chk.stderr)[-800:], flush=True)
if chk.returncode != 0:
    raise SystemExit("kb.txt does not match saved/metaqa/manifest.json; not starting")
pf = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_phase3.py"],
                    cwd="/tmp/repo", capture_output=True, text=True)
print("preflight:", (pf.stdout + pf.stderr)[-1500:], flush=True)
if pf.returncode != 0:
    raise SystemExit("preflight failed: tests/test_phase3.py; not starting")
RUNS = "/kaggle/working/runs/bakeoff"
prev = sorted(glob.glob("/kaggle/input/**/runs/bakeoff", recursive=True))
if prev:
    shutil.copytree(prev[0], RUNS, dirs_exist_ok=True)
    print("continued from", prev[0], flush=True)
os.makedirs(RUNS, exist_ok=True)
JOBS = JOBLIST
def name(j):
    return f"{j[0]}/trial_{int(j[2]):02d}" if j[1] == "trials" else f"{j[0]}/seed_{j[2]}"
JOBS.sort(key=lambda j: not os.path.exists(f"{RUNS}/{name(j)}.ckpt"))          # partial runs first
WORKER = r"""
import json, os, subprocess, sys, time
gpu, t_end, jobs = sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3])
RUNS, CLAIMS = sys.argv[4], "/tmp/claims"
env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONWARNINGS": "ignore"}
log = open(f"/kaggle/working/gpu{gpu}.log", "a")
def name(j):
    return f"{j[0]}/trial_{int(j[2]):02d}" if j[1] == "trials" else f"{j[0]}/seed_{j[2]}"
def complete(j):
    try:
        return json.load(open(f"{RUNS}/{name(j)}.json")).get("complete", False)
    except (OSError, ValueError):
        return False
def claim(j):
    try:
        os.close(os.open(f"{CLAIMS}/{name(j).replace('/', '_')}", os.O_CREAT | os.O_EXCL)); return True
    except FileExistsError:
        return False
while True:
    left = (t_end - time.time()) / 3600
    if left < 0.05:
        break
    job = next((j for j in jobs if not complete(j) and claim(j)), None)
    if job is None:
        break
    c = ["experiments/bakeoff.py", "run", "--candidate", job[0], f"--{job[1]}", job[2], "--device", "cuda",
         "--out", RUNS, "--max-hours", f"{left:.3f}"]
    log.write(f"=== {time.strftime('%H:%M:%S')} {' '.join(c)}\n"); log.flush()
    r = subprocess.run([sys.executable, *c], cwd="/tmp/repo", env=env, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"=== exit {r.returncode}\n"); log.flush()
"""
open("/tmp/worker.py", "w").write(WORKER)
os.makedirs("/tmp/claims", exist_ok=True)
t_end = T0 + BUDGET_H * 3600
procs = [subprocess.Popen([sys.executable, "/tmp/worker.py", g, str(t_end), json.dumps(JOBS), RUNS]) for g in ("0", "1")]
for p in procs:
    p.wait()
sh("grep -hE 'done|stopped|Error|Traceback|exit [1-9]|s/epoch' /kaggle/working/gpu*.log | tail -60")
sh(f"du -sh {RUNS}")
print(f"session wall time {(time.time() - T0) / 3600:.2f} h", flush=True)
'''
for k, v in (("SESSION", str(session)), ("COMMIT", COMMIT), ("BUDGETH", repr(budget)), ("KBURL", KB_URL),
             ("JOBLIST", json.dumps(JOBS))):
    cell = cell.replace(k, v)
compile(cell, "cell", "exec")
compile(cell.split('WORKER = r"""', 1)[1].split('"""', 1)[0], "worker", "exec")
slug = f"rtvm-p3-bakeoff-s{session}"
nb = {"cells": [{"cell_type": "code", "execution_count": None, "id": "bakeoff", "metadata": {}, "outputs": [], "source": cell}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(out, exist_ok=True)
open(os.path.join(out, slug + ".ipynb"), "w").write(json.dumps(nb))
json.dump({"id": f"bhaskarbahukhandi/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
           "kernel_sources": [f"bhaskarbahukhandi/{prev}"] if prev else []},
          open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
print("wrote", slug, f"({len(JOBS)} jobs)")

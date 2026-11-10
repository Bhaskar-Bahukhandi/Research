"""Write a Kaggle notebook (+ kernel-metadata.json) for one session of the G2 RotatE search.

    python experiments/kaggle_rotate_search.py OUT_DIR SESSION BUDGET_HOURS [PREVIOUS_SLUG]
    kaggle kernels push -p OUT_DIR -t 43200 --accelerator NvidiaTeslaT4

Session N attaches session N-1's output (PREVIOUS_SLUG, e.g. rtvm-g2-rotate-s1) and continues every
run from its checkpoint. Used 2026-11-10: s1 = 0.25 h shakedown, s2 = 11 h.
"""
import json, os, sys
out, session = sys.argv[1], int(sys.argv[2])
budget = float(sys.argv[3])
prev = sys.argv[4] if len(sys.argv) > 4 else None
COMMIT = "c5c44e2"
cell = r'''# RTVM-100M G2: RotatE random search on 2x T4, session SESSION (code at commit COMMIT).
# Resumable: finished runs are skipped, unfinished runs continue from their checkpoints.
# GPU 0: FB15k-237 trials 0-9, then FB seed 1.
# GPU 1: WN18RR trials 0-19, WN seeds 1 and 2, then FB15k-237 trials 10-19, then FB seed 2.
# A seed command fails harmlessly until all 20 trials of its dataset are complete.
import glob, os, shutil, subprocess, sys, time
T0 = time.time()
BUDGET_H = BUDGETH                                     # hours; Kaggle GPU sessions end at 12 h
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-3000:], flush=True)
sh("git clone -q https://github.com/Bhaskar-Bahukhandi/Research /tmp/repo && git -C /tmp/repo checkout -q COMMIT && git -C /tmp/repo log --oneline -1")
sh("pip install -q pykeen==1.11.1 2>&1 | tail -2; nvidia-smi -L")
RUNS = "/kaggle/working/runs/g2"
prev = sorted(glob.glob("/kaggle/input/**/runs/g2", recursive=True))
if prev:                                               # continue the previous session's search
    shutil.copytree(prev[0], RUNS, dirs_exist_ok=True)
    print("continued from", prev[0], flush=True)
os.makedirs(RUNS, exist_ok=True)
def cmd(model, ds, *job):
    return ["experiments/g2_search.py", "--model", model, "--dataset", ds, *job, "--device", "cuda", "--out", RUNS]
QUEUES = {
    "0": [cmd("rotate", "fb15k237", "--trials", "0-9"), cmd("rotate", "fb15k237", "--seeds", "1")],
    "1": [cmd("rotate", "wn18rr", "--trials", "0-19"), cmd("rotate", "wn18rr", "--seeds", "1", "2"),
          cmd("rotate", "fb15k237", "--trials", "10-19"), cmd("rotate", "fb15k237", "--seeds", "2")],
}
WORKER = r"""
import os, subprocess, sys, time, json
gpu, t_end, queue = sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3])
env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONWARNINGS": "ignore"}
log = open(f"/kaggle/working/gpu{gpu}.log", "a")
for c in queue:
    left = (t_end - time.time()) / 3600
    if left < 0.1:
        break
    log.write(f"=== {time.strftime('%H:%M:%S')} {' '.join(c)} (--max-hours {left:.2f})\n"); log.flush()
    r = subprocess.run([sys.executable, *c, "--max-hours", f"{left:.3f}"], cwd="/tmp/repo", env=env,
                       stdout=log, stderr=subprocess.STDOUT)
    log.write(f"=== exit {r.returncode}\n"); log.flush()
"""
open("/tmp/worker.py", "w").write(WORKER)
import json
t_end = T0 + BUDGET_H * 3600
procs = [subprocess.Popen([sys.executable, "/tmp/worker.py", g, str(t_end), json.dumps(q)]) for g, q in QUEUES.items()]
for p in procs:
    p.wait()
for ds in ("fb15k237", "wn18rr"):
    sh(f"cd /tmp/repo && python experiments/g2_search.py --model rotate --dataset {ds} --status --out {RUNS} 2>&1 | grep -v Warn")
sh("grep -hE 'done|stopped|Error|Traceback|exit [1-9]' /kaggle/working/gpu*.log | tail -60")
sh(f"du -sh {RUNS}")
print(f"session wall time {(time.time() - T0) / 3600:.2f} h", flush=True)
'''.replace("SESSION", str(session)).replace("COMMIT", COMMIT).replace("BUDGETH", repr(budget))
compile(cell, "cell", "exec")
slug = f"rtvm-g2-rotate-s{session}"
nb = {"cells": [{"cell_type": "code", "execution_count": None, "id": "search", "metadata": {}, "outputs": [], "source": cell}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(out, exist_ok=True)
open(os.path.join(out, slug + ".ipynb"), "w").write(json.dumps(nb))
json.dump({"id": f"bhaskarbahukhandi/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
           "kernel_sources": [f"bhaskarbahukhandi/{prev}"] if prev else []},
          open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
print("wrote", slug)

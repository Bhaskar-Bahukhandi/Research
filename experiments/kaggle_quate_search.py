"""Write a Kaggle notebook (+ kernel-metadata.json) for one session of the G2 QuatE search.

    python experiments/kaggle_quate_search.py OUT_DIR SESSION BUDGET_HOURS [PREVIOUS_SLUG]
    kaggle kernels push -p OUT_DIR -t SECONDS --accelerator NvidiaTeslaT4

Same scheduler as experiments/kaggle_rotate_search.py (one shared job list for both GPUs, unfinished
runs first, seed runs wait for all 20 trials of their dataset), for the QuatE of protocol note 2026-11-16
(quate_official, Adagrad, exact-filtered Bernoulli negatives; configs/g2_quate.yaml). Like RotatE it
needs only the committed saved/g2 files, so there is no Drive download. Differences from RotatE:

  * the preflight is the cuda case of tests/test_g2search.py::test_quate_official_adagrad_resumes_exactly
    (checkpoint and resume of the G2 QuatE setting on the GPU);
  * session 1 doubles as the timing run the note asks for before the search: give it a short budget
    (e.g. 1 h), read the s/epoch of trial 0 on each dataset, then size the later sessions. Its runs are
    part of the search and continue in session 2.

Pass PREVIOUS_SLUG (e.g. rtvm-g2-quate-s1) for every session after the first; push with the kaggle CLI,
because kernel_sources is what attaches the previous output (the MCP save_notebook data source did not,
2026-11-15). The notebook runs the commit checked out here, which must be clean and pushed.
"""
import json, os, subprocess, sys
out, session = sys.argv[1], int(sys.argv[2])
budget = float(sys.argv[3])
prev = sys.argv[4] if len(sys.argv) > 4 else None
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
if git("status", "--porcelain"):
    sys.exit("working tree is dirty: commit first")
COMMIT = git("rev-parse", "--short", "HEAD")
if not git("branch", "-r", "--contains", COMMIT):
    sys.exit(f"{COMMIT} is not on any remote branch: push first")
cell = r'''# RTVM-100M G2: QuatE random search on 2x T4, session SESSION (code at commit COMMIT).
# Resumable: finished runs are skipped, unfinished runs continue from their checkpoints.
# Both GPUs pull from one job list (FB15k-237 and WN18RR trials, then seeds 1 and 2 of each).
import glob, json, os, shutil, subprocess, sys, time
T0 = time.time()
BUDGET_H = BUDGETH                                     # hours; Kaggle GPU sessions end at 12 h
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-3000:], flush=True)
    return r.returncode
sh("git clone -q https://github.com/Bhaskar-Bahukhandi/Research /tmp/repo && git -C /tmp/repo checkout -q COMMIT && git -C /tmp/repo log --oneline -1")
sh("pip install -q pykeen==1.11.1 2>&1 | tail -2; nvidia-smi -L")
pf = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_g2search.py",
                     "-k", "quate_official_adagrad_resumes and cuda", "-rs"], cwd="/tmp/repo", capture_output=True, text=True)
print("preflight:", (pf.stdout + pf.stderr)[-1500:], flush=True)
if pf.returncode != 0 or "1 passed" not in pf.stdout:      # a skip (no GPU visible) also stops the session
    raise SystemExit("preflight failed: cuda QuatE resume test; not starting the search")
RUNS = "/kaggle/working/runs/g2"
prev = sorted(glob.glob("/kaggle/input/**/runs/g2", recursive=True))
if prev:                                               # continue the previous session's search
    shutil.copytree(prev[0], RUNS, dirs_exist_ok=True)
    print("continued from", prev[0], flush=True)
os.makedirs(RUNS, exist_ok=True)
# One shared job list; each GPU worker claims the next job (an O_EXCL file in /tmp/claims) and runs it alone.
# Unfinished runs (a checkpoint, no complete json) go first. A seed job waits until all 20 trials of its
# dataset are complete; a worker with nothing claimable waits while the other one still runs a job.
DS = ("fb15k237", "wn18rr")
JOBS = [[ds, "trial", i] for i in range(20) for ds in DS] + [[ds, "seed", s] for s in (1, 2) for ds in DS]
def stem(j):
    return f"trial_{j[2]:02d}" if j[1] == "trial" else f"seed_{j[2]}"
JOBS.sort(key=lambda j: not os.path.exists(f"{RUNS}/{j[0]}/quate/{stem(j)}.ckpt"))    # stable: partial first
WORKER = r"""
import json, os, subprocess, sys, time
gpu, t_end, jobs = sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3])
RUNS, CLAIMS = sys.argv[4], "/tmp/claims"
env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONWARNINGS": "ignore"}
log = open(f"/kaggle/working/gpu{gpu}.log", "a")
stem = lambda j: f"trial_{j[2]:02d}" if j[1] == "trial" else f"seed_{j[2]}"
tag = lambda j: f"{j[0]}-{stem(j)}"
def complete(ds, name):
    try:
        return json.load(open(f"{RUNS}/{ds}/quate/{name}.json")).get("complete", False)
    except (OSError, ValueError):
        return False
def ready(j):
    return j[1] == "trial" or all(complete(j[0], f"trial_{i:02d}") for i in range(20))
def claim(j):
    try:
        os.close(os.open(f"{CLAIMS}/{tag(j)}", os.O_CREAT | os.O_EXCL)); return True
    except FileExistsError:
        return False
while True:
    left = (t_end - time.time()) / 3600
    if left < 0.1:
        break
    todo = [j for j in jobs if not complete(j[0], stem(j)) and not os.path.exists(f"{CLAIMS}/{tag(j)}")]
    if not todo:
        break
    job = next((j for j in todo if ready(j) and claim(j)), None)
    if job is None:                       # only blocked seed jobs left: wait while another job is running
        if not any(f.endswith(".running") for f in os.listdir(CLAIMS)):
            break
        time.sleep(60); continue
    run = ["--trials", f"{job[2]}-{job[2]}"] if job[1] == "trial" else ["--seeds", str(job[2])]
    c = ["experiments/g2_search.py", "--model", "quate", "--dataset", job[0], *run, "--device", "cuda",
         "--out", RUNS, "--max-hours", f"{left:.3f}"]
    marker = f"{CLAIMS}/{tag(job)}.running"
    open(marker, "w").close()
    log.write(f"=== {time.strftime('%H:%M:%S')} {' '.join(c)}\n"); log.flush()
    r = subprocess.run([sys.executable, *c], cwd="/tmp/repo", env=env, stdout=log, stderr=subprocess.STDOUT)
    os.remove(marker)
    log.write(f"=== exit {r.returncode}\n"); log.flush()
"""
open("/tmp/worker.py", "w").write(WORKER)
os.makedirs("/tmp/claims", exist_ok=True)
t_end = T0 + BUDGET_H * 3600
procs = [subprocess.Popen([sys.executable, "/tmp/worker.py", g, str(t_end), json.dumps(JOBS), RUNS]) for g in ("0", "1")]
for p in procs:
    p.wait()
for ds in DS:
    sh(f"cd /tmp/repo && python experiments/g2_search.py --model quate --dataset {ds} --status --out {RUNS} 2>&1 | grep -v Warn")
sh("grep -hE 'done|stopped|Error|Traceback|exit [1-9]' /kaggle/working/gpu*.log | tail -60")
sh("grep -hE 's/epoch' /kaggle/working/gpu*.log | tail -6")    # timing: read this after session 1
sh(f"du -sh {RUNS}")
print(f"session wall time {(time.time() - T0) / 3600:.2f} h", flush=True)
'''.replace("SESSION", str(session)).replace("COMMIT", COMMIT).replace("BUDGETH", repr(budget))
compile(cell, "cell", "exec")
compile(cell.split('WORKER = r"""', 1)[1].split('"""', 1)[0], "worker", "exec")
slug = f"rtvm-g2-quate-s{session}"
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

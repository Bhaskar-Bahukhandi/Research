"""Write a Kaggle notebook (+ kernel-metadata.json) for one session of G2 secondary analysis S10.

    python experiments/kaggle_s10.py OUT_DIR SESSION BUDGET_HOURS [PREVIOUS_SLUG]
    kaggle kernels push -p OUT_DIR -t SECONDS --accelerator NvidiaTeslaT4

S10 trains the holonomy model with dense rotations (rank 32; experiments/g2_s10.py), seeds 0, 1 and 2,
on FB15k-237. As in experiments/kaggle_holonomy_search.py:
  * the path files come from the user's public Drive folder via gdown, and load_data checks them
    against the committed manifest;
  * the preflight is the cuda case of tests/test_g2search.py::test_holonomy_deadline_stops_after_a_checkpoint;
  * both GPUs claim jobs from one list (an O_EXCL file each), unfinished runs first;
  * a later session continues from PREVIOUS_SLUG's output.

The notebook runs the commit checked out here, which must be clean and pushed.
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
DRIVE = "https://drive.google.com/drive/folders/1PSjC29inTy6YcpZwD4fyB-XbuFQiZT_K"
cell = r'''# RTVM-100M G2 secondary analysis S10: dense-rotation holonomy model, session SESSION (code at commit COMMIT).
# Seeds 0, 1, 2 on FB15k-237 with the best trial's values at rank 32. Resumable.
import glob, json, os, shutil, subprocess, sys, time
T0 = time.time()
BUDGET_H = BUDGETH                                     # hours; Kaggle GPU sessions end at 12 h
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-3000:], flush=True)
    return r.returncode
sh("git clone -q https://github.com/Bhaskar-Bahukhandi/Research /tmp/repo && git -C /tmp/repo checkout -q COMMIT && git -C /tmp/repo log --oneline -1")
sh("pip install -q pykeen==1.11.1 gdown 2>&1 | tail -2; nvidia-smi -L")
pf = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_g2search.py",
                     "-k", "holonomy_deadline and cuda", "-rs"], cwd="/tmp/repo", capture_output=True, text=True)
print("preflight:", (pf.stdout + pf.stderr)[-1500:], flush=True)
if pf.returncode != 0 or "1 passed" not in pf.stdout:      # a skip (no GPU visible) also stops the session
    raise SystemExit("preflight failed: cuda holonomy resume test; not starting S10")
sh("gdown --folder -q 'DRIVE' -O /tmp/gd 2>&1 | tail -5; du -sh /tmp/gd")
hits = glob.glob("/tmp/gd/**/g2/fb15k237/train_paths.npy", recursive=True)
if len(hits) != 1:
    raise SystemExit(f"expected one g2/fb15k237/train_paths.npy in the Drive download, found {hits}")
DATA = hits[0].rsplit("/fb15k237/", 1)[0]
print("data folder:", DATA, f"({(time.time() - T0) / 60:.1f} min)", flush=True)
RUNS = "/kaggle/working/runs/g2s10"
prev = sorted(glob.glob("/kaggle/input/**/runs/g2s10", recursive=True))
if prev:
    shutil.copytree(prev[0], RUNS, dirs_exist_ok=True)
    print("continued from", prev[0], flush=True)
os.makedirs(RUNS, exist_ok=True)
JOBS = [0, 1, 2]
JOBS.sort(key=lambda s: not os.path.exists(f"{RUNS}/fb15k237/holonomy_dense/seed_{s}.ckpt"))    # partial first
WORKER = r"""
import json, os, subprocess, sys, time
gpu, t_end, jobs = sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3])
RUNS, DATA, CLAIMS = sys.argv[4], sys.argv[5], "/tmp/claims"
env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONWARNINGS": "ignore"}
log = open(f"/kaggle/working/gpu{gpu}.log", "a")
def complete(s):
    try:
        return json.load(open(f"{RUNS}/fb15k237/holonomy_dense/seed_{s}.json")).get("complete", False)
    except (OSError, ValueError):
        return False
def claim(s):
    try:
        os.close(os.open(f"{CLAIMS}/seed_{s}", os.O_CREAT | os.O_EXCL)); return True
    except FileExistsError:
        return False
while True:
    left = (t_end - time.time()) / 3600
    if left < 0.1:
        break
    job = next((s for s in jobs if not complete(s) and claim(s)), None)
    if job is None:
        break
    c = ["experiments/g2_s10.py", "--seeds", str(job), "--device", "cuda", "--data", DATA, "--out", RUNS,
         "--max-hours", f"{left:.3f}"]
    log.write(f"=== {time.strftime('%H:%M:%S')} {' '.join(c)}\n"); log.flush()
    r = subprocess.run([sys.executable, *c], cwd="/tmp/repo", env=env, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"=== exit {r.returncode}\n"); log.flush()
"""
open("/tmp/worker.py", "w").write(WORKER)
os.makedirs("/tmp/claims", exist_ok=True)
t_end = T0 + BUDGET_H * 3600
procs = [subprocess.Popen([sys.executable, "/tmp/worker.py", g, str(t_end), json.dumps(JOBS), RUNS, DATA]) for g in ("0", "1")]
for p in procs:
    p.wait()
sh("grep -hE 'parameters|done|stopped|Error|Traceback|exit [1-9]|s/epoch' /kaggle/working/gpu*.log | tail -40")
sh(f"du -sh {RUNS}")
print(f"session wall time {(time.time() - T0) / 3600:.2f} h", flush=True)
'''.replace("SESSION", str(session)).replace("COMMIT", COMMIT).replace("BUDGETH", repr(budget)).replace("DRIVE", DRIVE)
compile(cell, "cell", "exec")
compile(cell.split('WORKER = r"""', 1)[1].split('"""', 1)[0], "worker", "exec")
slug = f"rtvm-g2-s10-s{session}"
nb = {"cells": [{"cell_type": "code", "execution_count": None, "id": "s10", "metadata": {}, "outputs": [], "source": cell}],
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

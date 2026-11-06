"""Structure-only check of G2's negatives (run 2026-11-06, before any model was trained).

    .venv/Scripts/python experiments/structure_check.py

For a sample of 3,000 test positives per dataset, draws 5 head and 5 tail
role-matched negatives as in notes/protocol.md (without the fallback: it only
counts the sides, positive x head/tail, whose role pools are too small), then compares how often
positives and negatives have a direct training edge, their entity degrees, how
often they close a cycle of length <= 2 and <= 3, how many walks they have, and
what share of walks are length 3. Uses no model.
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_stats import adjacency  # noqa: E402
from pykeen.datasets import get_dataset  # noqa: E402


def walks(M, h, t, chunk=256):
    d = np.asarray(M[h, t]).ravel()
    w2 = np.zeros(len(h))
    w3 = np.zeros(len(h))
    for s in range(0, len(h), chunk):
        H = M[h[s:s + chunk]]
        T = M[t[s:s + chunk]].toarray()
        w2[s:s + chunk] = np.asarray(H.multiply(T).sum(axis=1)).ravel()
        w3[s:s + chunk] = ((H @ M).toarray() * T).sum(axis=1)
    return d, w2, w3


for key in ("fb15k237", "wn18rr"):
    ds = get_dataset(dataset=key)
    tr = ds.training.mapped_triples.numpy()
    te = ds.testing.mapped_triples.numpy()
    va = ds.validation.mapped_triples.numpy()
    known = set(map(tuple, np.concatenate([tr, va, te]).tolist()))
    heads, tails = defaultdict(set), defaultdict(set)
    for h, r, t in tr.tolist():
        heads[r].add(h)
        tails[r].add(t)
    heads = {r: np.array(sorted(v)) for r, v in heads.items()}
    tails = {r: np.array(sorted(v)) for r, v in tails.items()}
    rng = np.random.default_rng(0)
    pos = te[rng.choice(len(te), size=min(3000, len(te)), replace=False)]
    neg, short = [], 0
    for h, r, t in pos.tolist():
        for side, pool in (("h", heads[r]), ("t", tails[r])):
            got = 0
            for c in rng.permutation(pool):
                cand = (int(c), r, t) if side == "h" else (h, r, int(c))
                if cand not in known:
                    neg.append(cand)
                    got += 1
                    if got == 5:
                        break
            short += got < 5
    neg = np.array(neg)
    M = adjacency(tr, ds.num_entities)
    degree = np.asarray(M.sum(axis=1)).ravel()   # training facts touching each entity
    train_facts = set(map(tuple, tr.tolist()))
    reverse = np.mean([(t, r, h) in train_facts for h, r, t in pos.tolist()])
    print(f"{key:9s} positives whose exact reverse (t, r, h) is in training: {reverse:.3f}")
    for name, arr in (("positives", pos), ("negatives", neg)):
        direct = np.asarray(M[arr[:, 0], arr[:, 2]]).ravel() > 0
        print(f"{key:9s} {name:9s} direct training edge between h and t: {direct.mean():.3f}   "
              f"median degree h / t: {np.median(degree[arr[:, 0]]):.0f} / {np.median(degree[arr[:, 2]]):.0f}")
        d, w2, w3 = walks(M, arr[:, 0], arr[:, 2])
        cov = (d + w2 + w3) > 0
        tot = d + w2 + w3
        frac3 = np.median((w3 / np.maximum(tot, 1))[cov])
        print(f"{key:9s} {name:9s} n={len(arr):6d} covered(L<=3)={cov.mean():.3f} covered(L<=2)={((d + w2) > 0).mean():.3f} "
              f"median walks={np.median(tot[cov]):8.0f} median share of length-3 walks={frac3:.3f}")
    print(f"{key:9s} sides (positive x head/tail) with < 5 valid role-matched replacements: {short} of {2 * len(pos)}")

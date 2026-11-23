"""The Phase 3 page-rotation bake-off: candidates, their search spaces, and one trainer.

Protocol: notes/protocol.md, seventh note of 2026-11-22.
  * A "rotate": the G2 RotatE (rtvm.baselines, rotate_official), complex dimension 64.
  * B "cayley": the holonomy model's rank-r Cayley block rotations (rtvm.holonomy_model), d = 128 as
    two blocks of 64, trained on the single-triple loss only (self-adversarial NSSA on
    -||U_r psi_h - psi_t||, uniform head-or-tail negatives as in G2); no cycle terms.
  * C "quate": the G2 QuatE (quate_official), quaternion dimension 32.
  * Search spaces and fixed values: configs/g2_*.yaml, except: B's rank is searched over
    {1, 2, 4, 8} and its cycle-term values (alpha1, alpha2, margin, cycles_per_triple) are dropped;
    QuatE's number of negatives is its FB15k-237 value (10); dimensions as above.
  * 20 random-search trials per candidate (rtvm.g2search.sample_trials, trial seed 0, model seed 0).
  * Early stopping on V_tune filtered MRR (realistic ranks, both directions, filtered against the
    whole knowledge base): every 10 epochs, patience 5, cap 1,000; the best weights are kept.
  * TF32 in training, strict FP32 in evaluation.
"""
from __future__ import annotations

import copy
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from rtvm.g2search import CONFIGS, file_sha256, fixed_values, load_spec, sample_trials  # noqa: F401 (file_sha256 re-exported)

from .linkpred import cayley_scorers, link_prediction, pykeen_scorers, quate_scorers
from .metaqa import BakeoffData

CANDIDATES = ("rotate", "cayley", "quate")
N_TRIALS, TRIAL_SEED = 20, 0
DIMS = {"rotate": 64, "quate": 32}                 # 128 reals per entity; B has d = 128 (2 x 64)
G2_SOURCE = {"rotate": "rotate", "cayley": "holonomy", "quate": "quate"}
SPEC_DATASET = "fb15k237"                          # whose per-dataset fixed values are used (QuatE's 10 negatives)
B_DROPPED = ("alpha1", "alpha2", "margin", "cycles_per_triple")
B_RANKS = [1, 2, 4, 8]


def spec(candidate: str, root: str | Path = ".") -> tuple[dict, dict]:
    """(fixed values, search space) of a candidate, derived from its G2 config as the note says."""
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}; expected one of {CANDIDATES}")
    g2 = load_spec(Path(root) / CONFIGS[G2_SOURCE[candidate]])
    fixed, search = fixed_values(g2, SPEC_DATASET), copy.deepcopy(g2["search"])
    if candidate == "cayley":
        for name in B_DROPPED:
            fixed.pop(name, None)
            search.pop(name, None)
        search["rank"] = {"choice": list(B_RANKS)}
    else:
        fixed["model"], fixed["dim"] = g2["model"], DIMS[candidate]
    return fixed, search


def trials(candidate: str, root: str | Path = ".") -> list[dict]:
    """The candidate's 20 trials: fixed values plus one sampled value per searched name."""
    fixed, search = spec(candidate, root)
    return [{**fixed, **t} for t in sample_trials(search, N_TRIALS, TRIAL_SEED)]


@dataclass
class RunState:
    epoch: int = 0
    best_mrr: float = float("-inf")
    best_epoch: int = 0
    bad_evals: int = 0
    history: list = field(default_factory=list)


def _set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def evaluate(candidate: str, model, triples: np.ndarray, data: BakeoffData) -> dict:
    """Filtered realistic link prediction on `triples`, against every knowledge-base triple."""
    _set_tf32(False)
    model.eval()
    scorers = {"rotate": pykeen_scorers, "quate": quate_scorers, "cayley": cayley_scorers}[candidate](model)
    res = link_prediction(*scorers, triples, data.kb.triples, data.kb.num_entities, data.kb.num_relations)
    model.train()
    return res


# ---------------------------------------------------------------- the two model families

class _PyKEENRun:
    """RotatE or QuatE through rtvm.baselines' model definitions and PyKEEN's sLCWA loop."""

    def __init__(self, values: dict, seed: int, data: BakeoffData, device: torch.device):
        from pykeen.training import SLCWATrainingLoop
        from pykeen.triples import TriplesFactory
        from pykeen.utils import set_random_seed

        from rtvm.baselines import OPTIMIZERS, BaselineConfig, ExactFilterer, build_model
        names = set(BaselineConfig.__dataclass_fields__)
        cfg = BaselineConfig(**{k: v for k, v in values.items() if k in names}, seed=seed)
        self.config = cfg
        tf = TriplesFactory(mapped_triples=torch.as_tensor(data.train, dtype=torch.long),
                            entity_to_id={e: i for i, e in enumerate(data.kb.entities)},
                            relation_to_id={r: i for i, r in enumerate(data.kb.relations)})
        self.tf = tf
        set_random_seed(seed)
        self.model = build_model(cfg, tf).to(device)
        self.loop = SLCWATrainingLoop(
            model=self.model, triples_factory=tf,
            optimizer=getattr(torch.optim, OPTIMIZERS[cfg.optimizer])(self.model.get_grad_params(), lr=cfg.lr),
            negative_sampler="bernoulli" if cfg.model == "quate_official" else "basic",
            negative_sampler_kwargs=dict(num_negs_per_pos=cfg.num_negatives, filtered=cfg.filtered_negatives,
                                         filterer=ExactFilterer if cfg.filtered_negatives else None,
                                         filterer_kwargs=dict(device=str(device)) if cfg.filtered_negatives else None),
            automatic_memory_optimization=False)

    def train_to(self, epoch_now: int, target: int, batch_size: int) -> float:
        losses = self.loop.train(triples_factory=self.tf, num_epochs=target, batch_size=batch_size,
                                 continue_training=epoch_now > 0, use_tqdm=False)
        return float(np.mean(losses[-(target - epoch_now):]))

    def state(self) -> dict:
        return dict(model=self.model.state_dict(), optimizer=self.loop.optimizer.state_dict())

    def load(self, saved: dict, epoch: int) -> None:
        self.model.load_state_dict(saved["model"])
        self.loop.optimizer.load_state_dict(saved["optimizer"])
        self.loop._epoch = epoch


class _CayleyRun:
    """B: HolonomyModel with only the single-triple NSSA term (the G2 trainer's first term)."""

    def __init__(self, values: dict, seed: int, data: BakeoffData, device: torch.device):
        from rtvm.holonomy_model import HolonomyModel
        self.values, self.seed, self.data, self.device = values, seed, data, device
        gen = torch.Generator().manual_seed(seed)
        self.model = HolonomyModel(data.kb.num_entities, data.kb.num_relations, rank=values["rank"],
                                   rotation_init=values["rotation_init"], entity_init=values["entity_init"],
                                   generator=gen).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=values["lr"])

    def batch_loss(self, tr: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        from rtvm.holonomy_model import nssa_loss
        v, m = self.values, self.model
        as_t = lambda a: torch.as_tensor(np.asarray(a), device=self.device)   # noqa: E731
        k = v["num_negatives"]
        ent = rng.integers(self.data.kb.num_entities, size=(len(tr), k))
        head_side = rng.random((len(tr), k)) < 0.5
        neg_h = np.where(head_side, ent, tr[:, [0]])
        neg_t = np.where(head_side, tr[:, [2]], ent)
        R = m.rotations()
        pos_d = m.single_distance(as_t(tr[:, 0]), as_t(tr[:, 1]), as_t(tr[:, 2]), R)
        neg_d = m.single_distance(as_t(neg_h), as_t(tr[:, 1]), as_t(neg_t), R)
        return nssa_loss(pos_d, neg_d, gamma=v["gamma"], temperature=v["temperature"])

    def train_to(self, epoch_now: int, target: int, batch_size: int) -> float:
        train = self.data.train
        total, batches = 0.0, 0
        for epoch in range(epoch_now + 1, target + 1):
            rng = np.random.default_rng([self.seed, epoch])          # per-epoch stream: resume-safe
            self.model.train()
            for idx in np.array_split(rng.permutation(len(train)), max(1, int(np.ceil(len(train) / batch_size)))):
                loss = self.batch_loss(train[idx], rng)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total, batches = total + loss.item(), batches + 1
        return total / max(1, batches)

    def state(self) -> dict:
        return dict(model=self.model.state_dict(), optimizer=self.opt.state_dict())

    def load(self, saved: dict, epoch: int) -> None:
        self.model.load_state_dict(saved["model"])
        self.opt.load_state_dict(saved["optimizer"])


def _rng_state(device) -> dict:
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _set_rng_state(saved: dict, device) -> None:
    torch.set_rng_state(saved["torch"].cpu())
    np.random.set_state(saved["numpy"])
    random.setstate(saved["python"])
    if saved["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda"].cpu(), device)


def train(candidate: str, values: dict, seed: int, data: BakeoffData, device: str = "cpu",
          checkpoint: str | Path | None = None, log=print, deadline: float | None = None) -> dict:
    """Train one run to early stopping on V_tune MRR, resuming from `checkpoint` if it exists.

    deadline: a time.time() value; once passed, training stops right after the next evaluation's
    checkpoint, and the result has complete=False (calling again continues the run).
    """
    import pykeen

    from rtvm.utils import count_parameters
    device = torch.device(device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    run = (_CayleyRun if candidate == "cayley" else _PyKEENRun)(values, seed, data, device)
    state, best_weights = RunState(), None
    key = dict(candidate=candidate, values=values, seed=seed)

    ckpt = Path(checkpoint) if checkpoint else None
    if ckpt and ckpt.exists():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        if saved["key"] != key:
            raise ValueError(f"checkpoint {ckpt} was made for a different run")
        state, best_weights = RunState(**saved["state"]), saved["best_weights"]
        run.load(saved["run"], state.epoch)
        _set_rng_state(saved["rng"], device)
        log(f"resumed at epoch {state.epoch}")

    every, patience, cap = values["eval_every"], values["patience"], values["max_epochs"]
    start = time.time()
    first_epoch, last_time, last_epoch = state.epoch, start, state.epoch
    while state.epoch < cap and state.bad_evals < patience:
        target = min(cap, (state.epoch // every + 1) * every)
        _set_tf32(values["tf32"])
        loss = run.train_to(state.epoch, target, values["batch_size"])
        state.epoch = target
        res = evaluate(candidate, run.model, data.v_tune, data)
        mrr = res["mrr"]
        improved = mrr > state.best_mrr                       # NaN never improves
        if improved:
            state.best_mrr, state.best_epoch, state.bad_evals = mrr, target, 0
            best_weights = {k: v.detach().cpu().clone() for k, v in run.model.state_dict().items()}
        else:
            state.bad_evals += 1
        state.history.append(dict(epoch=target, loss=loss, v_tune_mrr=mrr, v_tune_hits_at_10=res["hits_at_10"]))
        if ckpt:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt.with_suffix(".tmp")
            torch.save(dict(key=key, run=run.state(), state=asdict(state), best_weights=best_weights,
                            rng=_rng_state(device)), tmp)
            tmp.replace(ckpt)
        now = time.time()
        per_epoch = (now - last_time) / max(1, target - last_epoch)
        last_time, last_epoch = now, target
        log(f"epoch {target}: loss {loss:.4f}  V_tune MRR {mrr:.4f}{'  best' if improved else ''}"
            f"  [{per_epoch:.1f} s/epoch]")
        if deadline is not None and now >= deadline:
            break

    complete = state.epoch >= cap or state.bad_evals >= patience
    if best_weights is not None:
        run.model.load_state_dict(best_weights)
    return dict(candidate=candidate, values=values, seed=seed, best_v_tune_mrr=state.best_mrr,
                best_epoch=state.best_epoch, epochs_run=state.epoch, complete=complete,
                epochs_this_process=state.epoch - first_epoch, history=state.history,
                train_seconds=time.time() - start, parameters=count_parameters(run.model),
                device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                tf32_training=values["tf32"], evaluation="strict FP32",
                versions=dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                              pykeen=pykeen.get_version()),
                data_provenance=data.provenance, model=run.model)


def build(candidate: str, values: dict, num_entities: int, num_relations: int, entities=None, relations=None):
    """An untrained model of the candidate with the right shapes, for loading saved weights."""
    from pykeen.triples import TriplesFactory

    from rtvm.baselines import BaselineConfig, build_model
    from rtvm.holonomy_model import HolonomyModel
    if candidate == "cayley":
        return HolonomyModel(num_entities, num_relations, rank=values["rank"],
                             rotation_init=values["rotation_init"], entity_init=values["entity_init"])
    names = set(BaselineConfig.__dataclass_fields__)
    cfg = BaselineConfig(**{k: v for k, v in values.items() if k in names})
    entities = entities or [str(i) for i in range(num_entities)]
    relations = relations or [str(i) for i in range(num_relations)]
    tf = TriplesFactory(mapped_triples=torch.zeros((1, 3), dtype=torch.long),
                        entity_to_id={e: i for i, e in enumerate(entities)},
                        relation_to_id={r: i for i, r in enumerate(relations)})
    return build_model(cfg, tf)

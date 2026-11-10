"""Training loop for the G2 holonomy model (one run = one config + one seed).

Rules implemented (notes/protocol.md):
  * training triples and training cycles only; 8 cycles per covered training triple per
    epoch, drawn from its saved paths (own edge already excluded), each paired with one
    corrupted cycle (the same path closed with another relation, see corrupt_closing);
  * TF32 allowed in training if config.tf32; every evaluation in strict FP32;
  * every `eval_every` epochs, evaluate on V1 the length-and-relation-stratified cycle AUC
    (protocol note of 2026-11-10; it selects weights and drives early stopping) and, for
    transparency, the original metric: grouped AUC of the triple holonomy score, covered
    positives against their covered negatives; stop after `patience` evaluations without
    improvement; at most `max_epochs` epochs;
  * every run records its GPU and precision.

The data arrays come from the G2 data build (negatives, paths). Use `train(...)`
from a script or a test; checkpoints resume automatically.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .g2eval import grouped_auc, stratified_cycle_auc
from .holonomy_model import (HolonomyModel, corrupt_closing, cycle_defect, cycle_defects, cycle_loss, fact_key,
                             nssa_loss, path_transport, triple_scores_from_cycles)


@dataclass
class HolonomyConfig:
    # searched (configs/g2_holonomy.yaml gives the ranges)
    rank: int = 2
    lr: float = 1e-3
    rotation_init: float = 0.1
    entity_init: float = 1.0
    gamma: float = 9.0
    temperature: float = 1.0
    alpha1: float = 0.1
    alpha2: float = 0.1
    margin: float = 2.0
    # fixed
    num_negatives: int = 256
    batch_size: int = 1024
    cycles_per_triple: int = 8
    eval_every: int = 10
    patience: int = 5
    max_epochs: int = 1000
    tf32: bool = True
    seed: int = 0


@dataclass
class HolonomyData:
    """Arrays for one dataset. Steps are table indices (holonomy_model.steps_from_codes)."""
    num_entities: int
    num_relations: int
    train: np.ndarray            # (n, 3)
    train_steps: np.ndarray      # (n, K, MAX_LEN), own edge excluded, filled slots first
    v1_pos: np.ndarray           # (m, 3)
    v1_pos_steps: np.ndarray     # (m, K, MAX_LEN)
    v1_neg: np.ndarray           # (m, 10, 3)
    v1_neg_steps: np.ndarray     # (m, 10, K, MAX_LEN)


@dataclass
class RunState:
    epoch: int = 0                       # epochs completed
    best_auc: float = float("-inf")
    best_epoch: int = 0
    bad_evals: int = 0
    history: list = field(default_factory=list)


def _set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def evaluate_v1(model: HolonomyModel, data: HolonomyData, device) -> dict[str, float]:
    """V1 metrics in strict FP32, from one pass over every saved cycle.

    stratified: the selection metric (protocol note of 2026-11-10).
    triple_grouped_auc: the original metric, grouped AUC of -triple holonomy score with
    covered positives against their covered negatives; logged, not used for selection.
    """
    _set_tf32(False)
    model.eval()
    m, k = data.v1_neg.shape[:2]
    pos_rel, neg_rel = data.v1_pos[:, 1], data.v1_neg[..., 1].reshape(-1)
    p_tri, p_def, p_len = cycle_defects(model, pos_rel, data.v1_pos_steps, device)
    n_tri, n_def, n_len = cycle_defects(model, neg_rel, data.v1_neg_steps.reshape(m * k, *data.v1_neg_steps.shape[2:]),
                                        device)
    pos, pos_cov = triple_scores_from_cycles(m, p_tri, p_def)
    neg, neg_cov = triple_scores_from_cycles(m * k, n_tri, n_def)
    return dict(
        stratified=stratified_cycle_auc(p_def, p_len, pos_rel[p_tri], n_def, n_len, neg_rel[n_tri]),
        triple_grouped_auc=grouped_auc(-pos, -neg.reshape(m, k), pos_cov, neg_cov.reshape(m, k)))


def train(config: HolonomyConfig, data: HolonomyData, device: str = "cpu",
          checkpoint: str | Path | None = None, log=print) -> dict:
    device = torch.device(device)
    gen = torch.Generator().manual_seed(config.seed)
    model = HolonomyModel(data.num_entities, data.num_relations, rank=config.rank,
                          rotation_init=config.rotation_init, entity_init=config.entity_init,
                          generator=gen).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    state, best_weights = RunState(), None

    ckpt = Path(checkpoint) if checkpoint else None
    if ckpt and ckpt.exists():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        if saved["config"] != asdict(config):
            raise ValueError(f"checkpoint {ckpt} was made with a different config")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        state, best_weights = RunState(**saved["state"]), saved["best_weights"]
        log(f"resumed at epoch {state.epoch}")

    pad = 2 * data.num_relations
    n_paths = (data.train_steps != pad).any(axis=-1).sum(axis=-1)          # filled slots per triple
    train_keys = np.sort(fact_key(data.train[:, 0], data.train[:, 1], data.train[:, 2],
                                  data.num_entities, data.num_relations))
    start = time.time()

    while state.epoch < config.max_epochs and state.bad_evals < config.patience:
        epoch = state.epoch + 1
        rng = np.random.default_rng([config.seed, epoch])     # per-epoch stream: resume-safe
        _set_tf32(config.tf32)
        model.train()
        total, batches = 0.0, 0
        for idx in np.array_split(rng.permutation(len(data.train)),
                                  max(1, int(np.ceil(len(data.train) / config.batch_size)))):
            loss = _batch_loss(model, data, idx, n_paths, train_keys, config, rng, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, batches = total + loss.item(), batches + 1
        state.epoch = epoch

        if epoch % config.eval_every == 0 or epoch == config.max_epochs:
            metrics = evaluate_v1(model, data, device)
            auc = metrics["stratified"]                          # selection metric (note 2026-11-10)
            improved = auc > state.best_auc
            if improved:
                state.best_auc, state.best_epoch, state.bad_evals = auc, epoch, 0
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                state.bad_evals += 1
            state.history.append(dict(epoch=epoch, loss=total / batches, v1_auc=auc,
                                      v1_triple_grouped_auc=metrics["triple_grouped_auc"]))
            log(f"epoch {epoch}: loss {total / batches:.4f}  V1 stratified cycle AUC {auc:.4f}"
                f"  (triple grouped AUC {metrics['triple_grouped_auc']:.4f}){'  best' if improved else ''}")
            if ckpt:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                tmp = ckpt.with_suffix(".tmp")
                torch.save(dict(config=asdict(config), model=model.state_dict(), optimizer=opt.state_dict(),
                                state=asdict(state), best_weights=best_weights), tmp)
                tmp.replace(ckpt)                        # atomic: a disconnect never leaves half a file

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return dict(config=asdict(config), best_v1_auc=state.best_auc, best_epoch=state.best_epoch,
                epochs_run=state.epoch, history=state.history, train_seconds=time.time() - start,
                device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                tf32_training=config.tf32, evaluation="strict FP32",
                versions=dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__),
                model=model)


def _batch_loss(model, data, idx, n_paths, train_keys, config, rng, device):
    tr = data.train[idx]
    R = model.rotations()
    as_t = lambda a: torch.as_tensor(np.asarray(a), device=device)   # noqa: E731

    # single-triple term: uniform head-or-tail corruption, k negatives per positive
    k = config.num_negatives
    ent = rng.integers(data.num_entities, size=(len(idx), k))
    head_side = rng.random((len(idx), k)) < 0.5
    neg_h = np.where(head_side, ent, tr[:, [0]])
    neg_t = np.where(head_side, tr[:, [2]], ent)
    pos_d = model.single_distance(as_t(tr[:, 0]), as_t(tr[:, 1]), as_t(tr[:, 2]), R)
    neg_d = model.single_distance(as_t(neg_h), as_t(np.repeat(tr[:, [1]], k, axis=1)), as_t(neg_t), R)
    loss = nssa_loss(pos_d, neg_d, gamma=config.gamma, temperature=config.temperature)

    # cycle term: C paths per covered triple, drawn with replacement from its saved paths
    cov = idx[n_paths[idx] > 0]
    if len(cov):
        C = config.cycles_per_triple
        slot = (rng.random((len(cov), C)) * n_paths[cov, None]).astype(np.int64)
        steps = data.train_steps[cov[:, None], slot].reshape(-1, data.train_steps.shape[-1])
        h, r, t = (np.repeat(data.train[cov, j], C) for j in range(3))
        r_bad = corrupt_closing(h, r, t, train_keys, data.num_entities, data.num_relations, rng)
        table = model.table(R)
        U_p = path_transport(table, as_t(steps))
        loss = loss + cycle_loss(cycle_defect(table, U_p, as_t(r)), cycle_defect(table, U_p, as_t(r_bad)),
                                 alpha1=config.alpha1, alpha2=config.alpha2, margin=config.margin)
    return loss


def load_data(dataset: str, saved: str | Path = "saved/g2", data: str | Path = "data/g2") -> HolonomyData:
    """HolonomyData from the files written by experiments/g2_build_data.py (V1 part only).

    Never loads test files: the training loop sees training triples and V1 only.
    """
    from pykeen.datasets import get_dataset

    from .holonomy_model import steps_from_codes

    ds = get_dataset(dataset=dataset)
    train = ds.training.mapped_triples.numpy().astype(np.int64)
    valid = ds.validation.mapped_triples.numpy().astype(np.int64)
    small, large = Path(saved) / dataset, Path(data) / dataset
    # Path codes are row indices into PyKEEN's arrays: refuse if the id mapping or row order changed.
    import hashlib
    import json
    inputs = json.loads((small / "manifest.json").read_text())["inputs"]
    for name, arr in (("train", train), ("valid", valid)):
        digest = hashlib.sha256(np.ascontiguousarray(arr, dtype=np.int64).tobytes()).hexdigest()
        if digest != inputs[f"{name}_sha256"]:
            raise ValueError(f"{dataset} {name} triples differ from the manifest: rebuild or fix the PyKEEN version")
    v1 = np.load(small / "valid_v1_index.npy")
    neg = np.load(small / "valid_negatives.npy").astype(np.int64)
    m, k = neg.shape[:2]
    def to_steps(codes):
        return steps_from_codes(codes, train[:, 1], ds.num_relations)

    pos_paths = np.load(large / "valid_pos_paths.npy")[v1]
    neg_paths = np.load(large / "valid_neg_paths.npy").reshape(m, k, *pos_paths.shape[1:])[v1]
    return HolonomyData(
        num_entities=ds.num_entities, num_relations=ds.num_relations, train=train,
        train_steps=to_steps(np.load(large / "train_paths.npy")),
        v1_pos=valid[v1], v1_pos_steps=to_steps(pos_paths),
        v1_neg=neg[v1], v1_neg_steps=to_steps(neg_paths))


def load_config(path: str | Path, **overrides) -> HolonomyConfig:
    """Fixed values from a YAML file's `fixed:` section, plus a trial's sampled values."""
    import yaml
    spec = yaml.safe_load(Path(path).read_text())
    values = dict(spec.get("fixed", {}))
    values.update(overrides)
    return HolonomyConfig(**values)


def copy_config(config: HolonomyConfig, **changes) -> HolonomyConfig:
    return HolonomyConfig(**{**asdict(copy.deepcopy(config)), **changes})

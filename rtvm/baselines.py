"""G2 baselines (RotatE, QuatE) trained in PyKEEN under the shared G2 rules.

Rules (notes/protocol.md, Models; notes of 2026-11-07 and 2026-11-08):
  * RotatE is the official-code definition (PhaseRotatEInteraction below), with training
    negatives filtered against the training triples; QuatE is PyKEEN's (provisional until
    its official-code check, configs/g2_quate.yaml);
  * every `eval_every` epochs, the V1 score is the grouped AUC of PyKEEN's score_hrt for
    every V1 positive against its 10 saved negatives (no coverage filter: that applies to
    the holonomy model only); stop after `patience` evaluations without improvement; at
    most `max_epochs` epochs; the best V1 weights are kept;
  * TF32 allowed in training if config.tf32; evaluation in strict FP32;
  * every run records its GPU, precision and library versions.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA starts (deterministic cuBLAS)
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from pykeen.nn.modules import Interaction
from pykeen.sampling.filtering import Filterer

from .g2eval import grouped_auc

MODELS = ("rotate_official", "quate_pykeen")


@dataclass
class BaselineConfig:
    model: str = "rotate_official"
    dim: int = 68
    # searched (configs/g2_rotate.yaml, configs/g2_quate.yaml)
    lr: float = 1e-3
    gamma: float = 9.0                 # RotatE: NSSA margin and init range
    temperature: float = 1.0           # RotatE: self-adversarial temperature
    regularizer_weight: float = 0.01   # QuatE: L2 weight on entities and relations
    # fixed
    num_negatives: int = 256
    batch_size: int = 1024
    filtered_negatives: bool = True
    loss: str = "nssa"
    eval_every: int = 10
    patience: int = 5
    max_epochs: int = 1000
    tf32: bool = True
    seed: int = 0


@dataclass
class BaselineData:
    training: object                   # pykeen TriplesFactory (training triples only)
    v1_pos: np.ndarray                 # (m, 3)
    v1_neg: np.ndarray                 # (m, 10, 3)
    provenance: dict | None = None


@dataclass
class RunState:
    epoch: int = 0
    best_auc: float = float("-inf")
    best_epoch: int = 0
    bad_evals: int = 0
    history: list = field(default_factory=list)


class PhaseRotatEInteraction(Interaction):
    """Official RotatE score: -sum_i |h_i * exp(i * theta_i) - t_i|, theta = r * pi / range.

    Moved here unchanged from experiments/reproduce_rotate.py (which now imports it), so the
    reproduction check and G2 use the same class (protocol note 2026-11-08).
    """

    def __init__(self, embedding_range: float):
        super().__init__()
        self.scale = math.pi / embedding_range

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rotation = torch.polar(torch.ones_like(r), r * self.scale)
        return -(h * rotation - t).abs().sum(dim=-1)


class ExactFilterer(Filterer):
    """Exact training-negative filter: a sorted array of training-triple keys and a binary search.

    Replaces PyKEEN's default BloomFilterer (protocol note 2026-11-10, exact filtering). The Bloom
    filter is approximate (it also drops about 0.1% of valid negatives on FB15k-237) and was half
    of RotatE's epoch time on a Kaggle T4; this one gives the same answer as PyKEEN's exact
    PythonSetFilterer. Like every PyKEEN filterer, a filtered negative is dropped (masked), not
    redrawn.

    device: where the sorted keys live and the search runs. PyKEEN samples negatives on the CPU
    inside its data loader; on a Kaggle T4 an exact CPU search still cost about 8 s per FB15k-237
    epoch, so train() passes the training device and the search runs there (same answer).
    """

    def __init__(self, mapped_triples: torch.Tensor, device: str | torch.device | None = None):
        super().__init__()
        mapped_triples = torch.as_tensor(mapped_triples, dtype=torch.long)
        self.num_entities = int(mapped_triples[:, [0, 2]].max()) + 1
        self.num_relations = int(mapped_triples[:, 1].max()) + 1
        keys = torch.sort(self._key(mapped_triples)).values
        self.register_buffer("keys", keys.to(device) if device is not None else keys, persistent=False)

    def _key(self, triples: torch.Tensor) -> torch.Tensor:
        return (triples[..., 0] * self.num_relations + triples[..., 1]) * self.num_entities + triples[..., 2]

    def contains(self, batch: torch.Tensor) -> torch.Tensor:
        b = batch.to(self.keys.device, non_blocking=True)
        key = self._key(b)
        # a candidate with an id beyond the training maximum must not alias a training key
        inside = (b[..., 0] < self.num_entities) & (b[..., 2] < self.num_entities) & (b[..., 1] < self.num_relations)
        pos = torch.searchsorted(self.keys, key).clamp_max(len(self.keys) - 1)
        return ((self.keys[pos] == key) & inside).to(batch.device)


def build_model(config: BaselineConfig, triples_factory):
    """The G2 baseline named by config.model, sized config.dim, seeded config.seed."""
    from pykeen.models import ERModel, QuatE

    if config.model == "rotate_official":
        rng = (config.gamma + 2.0) / config.dim       # official: (gamma + epsilon) / hidden_dim, epsilon = 2
        init = functools.partial(torch.nn.init.uniform_, a=-rng, b=rng)
        return ERModel(
            triples_factory=triples_factory,
            interaction=PhaseRotatEInteraction(rng),
            entity_representations_kwargs=dict(shape=config.dim, dtype=torch.cfloat, initializer=init),
            relation_representations_kwargs=dict(shape=config.dim, initializer=init),
            loss="nssa", loss_kwargs=dict(margin=config.gamma, adversarial_temperature=config.temperature,
                                          reduction="mean"),
            random_seed=config.seed)
    if config.model == "quate_pykeen":
        reg = dict(weight=config.regularizer_weight, p=2.0, normalize=True)
        return QuatE(triples_factory=triples_factory, embedding_dim=config.dim,
                     entity_regularizer_kwargs=reg, relation_regularizer_kwargs=reg,
                     loss="bcewithlogits", loss_kwargs=dict(reduction="mean"), random_seed=config.seed)
    raise ValueError(f"unknown baseline {config.model!r}; expected one of {MODELS}")


def _set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


@torch.no_grad()
def score_triples(model, triples: np.ndarray, device, chunk: int = 65536) -> np.ndarray:
    """PyKEEN score_hrt (higher = more plausible) for (n, 3) id triples, strict FP32."""
    _set_tf32(False)
    model.eval()
    out = np.empty(len(triples))
    for s in range(0, len(triples), chunk):
        batch = torch.as_tensor(triples[s:s + chunk], dtype=torch.long, device=device)
        out[s:s + chunk] = model.score_hrt(batch).view(-1).double().cpu().numpy()
    return out


def evaluate_v1(model, data: BaselineData, device) -> float:
    m, k = data.v1_neg.shape[:2]
    pos = score_triples(model, data.v1_pos, device)
    neg = score_triples(model, data.v1_neg.reshape(-1, 3), device).reshape(m, k)
    return grouped_auc(pos, neg)


def _rng_state(device) -> dict:
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _set_rng_state(saved: dict, device) -> None:
    torch.set_rng_state(saved["torch"])
    np.random.set_state(saved["numpy"])
    random.setstate(saved["python"])
    if saved["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda"], device)


def train(config: BaselineConfig, data: BaselineData, device: str = "cpu",
          checkpoint: str | Path | None = None, log=print, deadline: float | None = None) -> dict:
    """Train one baseline run to early stopping; same contract as rtvm.train_holonomy.train."""
    import pykeen
    from pykeen.training import SLCWATrainingLoop
    from pykeen.utils import set_random_seed

    device = torch.device(device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    # warn_only: some PyKEEN/CUDA ops have no deterministic kernel; the run still logs the setting
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_random_seed(config.seed)
    model = build_model(config, data.training).to(device)
    loop = SLCWATrainingLoop(
        model=model, triples_factory=data.training,
        optimizer=torch.optim.Adam(model.get_grad_params(), lr=config.lr),
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=config.num_negatives, filtered=config.filtered_negatives,
                                     filterer=ExactFilterer if config.filtered_negatives else None,
                                     filterer_kwargs=dict(device=str(device)) if config.filtered_negatives else None),
        automatic_memory_optimization=False)       # fixed batch size: no probing, no sub-batching
    state, best_weights = RunState(), None

    ckpt = Path(checkpoint) if checkpoint else None
    if ckpt and ckpt.exists():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        if saved["config"] != asdict(config):
            raise ValueError(f"checkpoint {ckpt} was made with a different config")
        model.load_state_dict(saved["model"])
        loop.optimizer.load_state_dict(saved["optimizer"])
        loop._epoch = saved["state"]["epoch"]
        state, best_weights = RunState(**saved["state"]), saved["best_weights"]
        _set_rng_state(saved["rng"], device)
        log(f"resumed at epoch {state.epoch}")

    start = time.time()
    first_epoch, last_eval_time, last_eval_epoch = state.epoch, start, state.epoch
    while state.epoch < config.max_epochs and state.bad_evals < config.patience:
        target = min(config.max_epochs, (state.epoch // config.eval_every + 1) * config.eval_every)
        _set_tf32(config.tf32)
        losses = loop.train(triples_factory=data.training, num_epochs=target, batch_size=config.batch_size,
                            continue_training=state.epoch > 0, use_tqdm=False)
        losses = losses[-(target - state.epoch):]        # PyKEEN returns every epoch since the loop was made
        state.epoch = target
        auc = evaluate_v1(model, data, device)
        improved = auc > state.best_auc                  # NaN never improves
        if improved:
            state.best_auc, state.best_epoch, state.bad_evals = auc, target, 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state.bad_evals += 1
        state.history.append(dict(epoch=target, loss=float(np.mean(losses)), v1_auc=auc))
        if ckpt:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt.with_suffix(".tmp")
            torch.save(dict(config=asdict(config), model=model.state_dict(), optimizer=loop.optimizer.state_dict(),
                            state=asdict(state), best_weights=best_weights, rng=_rng_state(device)), tmp)
            tmp.replace(ckpt)
        now = time.time()
        per_epoch = (now - last_eval_time) / max(1, target - last_eval_epoch)
        last_eval_time, last_eval_epoch = now, target
        log(f"epoch {target}: loss {np.mean(losses):.4f}  V1 grouped AUC {auc:.4f}"
            f"{'  best' if improved else ''}  [{per_epoch:.1f} s/epoch]")
        if deadline is not None and now >= deadline:
            break

    complete = state.epoch >= config.max_epochs or state.bad_evals >= config.patience
    if best_weights is not None:
        model.load_state_dict(best_weights)
    from .utils import count_parameters
    return dict(config=asdict(config), best_v1_auc=state.best_auc, best_epoch=state.best_epoch,
                epochs_run=state.epoch, complete=complete, epochs_this_process=state.epoch - first_epoch,
                history=state.history, train_seconds=time.time() - start,
                parameters=count_parameters(model),
                device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                tf32_training=config.tf32, evaluation="strict FP32", deterministic_algorithms="warn_only",
                versions=dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                              pykeen=pykeen.get_version()),
                data_provenance=data.provenance, model=model)


def load_data(dataset: str, saved: str | Path = "saved/g2") -> BaselineData:
    """Training triples, V1 positives and their saved negatives, checked against the manifest.

    Needs only the committed files under saved/g2 (no path files). Never loads test files.
    """
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=dataset)
    train_ = ds.training.mapped_triples.numpy().astype(np.int64)
    valid = ds.validation.mapped_triples.numpy().astype(np.int64)
    small = Path(saved) / dataset
    manifest_path = small / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, arr in (("train", train_), ("valid", valid)):
        digest = hashlib.sha256(np.ascontiguousarray(arr, dtype=np.int64).tobytes()).hexdigest()
        if digest != manifest["inputs"][f"{name}_sha256"]:
            raise ValueError(f"{dataset} {name} triples differ from the manifest: fix the PyKEEN version")

    def load(name: str) -> np.ndarray:
        entry = manifest["files"][name]
        path = small / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{path} does not match {manifest_path}")
        return np.load(path)

    v1 = load("valid_v1_index")
    neg = load("valid_negatives").astype(np.int64)
    return BaselineData(training=ds.training, v1_pos=valid[v1], v1_neg=neg[v1],
                        provenance=dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                        data_code_commit=manifest["code"]["commit"]))

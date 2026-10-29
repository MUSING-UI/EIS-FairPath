"""Load a portable, frozen teacher without historical experiment metadata."""
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path

import torch

from .models import StructureAwareHGTTeacher
from .sampling import SamplerConfig


@dataclass(frozen=True)
class LoadedTeacher:
    model: StructureAwareHGTTeacher
    sampler: SamplerConfig
    dataset: str
    architecture: dict
    sha256: str


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_teacher(path, *, dataset=None):
    """Load exact calibrated state on CPU, preserving its dtype and freezing it."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "eis-fairpath-teacher-v1":
        raise ValueError("Unsupported teacher checkpoint format")
    if dataset is not None and checkpoint["dataset"] != dataset:
        raise ValueError(f"Checkpoint dataset {checkpoint['dataset']!r} does not match {dataset!r}")
    architecture = dict(checkpoint["architecture"])
    sampler_fields = {field.name for field in fields(SamplerConfig)}
    sampler = SamplerConfig(**{key: value for key, value in checkpoint["sampler"].items()
                               if key in sampler_fields})
    if sampler.layers != architecture["layers"]:
        raise ValueError("Teacher and sampler layers must agree")
    model = StructureAwareHGTTeacher(**architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
    model.eval().requires_grad_(False)
    return LoadedTeacher(model, sampler, checkpoint["dataset"], architecture, file_sha256(path))

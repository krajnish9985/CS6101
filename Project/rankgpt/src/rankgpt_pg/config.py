"""Run configuration: YAML on disk, dataclass in memory, JSON in the output.

A run is fully described by one of these objects, and the resolved object
is written next to the results. That is what lets a number in the final
table be traced back to the model, prompt, depth and seed that produced
it weeks later.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .llm import LLMConfig


@dataclass
class RunConfig:
    """One permutation-generation experiment."""

    run_name: str
    model: LLMConfig

    data: str = "data/scifact/scifact_reranking_data.jsonl"
    dataset: str = "scifact"

    depth: int = 20
    prompt: str = "paper"
    max_passage_words: int = 300

    # None means every query in the file. Set it low to rehearse a run.
    limit: Optional[int] = None

    # Prompts are generated in chunks so that progress is visible in the
    # Slurm log and partial results survive a crash late in a long run.
    chunk_size: int = 64

    output_root: str = "results/runs"

    notes: str = ""

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["model"] = self.model.to_dict()
        return payload


def load_run_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
    """Read a YAML config, apply CLI overrides, and validate the result.

    ``overrides`` holds only keys the user passed explicitly on the
    command line, so a flag left unset never clobbers the YAML value.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    model_section = raw.pop("model", None)
    if not model_section:
        raise ValueError(f"{path} is missing the required 'model' section")

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key in {"model_path", "max_model_len", "tensor_parallel_size",
                   "gpu_memory_utilization", "enforce_eager", "seed"}:
            model_section[key] = value
        else:
            raw[key] = value

    unknown_model = set(model_section) - set(LLMConfig.__dataclass_fields__)
    if unknown_model:
        raise ValueError(f"unknown model key(s) in {path}: {sorted(unknown_model)}")

    unknown_run = set(raw) - set(RunConfig.__dataclass_fields__)
    if unknown_run:
        raise ValueError(f"unknown key(s) in {path}: {sorted(unknown_run)}")

    config = RunConfig(model=LLMConfig(**model_section), **raw)

    if config.depth <= 0:
        raise ValueError(f"depth must be positive, got {config.depth}")
    if config.chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {config.chunk_size}")

    return config


def git_commit() -> str:
    """Current commit, for provenance. Returns 'unknown' outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def git_is_dirty() -> bool:
    """Whether tracked files have uncommitted changes.

    Recorded in the output because a result produced from a dirty tree is
    not reproducible from the commit hash alone.
    """
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(output.strip())
    except Exception:
        return False
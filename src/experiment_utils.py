from __future__ import annotations
from typing import Self  # Python 3.11+

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")

def append_jsonl(path: Path, o: dict):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(o) + "\n")

def update_manifest(manifest_path, status: str, **metadata: Any) -> None:
    """Update the run status while retaining previously written metadata."""
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(metadata)
    manifest["status"] = status
    manifest[f"{status}_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    return manifest

@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    """Filesystem layout for one experiment run."""

    root: Path
    config: Path
    metrics: Path
    checkpoints: Path
    logs: Path

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        config: Path | None = None,
        metrics: Path | None = None,
        checkpoints: Path | None = None,
        logs: Path | None = None,
    ) -> Self:
        root_path = Path(root)
        return cls(
            root=root_path,
            config=config or root_path / "config",
            metrics=metrics or root_path / "metrics",
            checkpoints=checkpoints or root_path / "checkpoints",
            logs=logs or root_path / "logs",
        )

    @property
    def hyperparameters(self) -> Path:
        return self.config / "hyperparameters.json"

    @property
    def training_metrics(self) -> Path:
        return self.metrics / "training.jsonl"

    @property
    def validation_metrics(self) -> Path:
        return self.metrics / "validation.jsonl"

    @property
    def log_file(self) -> Path:
        return self.logs / "run.log"

    @property
    def manifest(self) -> Path:
        return self.root / "run_manifest.json"

    def mkdir(self):
        for directory in (
            self.root,
            self.config,
            self.metrics,
            self.checkpoints,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, step: int) -> Path:
        return self.checkpoints / f"checkpoint-{step:06d}"

# class Experiment:
#     """Manage the standard directory layout and metadata for one run."""

#     def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
#         self.paths = ExperimentPaths(
#             root=output_dir,
#             config=output_dir / "config",
#             metrics=output_dir / "metrics",
#             checkpoints=output_dir / "checkpoints",
#             logs=output_dir / "logs",
#         )
#         for directory in (
#             self.paths.root,
#             self.paths.config,
#             self.paths.metrics,
#             self.paths.checkpoints,
#             self.paths.logs,
#         ):
#             directory.mkdir(parents=True, exist_ok=True)
#         write_json(self.paths.hyperparameters, config)

#     def update_manifest(self, status: str, **metadata: Any) -> None:
#         """Update the run status while retaining previously written metadata."""
#         manifest: dict[str, Any] = {}
#         if self.paths.manifest.exists():
#             manifest = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
#         manifest.update(metadata)
#         manifest["status"] = status
#         manifest[f"{status}_at"] = datetime.now(timezone.utc).isoformat()
#         write_json(self.paths.manifest, manifest)


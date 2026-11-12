"""Artifact file layout and (de)serialization for a single run directory."""

import hashlib
from pathlib import Path

from pydantic import BaseModel

from soccervision.schemas.artifacts import DetectionsArtifact, RunMetadata

ANNOTATED_VIDEO_FILE = "annotated.mp4"
DETECTIONS_FILE = "detections.json"
METADATA_FILE = "metadata.json"


def write_json_atomic(path: Path, model: BaseModel, *, indent: int | None = None) -> None:
    """Write via a temp file + rename so readers never observe a half-written artifact."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(model.model_dump_json(indent=indent), encoding="utf-8")
    tmp.replace(path)


def load_detections(path: Path) -> DetectionsArtifact:
    return DetectionsArtifact.model_validate_json(path.read_bytes())


def load_run_metadata(path: Path) -> RunMetadata:
    return RunMetadata.model_validate_json(path.read_bytes())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

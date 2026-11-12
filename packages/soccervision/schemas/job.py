"""Pipeline job model and its state machine.

    PENDING ──► PROCESSING ──► COMPLETED
       │            │
       └────────────┴──────► FAILED

PENDING → FAILED covers jobs whose worker could not be launched. Jobs still PENDING when the
server restarts are resumed, not failed. COMPLETED and FAILED are terminal.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


_ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.PENDING: frozenset({JobStatus.PROCESSING, JobStatus.FAILED}),
    JobStatus.PROCESSING: frozenset({JobStatus.COMPLETED, JobStatus.FAILED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
}


class InvalidJobTransitionError(ValueError):
    def __init__(self, current: JobStatus, target: JobStatus) -> None:
        super().__init__(f"Invalid job transition {current.value} -> {target.value}")
        self.current = current
        self.target = target


def allowed_sources(target: JobStatus) -> frozenset[JobStatus]:
    """States from which ``target`` may be entered."""
    return frozenset(src for src, dsts in _ALLOWED_TRANSITIONS.items() if target in dsts)


def ensure_transition(current: JobStatus, target: JobStatus) -> None:
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidJobTransitionError(current, target)


def is_terminal(status: JobStatus) -> bool:
    return not _ALLOWED_TRANSITIONS[status]


class PipelineJob(BaseModel):
    job_id: str
    video_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0, description="Fraction of decoded frames processed.")
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    error_code: str | None = None
    pipeline_version: str
    configuration: dict[str, Any]
    artifact_paths: dict[str, str] = Field(
        default_factory=dict, description="Artifact name -> path relative to the artifact root."
    )

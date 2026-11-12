from fastapi import APIRouter
from fastapi.responses import FileResponse

from soccervision_api.api.deps import DbDep, SettingsDep, require_job
from soccervision_api.errors import ApiError
from soccervision_api.schemas import ARTIFACT_NAMES, JobResponse
from soccervision_api.services.results import resolve_artifact

router = APIRouter(prefix="/jobs", tags=["jobs"])

_MEDIA_TYPES = {
    "annotated_video": "video/mp4",
    "detections": "application/json",
    "metadata": "application/json",
}
_NAME_BY_FILE = {filename: name for name, filename in ARTIFACT_NAMES.items()}


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: DbDep) -> JobResponse:
    return JobResponse.from_job(require_job(db, job_id))


@router.get("/{job_id}/artifacts/{filename}", response_class=FileResponse)
def get_artifact(job_id: str, filename: str, db: DbDep, settings: SettingsDep) -> FileResponse:
    name = _NAME_BY_FILE.get(filename)
    if name is None:
        raise ApiError(404, "artifact_not_found", f"Unknown artifact '{filename}'.")
    job = require_job(db, job_id)
    path = resolve_artifact(settings, job, name)
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES[name],
        filename=filename if name != "annotated_video" else None,
    )

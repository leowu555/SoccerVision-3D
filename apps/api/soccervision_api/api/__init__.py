from fastapi import APIRouter

from soccervision_api.api import jobs, system, videos
from soccervision_api.schemas import API_PREFIX

router = APIRouter(prefix=API_PREFIX)
router.include_router(system.router)
router.include_router(videos.router)
router.include_router(jobs.router)

__all__ = ["router"]

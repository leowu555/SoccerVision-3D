"""ASGI entrypoint: ``uvicorn soccervision_api.main:app``."""

import logging

from soccervision_api.app import create_app
from soccervision_api.settings import Settings

_settings = Settings()
logging.basicConfig(
    level=_settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

app = create_app(_settings)

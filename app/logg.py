import logging
import logging_loki
from utils import LOKI_ADDRESS

loki_handler = logging_loki.LokiHandler(
    url=f"http://{LOKI_ADDRESS}:3100/loki/api/v1/push",
    tags={"application": "fastapi"},
    auth=("username", "password"),
    version="2",
)

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
logger.addHandler(loki_handler)

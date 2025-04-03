import logging
import socket
import logging_loki
from utils import LOKI_ADDRESS

def is_loki_reachable(address, port=3100, timeout=2):
    try:
        with socket.create_connection((address, port), timeout):
            return True
    except OSError:
        return False

if LOKI_ADDRESS and is_loki_reachable(LOKI_ADDRESS, port=3100):
    loki_handler = logging_loki.LokiHandler(
        url=f"http://{LOKI_ADDRESS}:3100/loki/api/v1/push",
        tags={"application": "fastapi"},
        auth=("username", "password"),
        version="2",
    )
    logger = logging.getLogger("app_logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(loki_handler)
else:
    logger = logging.getLogger("uvicorn")

logger.info("[STARTUP] Aplicação FastAPI iniciada.")

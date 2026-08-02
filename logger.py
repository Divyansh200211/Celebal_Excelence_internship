from __future__ import annotations

import logging
from pathlib import Path

from src.config import EXECUTION_LOG_PATH


def get_logger(name: str = "rag_app") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_path = Path(EXECUTION_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(stream_handler)

    return logger

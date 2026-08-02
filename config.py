from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def sanitize_api_key(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().strip('"').strip("'")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "3"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-english-v3.0")
LLM_MODEL = os.getenv("LLM_MODEL", "command-r-08-2024")
VECTOR_DB_PATH = str(DATABASE_DIR / "metadata.json")
METADATA_PATH = str(DATABASE_DIR / "metadata.json")
EXECUTION_LOG_PATH = str(LOGS_DIR / "execution.log")
METRICS_REPORT_PATH = str(REPORTS_DIR / "system_metrics.md")
PINECONE_API_KEY = sanitize_api_key(os.getenv("PINECONE_API_KEY", ""))
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
COHERE_API_KEY = sanitize_api_key(os.getenv("COHERE_API_KEY", ""))


@dataclass
class AppConfig:
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k: int = TOP_K
    embedding_model: str = EMBEDDING_MODEL
    llm_model: str = LLM_MODEL
    vector_db_path: str = VECTOR_DB_PATH
    metadata_path: str = METADATA_PATH
    execution_log_path: str = EXECUTION_LOG_PATH
    metrics_report_path: str = METRICS_REPORT_PATH
    pinecone_api_key: str = PINECONE_API_KEY
    pinecone_environment: str = PINECONE_ENVIRONMENT
    pinecone_index_name: str = PINECONE_INDEX_NAME
    cohere_api_key: str = COHERE_API_KEY

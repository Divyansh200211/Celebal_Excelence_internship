from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.config import METRICS_REPORT_PATH


def write_metrics_report(metrics: Dict[str, Any]) -> None:
    report_path = Path(METRICS_REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# System Metrics Report

- Document Count: {metrics.get('document_count', 0)}
- Page Count: {metrics.get('page_count', 0)}
- Character Count: {metrics.get('character_count', 0)}
- Chunk Size: {metrics.get('chunk_size', 0)}
- Chunk Overlap: {metrics.get('chunk_overlap', 0)}
- Total Chunks: {metrics.get('total_chunks', 0)}
- Embedding Model: {metrics.get('embedding_model', 'N/A')}
- Embedding Dimensions: {metrics.get('embedding_dimensions', 'N/A')}
- Vector Database: {metrics.get('vector_db', 'N/A')}
- LLM: {metrics.get('llm_model', 'N/A')}
- Top-K: {metrics.get('top_k', 0)}
- Average Retrieval Time (s): {metrics.get('avg_retrieval_time', 0):.4f}
- Average Generation Time (s): {metrics.get('avg_generation_time', 0):.4f}
"""
    report_path.write_text(content, encoding="utf-8")

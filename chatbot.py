from __future__ import annotations

import os
import time
from typing import List, Dict, Any

import cohere
from dotenv import load_dotenv

from src.config import AppConfig, sanitize_api_key
from src.logger import get_logger

load_dotenv()
logger = get_logger("chatbot")


class RAGChatbot:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        api_key = sanitize_api_key(self.config.cohere_api_key or os.getenv("COHERE_API_KEY"))
        if not api_key:
            raise ValueError("Missing COHERE_API_KEY in environment variables")
        self.client = cohere.Client(api_key)

    def _get_model_candidates(self) -> List[str]:
        requested_model = (self.config.llm_model or os.getenv("LLM_MODEL") or "").strip()
        candidates: List[str] = []
        if requested_model:
            candidates.append(requested_model)
        for fallback_model in ("command-r-08-2024", "command-r-plus-08-2024", "command-r", "command-r-plus"):
            if fallback_model not in candidates:
                candidates.append(fallback_model)
        return candidates

    def build_prompt(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join([
            f"Source: {chunk.get('metadata', {}).get('filename', 'unknown')}\n{chunk['chunk']}"
            for chunk in context_chunks
        ])
        return f"""You are a grounded question answering assistant.
Answer ONLY using the provided retrieved context.
If the answer is not present in the retrieved context, respond exactly:
"I couldn't find that information in the provided documents."

Question: {question}

Retrieved Context:
{context_text}

Answer:"""

    def answer_question(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not context_chunks:
            return {
                "answer": "I couldn't find that information in the provided documents.",
                "response_time": 0.0,
            }

        prompt = self.build_prompt(question, context_chunks)
        start = time.time()
        last_error: Exception | None = None

        for model_name in self._get_model_candidates():
            try:
                response = self.client.generate(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.1,
                )
                answer = response.generations[0].text
                logger.info("LLM response generated with model %s", model_name)
                break
            except Exception as exc:
                last_error = exc
                try:
                    response = self.client.chat(model=model_name, message=prompt)
                    answer = response.text
                    logger.info("LLM response generated with model %s", model_name)
                    break
                except Exception as chat_exc:
                    last_error = chat_exc
                    continue
        else:
            raise RuntimeError(f"Unable to generate a response with Cohere: {last_error}") from last_error

        generation_time = time.time() - start
        logger.info("LLM response generated in %.4fs", generation_time)
        return {
            "answer": (answer or "I couldn't find that information in the provided documents.").strip(),
            "response_time": generation_time,
        }

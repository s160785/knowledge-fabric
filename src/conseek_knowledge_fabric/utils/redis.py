"""Redis memory and cache integration for conversation history, embeddings, and QA."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

try:
    import redis
except ImportError:
    redis = None

from conseek_knowledge_fabric.utils.logging import get_logger

logger = get_logger(__name__)


class RedisMemory:
    """Manages conversation history, query embeddings, and QA cache in Redis with in-memory fallback."""

    _shared_history: dict[str, list[dict[str, str]]] = {}
    _shared_embeddings: dict[str, list[float]] = {}
    _shared_qa: dict[str, dict[str, Any]] = {}

    def __init__(self, redis_url: str | None = None, ttl: int = 86400) -> None:
        self.ttl = ttl
        self.redis_client = None
        self._local_history = self._shared_history
        self._local_embeddings = self._shared_embeddings
        self._local_qa = self._shared_qa

        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if redis is not None and url:
            try:
                client = redis.from_url(url, decode_responses=True, socket_timeout=2)
                client.ping()
                self.redis_client = client
                logger.info("Connected to Redis cache at %s", url)
            except Exception as exc:
                logger.warning("Redis unavailable (%s); falling back to in-memory cache", exc)
                self.redis_client = None

    def get_history(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        if self.redis_client:
            try:
                key = f"ba_chat:history:{session_id}"
                items = self.redis_client.lrange(key, -limit, -1)
                logger.debug("Redis memory get_history fetched %d items for session_id=%s", len(items), session_id)
                return [json.loads(item) for item in items]
            except Exception as exc:
                logger.debug("Redis get_history error: %s", exc)
        local_items = self._local_history.get(session_id, [])[-limit:]
        logger.debug("In-memory fallback get_history fetched %d items for session_id=%s", len(local_items), session_id)
        return local_items

    def add_history(self, session_id: str, role: str, content: str, max_items: int = 20) -> None:
        entry = {"role": role, "content": content}
        if self.redis_client:
            try:
                key = f"ba_chat:history:{session_id}"
                self.redis_client.rpush(key, json.dumps(entry))
                self.redis_client.ltrim(key, -max_items, -1)
                self.redis_client.expire(key, self.ttl)
                logger.info("Saved %s turn to Redis conversation memory [session_id=%s]", role, session_id)
                return
            except Exception as exc:
                logger.debug("Redis add_history error: %s", exc)
        hist = self._local_history.setdefault(session_id, [])
        hist.append(entry)
        if len(hist) > max_items:
            self._local_history[session_id] = hist[-max_items:]
        logger.info("Saved %s turn to in-memory conversation storage [session_id=%s]", role, session_id)

    def clear_history(self, session_id: str) -> None:
        logger.info("Clearing conversation memory for session_id=%s", session_id)
        if self.redis_client:
            try:
                self.redis_client.delete(f"ba_chat:history:{session_id}")
            except Exception as exc:
                logger.debug("Redis clear_history error: %s", exc)
        self._local_history.pop(session_id, None)

    def get_embedding(self, text: str) -> list[float] | None:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self.redis_client:
            try:
                raw = self.redis_client.get(f"ba_chat:embed:{h}")
                if raw:
                    logger.debug("Redis embedding cache HIT for hash=%s", h[:8])
                    return json.loads(raw)
            except Exception as exc:
                logger.debug("Redis get_embedding error: %s", exc)
        cached = self._local_embeddings.get(h)
        if cached:
            logger.debug("In-memory embedding cache HIT for hash=%s", h[:8])
        return cached

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self.redis_client:
            try:
                self.redis_client.set(f"ba_chat:embed:{h}", json.dumps(embedding), ex=self.ttl)
                logger.info("Cached embedding in Redis for hash=%s (ttl=%ds)", h[:8], self.ttl)
                return
            except Exception as exc:
                logger.debug("Redis set_embedding error: %s", exc)
        self._local_embeddings[h] = embedding
        logger.info("Cached embedding in memory for hash=%s", h[:8])

    def get_qa_cache(self, query: str) -> dict[str, Any] | None:
        h = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if self.redis_client:
            try:
                raw = self.redis_client.get(f"ba_chat:qa:{h}")
                if raw:
                    logger.info("Redis QA cache HIT for query hash=%s", h[:8])
                    return json.loads(raw)
            except Exception as exc:
                logger.debug("Redis get_qa_cache error: %s", exc)
        cached = self._local_qa.get(h)
        if cached:
            logger.info("In-memory QA cache HIT for query hash=%s", h[:8])
        return cached

    def set_qa_cache(self, query: str, data: dict[str, Any], ttl: int = 3600) -> None:
        h = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if self.redis_client:
            try:
                self.redis_client.set(f"ba_chat:qa:{h}", json.dumps(data, default=str), ex=ttl)
                logger.info("Cached QA response in Redis for query hash=%s (ttl=%ds)", h[:8], ttl)
                return
            except Exception as exc:
                logger.debug("Redis set_qa_cache error: %s", exc)
        self._local_qa[h] = data
        logger.info("Cached QA response in memory for query hash=%s", h[:8])

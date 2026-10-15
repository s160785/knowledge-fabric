"""Serialize the canonical graph into the repository's Serenity response envelope."""

from typing import Any

from conseek_knowledge_fabric.models import KnowledgeGraph


def serialize(graph: KnowledgeGraph, mode: str = "full") -> dict[str, Any]:
	if mode not in {"full", "coding-agent"}:
		raise ValueError(f"Unsupported response mode: {mode}")
	return {"mode": mode, "knowledge": graph.to_dict()}
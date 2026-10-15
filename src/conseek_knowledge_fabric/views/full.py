"""Build comprehensive knowledge responses for batch consumers."""

from conseek_knowledge_fabric.models import KnowledgeGraph
from conseek_knowledge_fabric.serenity import serialize


def build(graph: KnowledgeGraph) -> dict[str, object]:
	return serialize(graph, "full")
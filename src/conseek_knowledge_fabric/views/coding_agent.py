"""Build focused knowledge responses for coding agents."""

from dataclasses import replace

from conseek_knowledge_fabric.models import KnowledgeGraph
from conseek_knowledge_fabric.serenity import serialize


def build(graph: KnowledgeGraph, focus: set[str] | None = None) -> dict[str, object]:
	if not focus:
		return serialize(graph, "coding-agent")

	selected = {symbol.id for symbol in graph.symbols if symbol.name in focus or symbol.id in focus}
	for relationship in graph.relationships:
		if relationship.source in selected:
			selected.update(symbol.id for symbol in graph.symbols if symbol.id == relationship.target)
		if relationship.target in selected:
			selected.add(relationship.source)

	focused_graph = replace(
		graph,
		symbols=[symbol for symbol in graph.symbols if symbol.id in selected],
		relationships=[
			relationship for relationship in graph.relationships
			if relationship.source in selected or relationship.target in selected
		],
	)
	return serialize(focused_graph, "coding-agent")
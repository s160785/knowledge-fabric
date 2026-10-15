"""FastAPI routes for building and browsing code knowledge graphs."""

from pathlib import Path
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from conseek_knowledge_fabric.app.schemas import BuildRequest, BAChatRequest, BAChatResponse
from conseek_knowledge_fabric.ba_agent import RepoBAAgent
from conseek_knowledge_fabric.utils.browser_views import VIEW_QUERIES, browser_view_url
from conseek_knowledge_fabric.utils.graph_pipeline import build_repository_graph
from conseek_knowledge_fabric.utils.logging import get_logger, log_call


STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
app = FastAPI(title="Code Knowledge Graph", docs_url="/api/docs", redoc_url=None)
logger = get_logger(__name__)


@app.get("/", include_in_schema=False)
@log_call(logger)
def home() -> FileResponse:
	return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/build")
@log_call(logger)
def build_graph(request: BuildRequest) -> dict[str, object]:
	try:
		logger.info("Starting repository graph build request")
		result = build_repository_graph(
			request.repository,
			request.neo4j_uri,
			request.neo4j_username,
			request.neo4j_password.get_secret_value(),
			request.neo4j_database,
			request.branch,
		)
		return {
			**result,
			"browser_urls": {
				view: browser_view_url(request.neo4j_browser_url, result["repository"], view, request.neo4j_database)
				for view in VIEW_QUERIES
			},
		}
	except (OSError, RuntimeError, ValueError, TimeoutError) as error:
		logger.error("Graph build request failed exception_type=%s", type(error).__name__)
		raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/ba-chat", response_model=BAChatResponse)
@log_call(logger)
def ba_chat(request: BAChatRequest) -> BAChatResponse:
	try:
		logger.info("BA agent chat request received")
		agent = RepoBAAgent(
			uri=request.neo4j_uri,
			username=request.neo4j_username,
			password=request.neo4j_password.get_secret_value(),
			database=request.neo4j_database,
		)
		try:
			response = agent.ask(request.question)
			return BAChatResponse(
				question=response["question"],
				answer=response["answer"],
				evidence=response["evidence"],
			)
		finally:
			agent.close()
	except (OSError, RuntimeError, ValueError, TimeoutError) as error:
		logger.error("BA agent chat request failed exception_type=%s", type(error).__name__)
		raise HTTPException(status_code=400, detail=str(error)) from error


def main() -> None:
	uvicorn.run(
		"conseek_knowledge_fabric.app.main:app",
		host=os.getenv("CONSEEK_HOST", "127.0.0.1"),
		port=int(os.getenv("CONSEEK_PORT", "8000")),
		reload=False,
	)


if __name__ == "__main__":
	main()
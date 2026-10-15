"""Run the local Code Knowledge Graph web application."""

import uvicorn

from conseek_knowledge_fabric.app.main import main


if __name__ == "__main__":
	uvicorn.run("conseek_knowledge_fabric.app.main:app", host="127.0.0.1", port=8000, reload=True)

# running fastapi server with uvicorn
# uvicorn conseek_knowledge_fabric.app.main:app --host
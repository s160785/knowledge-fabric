import asyncio
import os
import time

# Set environment variables before importing litellm
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from litellm import ModelResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Suppress verbose litellm logs
litellm.suppress_debug_info = True

# Point to your Graphify configuration
server_params = StdioServerParameters(
    command="python3",
    args=["-m", "graphify.serve", "graphify-out/graph.json"]
)

async def ask_local_chatbot(
    client_query: str,
    model: str = os.getenv("LLM_MODEL", "ollama/gemma3:4b"),
    **kwargs
):
    # 1. Fetch raw context from Graphify MCP server
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            response = await session.call_tool(
                name="query_graph", 
                arguments={
                    "query": client_query,
                    "question": client_query,  # Solves internal logger crash
                    "budget": 3500
                }
            )
            if response.content:
                first_item = response.content[0]
                graph_context = getattr(first_item, "text", str(first_item))
            else:
                graph_context = "No graph context found."

    # Print Graph Context
    print("\n=== Graph Context ===")
    print(graph_context)

    # 2. Structure the prompt to turn technical code-nodes into descriptive client summaries
    system_instruction = (
        "You are a friendly, descriptive customer-facing AI chatbot. The user is a client "
        "asking a high-level question about a repository codebase. Below is raw technical "
        "graph context (modules, files, function names) extracted by Graphify. "
        "Translate this data into a clear, readable, non-technical or semi-technical explanation. "
        "Do not mention code terms like 'BFS traversal' or 'token budget'—focus entirely on what the system does."
    )

    # 3. Call LLM via LiteLLM (supports Ollama, OpenAI, Anthropic, Bedrock, etc.)
    start_time = time.perf_counter()
    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Client Question: {client_query}\n\nGraphify Context:\n{graph_context}"}
        ],
        **kwargs
    )
    response_time = time.perf_counter() - start_time

    # 4. Extract token usage and cost metrics
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = 0.0

    print("\n=== Metrics ===")
    print(f"Model: {model}")
    print(f"Response Time: {response_time:.2f}s")
    print(f"Tokens: Prompt={prompt_tokens}, Completion={completion_tokens}, Total={total_tokens}")
    print(f"Cost: ${cost:.6f}")

    answer = ""
    if isinstance(response, ModelResponse) and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message"):
            answer = choice.message.content or ""

    return answer

# Run the system
if __name__ == "__main__":
    answer = asyncio.run(ask_local_chatbot("what is the scope of the project"))
    print("\n=== Chatbot Descriptive Answer ===")
    print(answer)

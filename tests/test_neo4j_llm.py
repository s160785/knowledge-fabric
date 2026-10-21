import os
from neo4j import GraphDatabase
import ollama
from google import genai
from google.genai import types

# -------------------------------------------------------------
# 1. DATABASE CONFIGURATION
# -------------------------------------------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Sts@9550"

# Make sure you created a Vector Index in Neo4j over your embeddings!
# Example Cypher to run once in Neo4j Browser:
# CREATE VECTOR INDEX node_embeddings_idx IF NOT EXISTS
# FOR (n:Symbol) ON (n.embedding)
# OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
VECTOR_INDEX_NAME = "symbol_embeddings"


def get_graph_context_via_vector(client_query: str, top_k: int = 5):
    """
    1. Generates an embedding for the client's query.
    2. Performs a vector similarity search inside Neo4j.
    3. Traverses the graph to return the target node summaries and their connections.
    """
    # Generate query embedding matching your vector model size (e.g., using Ollama)
    emb_res = ollama.embeddings(model="nomic-embed-text", prompt=client_query)
    query_vector = emb_res["embedding"]
    
    # Advanced GraphRAG Cypher Query: Finds similar nodes + uncovers what they talk to
    cypher_query = """
        CALL db.index.vector.queryNodes($index, $k, $queryEmbedding)
        YIELD node AS targetNode, score
        MATCH (targetNode)-[r]->(connectedNode)
        RETURN 
            targetNode.name AS element, 
            targetNode.summary AS description, 
            type(r) AS relationship, 
            connectedNode.name AS dependency,
            score
        ORDER BY score DESC
        """
    
    context_strings = []
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        result = session.run(
            cypher_query, 
            index=VECTOR_INDEX_NAME, 
            k=top_k, 
            queryEmbedding=query_vector
        )
        for record in result:
            fmt = (
                f"- Code Element: '{record['element']}' (Relevance: {record['score']:.2f})\n"
                f"  Summary: {record['description']}\n"
                f"  Relationship: Establishes a [{record['relationship']}] link to '{record['dependency']}'\n"
            )
            context_strings.append(fmt)
            
    driver.close()
    return "\n".join(context_strings) if context_strings else "No matching structural records found in database."

# -------------------------------------------------------------
# 2. CHATBOT LLM ROUTING
# -------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are a consultative customer-facing AI chatbot. The user is a client asking high-level "
    "questions about a software platform's architecture. Below is a structural context slice extracted "
    "directly from a Neo4j Knowledge Graph using vector similarity and relationship mappings. "
    "Synthesize this factual graph data into a highly readable, descriptive, and non-technical summary. "
    "Do not quote graph database syntax or raw vector metrics—focus purely on explaining the functional scope."
)

def run_ollama_chatbot(client_query: str):
    """Answers using 100% Local LLM Processing"""
    print("🤖 Retrieving records from Neo4j Vector Index...")
    graph_context = get_graph_context_via_vector(client_query)
    print("Graph Context Retrieved:\n", graph_context)
    print("🧠 Generating descriptive response via Local Ollama...")
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Client Question: {client_query}\n\nGraph Context:\n{graph_context}"}
        ]
    )
    return response['message']['content']

def run_gemini_chatbot(client_query: str):
    """Answers using Cloud-Hosted Google Gemini 2.5"""
    print("🤖 Retrieving records from Neo4j Vector Index...")
    graph_context = get_graph_context_via_vector(client_query)
    print("Graph Context Retrieved:\n", graph_context)
    print("✨ Generating descriptive response via Gemini Cloud...")
    client = genai.Client() # Assumes GEMINI_API_KEY environment variable is set
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Client Question: {client_query}\n\nGraph Context:\n{graph_context}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3
        )
    )
    return response.text

# -------------------------------------------------------------
# 3. EXECUTION RUNTIME
# -------------------------------------------------------------
if __name__ == "__main__":
    query = "what is the scope of the project"
    
    # Execute Local Option
    local_answer = run_ollama_chatbot(query)
    print("\n=== Ollama GraphRAG Answer ===\n", local_answer)
    
    # Execute Cloud Option (Uncomment if you have your Gemini API Key set)
    # cloud_answer = run_gemini_chatbot(query)
    # print("\n=== Gemini GraphRAG Answer ===\n", cloud_answer)

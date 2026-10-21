import json
from neo4j import GraphDatabase

# 1. Update your Neo4j connection credentials here
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "Sts@9550")  # Replace with your actual password

# 2. Define the Cypher query
QUERY = "MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as props"

def export_to_graphify_json():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            print("Fetching data from Neo4j...")
            result = session.run(QUERY)
            
            graphify_nodes = []
            
            for record in result:
                # Fallback to name property, or use the first label as the name anchor
                labels = list(record["labels"])
                props = record["props"]
                display_name = props.get("name", props.get("title", f"{labels[0] or 'Node'}_{record['id']}"))
                
                # Format to match Graphify specification requirements
                graphify_nodes.append({
                    "id": str(record["id"]),
                    "name": display_name,
                    "type": labels[0] if labels else "Unknown",
                    "properties": props
                })
            
            # Construct the exact Graphify root schema envelope
            graphify_schema = {
                "graphify_version": "8.0",
                "nodes": graphify_nodes,
                "edges": []  # Empty since the query only requests nodes (n)
            }
                
            # 3. Write directly to a clean graph.json file
            output_filename = "neo4j_graph.json"
            with open(output_filename, "w", encoding="utf-8") as json_file:
                json.dump(graphify_schema, json_file, indent=2, ensure_ascii=False)
                
            print(f"Success! Converted {len(graphify_nodes)} nodes into Graphify schema format inside '{output_filename}'.")

if __name__ == "__main__":
    export_to_graphify_json()

import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.node_parser import MarkdownNodeParser
from config import setup_global_settings
import traceback
import logging
import sys

# Enable LlamaIndex debug logging so you can see every prompt sent to Ollama and the responses!
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def build_knowledge_graph():
    """Reads Markdown documents and builds a Property Graph in Neo4j."""
    print("Setting up global LLM settings...")
    llm = setup_global_settings()

    input_dir = "markdown_output"
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"Error: '{input_dir}' is empty or does not exist. Please run the conversion script first.")
        return

    print("Loading documents...")
    def extract_metadata_from_path(filepath):
        # Assumes structure: markdown_output/<State>/<Year>/<filename>/...
        parts = Path(filepath).parts
        # We need to find 'markdown_output' in the path to safely extract state and year
        try:
            base_idx = parts.index("markdown_output")
            # If structure is correctly followed: parts[base_idx+1] is State, parts[base_idx+2] is Year
            state = parts[base_idx + 1] if len(parts) > base_idx + 1 else "Unknown"
            year = parts[base_idx + 2] if len(parts) > base_idx + 2 else "Unknown"
        except ValueError:
            state, year = "Unknown", "Unknown"
        
        return {"state": state, "year": year}

    # Load all markdown files recursively and inject metadata
    documents = SimpleDirectoryReader(
        input_dir, 
        recursive=True, 
        required_exts=[".md"],
        file_metadata=extract_metadata_from_path
    ).load_data()
    print(f"Loaded {len(documents)} documents.")

    print("Connecting to Neo4j...")
    try:
        graph_store = Neo4jPropertyGraphStore(
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            database="neo4j",
        )
    except Exception as e:
        print("Failed to connect to Neo4j. Is the Neo4j database running?")
        traceback.print_exc()
        return

    print("Configuring Property Graph extraction...")
    # We use SimpleLLMPathExtractor to extract triplets (Node -> Relation -> Node)
    # The LLM will automatically deduce relations.
    path_extractor = SimpleLLMPathExtractor(
        llm=llm,
        max_paths_per_chunk=10,
        num_workers=1 # Local models usually process one chunk at a time best
    )

    print("Parsing documents into smart Markdown chunks...")
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Parsed into {len(nodes)} smart chunks.")

    print("Building Property Graph Index. This may take a while depending on your LLM speed...")
    try:
        index = PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=[path_extractor],
            property_graph_store=graph_store,
            show_progress=True,
        )
        print("Successfully built the Knowledge Graph!")
    except Exception as e:
        print("An error occurred while building the graph.")
        traceback.print_exc()

if __name__ == "__main__":
    build_knowledge_graph()

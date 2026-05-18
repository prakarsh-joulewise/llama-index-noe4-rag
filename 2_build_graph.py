import os
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.node_parser import MarkdownNodeParser
from config import setup_global_settings
import traceback

def build_knowledge_graph():
    """Reads Markdown documents and builds a Property Graph in Neo4j."""
    print("Setting up global LLM settings...")
    llm = setup_global_settings()

    input_dir = "markdown_output"
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"Error: '{input_dir}' is empty or does not exist. Please run the conversion script first.")
        return

    print("Loading documents...")
    # Load all markdown files recursively
    documents = SimpleDirectoryReader(input_dir, recursive=True, required_exts=[".md"]).load_data()
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

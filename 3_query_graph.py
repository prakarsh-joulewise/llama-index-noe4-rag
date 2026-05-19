import os
from llama_index.core import PropertyGraphIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.postprocessor import LLMRerank
from config import setup_global_settings
import traceback

def query_knowledge_graph():
    """Connects to the Neo4j Property Graph and allows querying."""
    print("Setting up global LLM settings...")
    llm = setup_global_settings()

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

    print("Loading Property Graph Index from Neo4j...")
    # Load the index from the existing graph store
    try:
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
        )
    except Exception as e:
        print("Failed to load Property Graph. Have you run the build script yet?")
        traceback.print_exc()
        return

    print("Configuring LLM Reranker for exact keyword matching...")
    # LLMRerank acts as a filter: it looks at the 20 tables retrieved by vector search, 
    # reads them, and explicitly filters out the ones that don't say "PuVVNL" before 
    # generating the final answer.
    reranker = LLMRerank(
        choice_batch_size=5,
        top_n=3,
        llm=llm
    )

    print("\n--- Optional Metadata Filtering ---")
    state_filter = input("Enter State to filter by (or press Enter to skip): ").strip()
    year_filter = input("Enter Year to filter by (or press Enter to skip): ").strip()
    
    filters_list = []
    if state_filter:
        filters_list.append(ExactMatchFilter(key="state", value=state_filter))
    if year_filter:
        filters_list.append(ExactMatchFilter(key="year", value=year_filter))
        
    metadata_filters = MetadataFilters(filters=filters_list) if filters_list else None

    # Create a query engine
    # Setting `llm` parameter allows the engine to synthesize the final answer based on retrieved nodes
    query_engine = index.as_query_engine(
        llm=llm,
        include_text=True, # Set to True to retrieve the underlying document text chunks as well
        similarity_top_k=20, # Fetch 20 tables
        node_postprocessors=[reranker], # Use the LLM to filter the 20 tables down to the correct 3
        filters=metadata_filters # Strictly filter out nodes from other states/years
    )

    print("\n--- Knowledge Graph Query Engine ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_query = input("Ask a question about the legal documents: ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        if not user_query.strip():
            continue

        print("Querying graph...")
        try:
            response = query_engine.query(user_query)
            print("\nResponse:")
            print("-" * 40)
            print(response)
            print("-" * 40)
        except Exception as e:
            print("Error querying the graph:")
            traceback.print_exc()

if __name__ == "__main__":
    query_knowledge_graph()

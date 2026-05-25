import os
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

def get_llm():
    """Initializes and returns the OpenAI-compatible LLM."""
    llm = OpenAILike(
        model=os.getenv("LLM_MODEL_NAME", "gemma4:e4b"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
        api_base=os.getenv("LLM_API_BASE", "http://122.186.70.126:11434/v1"),
        context_window=32768,
        is_chat_model=True
    )
    return llm

def setup_global_settings():
    """Configures global settings for LlamaIndex."""
    llm = get_llm()
    Settings.llm = llm
    # We can also configure a local embedding model if needed
    Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
    
    # We'll stick to basic settings for now
    return llm

def get_neo4j_driver():
    """Returns a Neo4j driver connection."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    return GraphDatabase.driver(uri, auth=(user, password))

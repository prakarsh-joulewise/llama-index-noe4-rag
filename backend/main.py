import os
import sys
import re
import json
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import setup_global_settings, get_neo4j_driver
from llama_index.core import Settings, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

app = FastAPI(title="Joulewise RAG Chat Agent API")

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and database configuration
print("Setting up global LLM and embedding settings...")
llm = setup_global_settings()

from llama_index.core.embeddings.utils import resolve_embed_model
embed_model = resolve_embed_model(Settings.embed_model)
print(f"Embedding model: {embed_model.model_name}")

print("Connecting to Neo4j...")
driver = get_neo4j_driver()

# Verify database connection
try:
    with driver.session() as session:
        result = session.run("MATCH (c:Chunk) RETURN count(c) as cnt").single()
        print(f"  Connected to Neo4j. Found {result['cnt']} chunks.")
except Exception as e:
    print(f"  [Error] Failed to connect to Neo4j: {e}")

# Import retriever and reranker from parent module
# This ensures that any retrieval optimizations are automatically shared
try:
    import importlib
    query_graph_module = importlib.import_module("3_query_graph")
    HybridNeo4jRetriever = query_graph_module.HybridNeo4jRetriever
    SmartTableBoostReranker = query_graph_module.SmartTableBoostReranker
    qa_prompt_tmpl = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n\n"
        "Important instructions for accuracy:\n"
        "1. Be extremely precise with numbers. Extract both the 'Proposed' (Petitioner's Submission) and the 'Approved' (Commission's Analysis) values if the query asks for both.\n"
        "2. If you find a table with 'STATUS: **PROPOSED (Petitioner's Submission)**' containing a rate, and another with 'STATUS: **APPROVED (Commission's Analysis)**' containing a rate, you MUST state both rates clearly with their respective numerical values.\n"
        "3. Cite the source document name, state, and fiscal year/page context for any statistics, costs, or charges you mention.\n"
        "4. If there are conflicting or multiple values (e.g., proposed vs. approved) or different values for different categories, list them both clearly and explain the difference based on the context.\n"
        "5. If the context contains both average rates (e.g. Average Wheeling Charge) and voltage-wise or category-specific rates, you MUST state both the average rate and the voltage-wise/category-specific rates clearly.\n\n"
        "Query: {query_str}\n"
        "Answer: "
    )
except Exception as e:
    print(f"  [Error] Failed to import RAG pipeline components: {e}")
    traceback.print_exc()

# Initialize static components
reranker_type = os.getenv("RERANKER_TYPE", "smart-table")
top_n = int(os.getenv("RERANKER_TOP_N", "8"))

retriever = HybridNeo4jRetriever(
    driver=driver,
    embed_model=embed_model,
    vector_top_k=50 if reranker_type == "smart-table" else 20,
    fulltext_top_k=50 if reranker_type == "smart-table" else 20,
    rrf_k=60
)

if reranker_type == "smart-table":
    reranker = SmartTableBoostReranker(top_n=top_n)
else:
    reranker = query_graph_module.DebugReranker(
        reranker_type=reranker_type,
        top_n=top_n,
        llm=llm
    )

# Build query engine with streaming enabled
qa_tmpl = PromptTemplate(qa_prompt_tmpl)
response_synthesizer = get_response_synthesizer(
    llm=llm,
    response_mode="compact",
    text_qa_template=qa_tmpl,
    streaming=True
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[reranker]
)


class StatusResponse(BaseModel):
    neo4j_connected: bool
    neo4j_chunks: int
    llm_model: str
    reranker_type: str
    top_n: int


@app.get("/api/status", response_model=StatusResponse)
def get_status():
    """Returns the connection status and configurations of the RAG pipeline."""
    neo4j_ok = False
    chunks_count = 0
    try:
        with driver.session() as session:
            res = session.run("MATCH (c:Chunk) RETURN count(c) as cnt").single()
            chunks_count = res["cnt"]
            neo4j_ok = True
    except Exception:
        pass
        
    return StatusResponse(
        neo4j_connected=neo4j_ok,
        neo4j_chunks=chunks_count,
        llm_model=os.getenv("LLM_MODEL_NAME", "Unknown"),
        reranker_type=reranker_type,
        top_n=top_n
    )


def rewrite_query(message: str, history: list) -> str:
    """Uses the LLM to rewrite a follow-up query into a standalone, context-rich query."""
    if not history:
        return message
        
    # Format history for prompt context
    history_str = ""
    for turn in history[-5:]:  # Limit context to last 5 turns
        role = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"]
        # Strip long markdown tables from history context to keep prompt clean
        content_clean = re.sub(r'\|.*\|', '[Table omitted]', content)
        history_str += f"{role}: {content_clean}\n"
        
    rewrite_prompt = (
        "You are a helpful assistant that reformulates user follow-up questions.\n"
        "Given the conversation history and a new follow-up question, rewrite it into a single, standalone search query that contains all necessary context (like states, specific terms, or years) to search a database.\n\n"
        "Rules:\n"
        "1. Do NOT include specific section numbers, section paths (e.g. 'Section 5.2'), document names, or page numbers in the rewritten query, as this will over-restrict search results. Keep it focused on the core topic (e.g. 'transmission charges', 'wheeling charges'), state, and year.\n"
        "2. Do NOT answer the question. Only output the rewritten question.\n\n"
        f"Conversation History:\n{history_str}\n"
        f"Follow-up Question: {message}\n"
        "Standalone Search Query: "
    )
    
    try:
        print(f"  [Memory] Rewriting query based on history...")
        response = llm.complete(rewrite_prompt)
        rewritten = str(response).strip()
        # Clean up any potential quotes around the rewritten query
        rewritten = rewritten.strip('"\'')
        print(f"  [Memory] Original: \"{message}\" -> Rewritten: \"{rewritten}\"")
        return rewritten
    except Exception as e:
        print(f"  [Warning] Query rewrite failed: {e}. Using original message.")
        return message


@app.websocket("/api/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """WebSocket endpoint that receives user queries, processes them, 

    sends retrieved source metadata, and streams synthesized answer tokens.
    """
    await websocket.accept()
    print("  [WebSocket] Client connected.")
    
    try:
        while True:
            # Receive text message
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            
            message = data.get("message", "").strip()
            history = data.get("history", [])
            
            if not message:
                continue
                
            # 1. Rewrite query based on context history
            search_query = rewrite_query(message, history)
            
            # 2. Expose and enhance state filters (e.g. UP -> Uttar Pradesh)
            enhanced_query = search_query
            for code, full_name in [("AP", "Andhra Pradesh"), ("UP", "Uttar Pradesh"),
                                     ("MH", "Maharashtra"), ("KA", "Karnataka"),
                                     ("TN", "Tamil Nadu"), ("TS", "Telangana")]:
                # match whole words/tokens
                if code in search_query.upper().split():
                    enhanced_query = search_query + f" ({full_name})"
                    break
            
            await websocket.send_json({
                "type": "status",
                "content": f"Searching database for: \"{enhanced_query}\"..."
            })
            
            # 3. Execute query on LlamaIndex engine
            try:
                # The response object has the sources pre-retrieved even when streaming
                response = query_engine.query(enhanced_query)
                
                # Send source nodes first
                sources = []
                for score_node in response.source_nodes:
                    node = score_node.node
                    sources.append({
                        "id": node.node_id,
                        "score": float(score_node.score) if score_node.score is not None else 0.0,
                        "document_name": node.metadata.get("document_name", "Unknown"),
                        "state": node.metadata.get("state", "Unknown"),
                        "year": node.metadata.get("year", "Unknown"),
                        "header_path": re.sub(r'<[^>]+>', '', node.metadata.get("header_path", "Unknown")),
                        "chunk_type": node.metadata.get("chunk_type", "Unknown")
                    })
                
                await websocket.send_json({
                    "type": "sources",
                    "content": sources
                })
                
                # Stream the text response tokens
                print(f"  [WebSocket] Streaming response tokens for query: \"{enhanced_query}\"")
                for token in response.response_gen:
                    await websocket.send_json({
                        "type": "token",
                        "content": token
                    })
                    
                # Send done signal
                await websocket.send_json({"type": "done"})
                print("  [WebSocket] Finished streaming response.")
                
            except Exception as query_err:
                err_msg = f"Error during query execution: {query_err}"
                print(f"  [Error] {err_msg}")
                traceback.print_exc()
                await websocket.send_json({
                    "type": "error",
                    "content": err_msg
                })
                
    except WebSocketDisconnect:
        print("  [WebSocket] Client disconnected.")
    except Exception as e:
        print(f"  [WebSocket Exception] {e}")
        traceback.print_exc()


# Mount static files folder to serve the UI
# This allows access to http://localhost:8000/ui/index.html
ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.exists(ui_dir):
    print(f"Mounting static UI folder from: {ui_dir}")
    app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")
else:
    print(f"  [Warning] UI directory not found at: {ui_dir}. Skipping mount.")


if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 so it is accessible from the network (Ubuntu server exposure)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

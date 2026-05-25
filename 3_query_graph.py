import os
# Force huggingface offline mode to prevent checking for updates online / hanging
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import traceback

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from llama_index.core import Settings, PromptTemplate
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from config import setup_global_settings, get_neo4j_driver
from utils import resolve_state_name


import re

def detect_state_from_query(query_text):
    """Detects target state from query text, mapping both codes and full names."""
    normalized = query_text.upper()
    
    # Check for direct mentions of state codes or full names
    state_keywords = {
        "UTTAR PRADESH": "Uttar Pradesh",
        "UP": "Uttar Pradesh",
        "ANDHRA PRADESH": "Andhra Pradesh",
        "AP": "Andhra Pradesh",
        "MAHARASHTRA": "Maharashtra",
        "MH": "Maharashtra",
        "DELHI": "Delhi",
        "DL": "Delhi",
        "KARNATAKA": "Karnataka",
        "KA": "Karnataka",
        "TAMIL NADU": "Tamil Nadu",
        "TN": "Tamil Nadu",
        "TELANGANA": "Telangana",
        "TS": "Telangana",
        "GUJARAT": "Gujarat",
        "GJ": "Gujarat",
        "RAJASTHAN": "Rajasthan",
        "RJ": "Rajasthan",
        "MADHYA PRADESH": "Madhya Pradesh",
        "MP": "Madhya Pradesh",
        "WEST BENGAL": "West Bengal",
        "WB": "West Bengal",
        "HARYANA": "Haryana",
        "HR": "Haryana",
        "PUNJAB": "Punjab",
        "PB": "Punjab",
        "KERALA": "Kerala",
        "KL": "Kerala",
        "BIHAR": "Bihar",
        "BR": "Bihar",
        "JHARKHAND": "Jharkhand",
        "JH": "Jharkhand",
        "ODISHA": "Odisha",
        "OD": "Odisha",
        "ASSAM": "Assam",
        "AS": "Assam",
        "HIMACHAL PRADESH": "Himachal Pradesh",
        "HP": "Himachal Pradesh",
        "UTTARAKHAND": "Uttarakhand",
        "UK": "Uttarakhand",
        "UA": "Uttarakhand"
    }
    
    # Match whole words/tokens to avoid false matches (e.g. "UP" in "support")
    words = set(re.findall(r'\b[A-Z]+\b', normalized))
    
    # Check full names first
    for name_key, resolved in state_keywords.items():
        if " " in name_key:
            if name_key in normalized:
                return resolved
                
    # Check single-word / code keys
    for word in words:
        if word in state_keywords:
            return state_keywords[word]
            
    return None


class HybridNeo4jRetriever(BaseRetriever):
    """Custom retriever that combines Neo4j vector search and full-text search
    using Reciprocal Rank Fusion (RRF) for optimal retrieval.
    """
    
    def __init__(self, driver, embed_model, vector_top_k=15, fulltext_top_k=15, rrf_k=60):
        super().__init__()
        self._driver = driver
        self._embed_model = embed_model
        self._vector_top_k = vector_top_k
        self._fulltext_top_k = fulltext_top_k
        self._rrf_k = rrf_k
    
    def _retrieve(self, query_bundle: QueryBundle):
        query_text = query_bundle.query_str
        
        # Detect state filter from query text
        state = detect_state_from_query(query_text)
        if state:
            print(f"  [Retriever] Filtering results for State: '{state}'")
        
        # 1. Vector search
        query_embedding = self._embed_model.get_text_embedding(query_text)
        vector_results = self._vector_search(query_embedding, state=state)
        
        # 2. Full-text search
        fulltext_results = self._fulltext_search(query_text, state=state)
        
        # 3. Reciprocal Rank Fusion
        fused = self._rrf_fuse(vector_results, fulltext_results)
        
        return fused
    
    def _vector_search(self, embedding, state=None):
        """Performs cosine similarity vector search on Chunk embeddings, with optional state filter."""
        results = []
        # If filtering by state, retrieve more candidate nodes from the index before filtering
        top_k = 150 if state else self._vector_top_k
        with self._driver.session() as session:
            records = session.run("""
                CALL db.index.vector.queryNodes('chunk_vector_index', $top_k, $embedding)
                YIELD node, score
                WHERE $state IS NULL OR node.state = $state
                RETURN node.id AS id, node.text AS text, node.state AS state,
                       node.year AS year, node.document_name AS doc_name,
                       node.header_path AS header_path, node.chunk_type AS chunk_type,
                       score
                ORDER BY score DESC
                LIMIT $limit
            """, {"top_k": top_k, "embedding": embedding, "state": state, "limit": self._vector_top_k})
            
            for rank, record in enumerate(records):
                results.append({
                    "id": record["id"],
                    "text": record["text"],
                    "state": record["state"],
                    "year": record["year"],
                    "doc_name": record["doc_name"],
                    "header_path": record["header_path"],
                    "chunk_type": record["chunk_type"],
                    "score": record["score"],
                    "rank": rank + 1,
                    "source": "vector"
                })
        return results
    
    def _fulltext_search(self, query_text, state=None):
        """Performs native Neo4j full-text (BM25) search on Chunk.text, with optional state filter."""
        # Escape special Lucene characters in the query
        escaped = self._escape_lucene(query_text)
        results = []
        with self._driver.session() as session:
            try:
                records = session.run("""
                    CALL db.index.fulltext.queryNodes('chunk_text_index', $query)
                    YIELD node, score
                    WHERE $state IS NULL OR node.state = $state
                    RETURN node.id AS id, node.text AS text, node.state AS state,
                           node.year AS year, node.document_name AS doc_name,
                           node.header_path AS header_path, node.chunk_type AS chunk_type,
                           score
                    ORDER BY score DESC
                    LIMIT $limit
                """, {"query": escaped, "state": state, "limit": self._fulltext_top_k})
                
                for rank, record in enumerate(records):
                    results.append({
                        "id": record["id"],
                        "text": record["text"],
                        "state": record["state"],
                        "year": record["year"],
                        "doc_name": record["doc_name"],
                        "header_path": record["header_path"],
                        "chunk_type": record["chunk_type"],
                        "score": record["score"],
                        "rank": rank + 1,
                        "source": "fulltext"
                    })
            except Exception as e:
                print(f"  [Warning] Full-text search error: {e}")
        return results
    
    def _escape_lucene(self, text):
        """Escapes special Lucene query characters."""
        special_chars = r'+-&|!(){}[]^"~*?:\/'
        escaped = ""
        for ch in text:
            if ch in special_chars:
                escaped += "\\" + ch
            else:
                escaped += ch
        return escaped
    
    def _rrf_fuse(self, vector_results, fulltext_results):
        """Reciprocal Rank Fusion to combine vector and full-text search results."""
        scores = {}  # id -> rrf_score
        data = {}    # id -> result dict
        
        for r in vector_results:
            rid = r["id"]
            scores[rid] = scores.get(rid, 0) + 1.0 / (self._rrf_k + r["rank"])
            data[rid] = r
        
        for r in fulltext_results:
            rid = r["id"]
            scores[rid] = scores.get(rid, 0) + 1.0 / (self._rrf_k + r["rank"])
            if rid not in data:
                data[rid] = r
        
        # Sort by fused score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Convert to NodeWithScore objects
        nodes_with_scores = []
        for rid in sorted_ids:
            r = data[rid]
            node = TextNode(
                text=r["text"],
                id_=r["id"],
                metadata={
                    "state": r["state"],
                    "year": r["year"],
                    "document_name": r["doc_name"],
                    "header_path": r["header_path"],
                    "chunk_type": r["chunk_type"]
                }
            )
            nws = NodeWithScore(node=node, score=scores[rid])
            nodes_with_scores.append(nws)
        
        return nodes_with_scores


class SmartTableBoostReranker(BaseNodePostprocessor):
    """Custom postprocessor that reranks chunks using a cross-encoder, 
    boosts table chunks matching query topics (with specific/generic split), and deduplicates identical bodies.
    """
    def __init__(self, top_n=5):
        super().__init__()
        self._internal = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=100  # get all scores first
        )
        self._top_n = top_n
        
    def _clean_node_text(self, node):
        text = node.get_content()
        
        # Extract prefix line and body
        parts = text.split("\n", 1)
        if len(parts) < 2:
            return
            
        prefix = parts[0]
        body = parts[1]
        
        # Parse metadata from prefix
        state_match = re.search(r'State:\s*([^|]+)', prefix)
        year_match = re.search(r'Year:\s*([^|]+)', prefix)
        doc_match = re.search(r'Document:\s*([^|]+)', prefix)
        context_match = re.search(r'Context:\s*(.+)', prefix)
        
        state = state_match.group(1).strip() if state_match else "Unknown"
        year = year_match.group(1).strip() if year_match else "Unknown"
        doc = doc_match.group(1).strip() if doc_match else "Unknown"
        header_path = context_match.group(1).strip() if context_match else ""
        
        # Clean HTML span tags and duplicates
        header_path_clean = re.sub(r'<[^>]+>', '', header_path)
        header_path_clean = "/".join([p.strip() for p in header_path_clean.split("/") if p.strip()])
        
        # Determine proposed/approved status
        status = "GENERAL CONTEXT"
        if "petitioner" in header_path_clean.lower():
            status = "PROPOSED (Petitioner's Submission)"
        elif "commission" in header_path_clean.lower() or "approved" in header_path_clean.lower():
            status = "APPROVED (Commission's Analysis)"
            
        # Build clean structured text
        clean_text = (
            f"SOURCE DOCUMENT: {doc}\n"
            f"STATE: {state} | YEAR: {year}\n"
            f"SECTION PATH: {header_path_clean}\n"
            f"STATUS: **{status}**\n"
            f"-----------------------------------------\n"
            f"{body.strip()}"
        )
        
        node.set_content(clean_text)
        
    def _postprocess_nodes(self, nodes, query_bundle=None):
        query_text = query_bundle.query_str.lower() if query_bundle else ""
        
        # 1. Get base scores from cross-encoder
        scored_nodes = self._internal.postprocess_nodes(nodes, query_bundle)
        
        # 2. Apply Custom Specific/Generic Table Boosting
        keywords = ["charge", "rate", "tariff", "cost", "fee", "table", "proposed", "approved", "rs", "kwh", "price", "arr", "surcharge", "subsidy"]
        is_numeric_query = any(kw in query_text for kw in keywords)
        
        if is_numeric_query:
            print("\n  [Reranker] Query targets rates/charges. Applying specific/generic table boosting...")
            
            # Divide keywords into specific (safe for body/text matches) and generic (strict header path matches only)
            specific_keywords = ["wheeling", "cross-subsidy", "cross subsidy", "additional surcharge", "green energy", "subsidy", "surcharge"]
            generic_keywords = ["transmission", "open access", "true-up", "true up", "apr"]
            
            matched_specific = [kw for kw in specific_keywords if kw in query_text]
            matched_generic = [kw for kw in generic_keywords if kw in query_text]
            
            for nws in scored_nodes:
                node = nws.node
                chunk_type = node.metadata.get("chunk_type", "")
                is_table = (chunk_type == "table")
                
                if is_table:
                    # Base boost for being a table
                    boost = 0.20
                    node_text_lower = node.get_content().lower()
                    node_header_lower = node.metadata.get("header_path", "").lower()
                    
                    topic_match = False
                    
                    # 1. Check specific keywords in body or header
                    if matched_specific:
                        for topic in matched_specific:
                            if topic in node_text_lower or topic in node_header_lower:
                                topic_match = True
                                break
                                
                    # 2. Check generic keywords in header path only
                    if not topic_match and matched_generic:
                        for topic in matched_generic:
                            if topic in node_header_lower:
                                topic_match = True
                                break
                                
                    # 3. If query contains no specific or generic keywords, default to match all
                    if not matched_specific and not matched_generic:
                        topic_match = True
                        
                    if topic_match:
                        boost += 0.40
                        
                    old_score = nws.score if nws.score is not None else 0.0
                    nws.score = old_score + boost
                    print(f"    Boosting Table Node {node.id_[:8]} | Score: {old_score:.4f} -> {nws.score:.4f} (topic_match={topic_match}) | Header: {node_header_lower[:60]}...")
            
            # Re-sort nodes by the new score
            scored_nodes = sorted(scored_nodes, key=lambda x: x.score if x.score is not None else -9999.0, reverse=True)
            
        # 3. Deduplicate nodes by body content (skipping the metadata prefix line)
        unique_nodes = []
        seen_bodies = set()
        for nws in scored_nodes:
            text = nws.node.get_content()
            parts = text.split("\n", 1)
            body = parts[1].strip() if len(parts) >= 2 else text.strip()
            
            # Normalize and check signature of the body
            normalized_body = " ".join(body.lower().split())
            body_sig = normalized_body[:150]
            
            if body_sig not in seen_bodies:
                seen_bodies.add(body_sig)
                unique_nodes.append(nws)
            else:
                print(f"    Filtering duplicate body node: {nws.node.id_[:8]}")
                
        # 4. Slice to self._top_n
        filtered = unique_nodes[:self._top_n]
        
        # 5. Clean up the node texts for the selected top_n nodes
        print(f"\n  [Reranker] Selected top {len(filtered)} chunks (cleaned & body-deduplicated):")
        for i, n in enumerate(filtered):
            self._clean_node_text(n.node)
            score_str = f"{n.score:.4f}" if n.score is not None else "None"
            chunk_type = n.node.metadata.get("chunk_type")
            print(f"    [{i+1}] Score={score_str} | Type={chunk_type} | {n.node.get_content()[:120].replace(chr(10), ' ')}...")
            
        return filtered


class DebugReranker(BaseNodePostprocessor):
    """Debug wrapper that can rerank using LLM, SentenceTransformers, or None (pass-through)."""
    
    def __init__(self, reranker_type="none", top_n=8, llm=None, choice_batch_size=5):
        super().__init__()
        self._reranker_type = reranker_type.lower()
        self._top_n = top_n
        
        if self._reranker_type == "llm":
            self._internal = LLMRerank(
                choice_batch_size=choice_batch_size,
                top_n=top_n,
                llm=llm
            )
        elif self._reranker_type == "sentence-transformer":
            self._internal = SentenceTransformerRerank(
                model="BAAI/bge-reranker-base",
                top_n=top_n
            )
        else:
            self._internal = None

    def _postprocess_nodes(self, nodes, query_bundle=None):
        print(f"\n  [Retriever] Found {len(nodes)} chunks via hybrid search:")
        for i, n in enumerate(nodes[:10]):  # Show top 10
            src = n.node.metadata.get("chunk_type", "?")
            doc = n.node.metadata.get("document_name", "?")
            print(f"    [{i+1}] RRF={n.score:.4f} | {src} | {doc} | {n.node.get_content()[:100].replace(chr(10), ' ')}...")
        
        if len(nodes) > 10:
            print(f"    ... and {len(nodes) - 10} more")
        
        if self._reranker_type == "llm":
            print("\n  [Reranker] LLM is scoring and filtering chunks. Please wait...")
            filtered = self._internal.postprocess_nodes(nodes, query_bundle)
        elif self._reranker_type == "sentence-transformer":
            print("\n  [Reranker] SentenceTransformer is scoring and filtering chunks. Please wait...")
            filtered = self._internal.postprocess_nodes(nodes, query_bundle)
        else:
            print(f"\n  [Reranker] Reranker is disabled (pass-through). Selecting top {self._top_n} chunks...")
            filtered = nodes[:self._top_n]
            
        print(f"\n  [Reranker] Selected top {len(filtered)} chunks:")
        for i, n in enumerate(filtered):
            score_str = f"{n.score:.4f}" if n.score is not None else "None"
            print(f"    [{i+1}] Score={score_str} | {n.node.get_content()[:120].replace(chr(10), ' ')}...")
        
        return filtered


def query_knowledge_graph():
    """Interactive query engine with hybrid retrieval and LLM reranking."""
    print("Setting up global LLM and embedding settings...")
    llm = setup_global_settings()
    
    # Resolve embedding model
    from llama_index.core.embeddings.utils import resolve_embed_model
    embed_model = resolve_embed_model(Settings.embed_model)
    print(f"Embedding model: {embed_model.model_name}")
    
    print("Connecting to Neo4j...")
    driver = get_neo4j_driver()
    
    # Quick verification
    with driver.session() as session:
        result = session.run("MATCH (c:Chunk) RETURN count(c) as cnt").single()
        print(f"  Found {result['cnt']} chunks in database.")
    
    # Load settings from env
    reranker_type = os.getenv("RERANKER_TYPE", "none")
    top_n = int(os.getenv("RERANKER_TOP_N", "8"))
    
    # Build hybrid retriever (retrieve enough candidate chunks for reranking)
    retriever = HybridNeo4jRetriever(
        driver=driver,
        embed_model=embed_model,
        vector_top_k=50 if reranker_type == "smart-table" else 20,
        fulltext_top_k=50 if reranker_type == "smart-table" else 20,
        rrf_k=60
    )
    
    # Build reranker (LLM, SentenceTransformer, SmartTable, or None)
    if reranker_type == "smart-table":
        reranker = SmartTableBoostReranker(top_n=top_n)
    else:
        reranker = DebugReranker(
            reranker_type=reranker_type,
            top_n=top_n,
            llm=llm
        )
    
    # Custom QA prompt to enforce numeric precision and source citing
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
    qa_tmpl = PromptTemplate(qa_prompt_tmpl)
    
    # Build response synthesizer
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="compact",
        text_qa_template=qa_tmpl
    )
    
    # Build query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker]
    )
    
    print("\n" + "="*60)
    print("  Knowledge Graph Query Engine (Hybrid Retrieval)")
    print("  Type 'exit' or 'quit' to stop.")
    print("="*60 + "\n")
    
    while True:
        user_query = input("Ask a question: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        
        if not user_query.strip():
            continue
        
        # Resolve state codes in the query for better retrieval
        # (e.g., "AP" -> also search "Andhra Pradesh")
        enhanced_query = user_query
        for code, full_name in [("AP", "Andhra Pradesh"), ("UP", "Uttar Pradesh"),
                                 ("MH", "Maharashtra"), ("KA", "Karnataka"),
                                 ("TN", "Tamil Nadu"), ("TS", "Telangana")]:
            if code in user_query.upper().split():
                enhanced_query = user_query + f" ({full_name})"
                break
        
        print(f"\nQuerying: \"{enhanced_query}\"")
        try:
            response = query_engine.query(enhanced_query)
            print("\n" + "="*60)
            print("ANSWER:")
            print("-"*60)
            print(response)
            print("="*60 + "\n")
        except Exception as e:
            print("Error querying:")
            traceback.print_exc()
    
    driver.close()
    print("Goodbye!")


if __name__ == "__main__":
    query_knowledge_graph()

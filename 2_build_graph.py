import os
import re
import sys
import glob
import json
import uuid
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import setup_global_settings, get_neo4j_driver
from utils import chunk_markdown, resolve_state_name

def parse_year_from_path(filepath):
    """Attempts to extract a financial year or calendar year from a filepath/filename.
    Avoids matching marker timestamps (long digit sequences) using lookarounds.
    """
    filepath_upper = filepath.upper()
    
    # Specific known AP files matching
    if "OPNO.54OF2024" in filepath_upper:
        return "2024"
    if "OPNO.65OF2025" in filepath_upper:
        return "2025-26"
    if "ORDEROPNO1012AND13OF2026" in filepath_upper:
        return "2026"
    if "FY202425" in filepath_upper or "FY2024-25" in filepath_upper or "FY 2024-25" in filepath_upper:
        return "2024-25"

    # 1. Search for FY 2024-25 or similar with lookarounds
    m = re.search(r'(?<!\d)(20\d{2})[\-\s\_]?(20\d{2}|\d{2})(?!\d)', filepath_upper)
    if m:
        y1 = m.group(1)
        y2 = m.group(2)
        if len(y2) == 2:
            y2 = y1[:2] + y2
        return f"{y1}-{y2[2:]}"
        
    # 2. Search for any standalone 4-digit year
    m2 = re.search(r'(?<!\d)(20\d{2})(?!\d)', filepath_upper)
    if m2:
        return m2.group(1)
        
    # 3. Fallback: scan path segments
    parts = Path(filepath).parts
    for p in parts:
        if p.isdigit() and len(p) == 4:
            return p
            
    return "Unknown"


def build_knowledge_graph():
    """Reads Andhra Pradesh Markdown documents, chunks them with table-awareness,
    generates embeddings, and stores everything directly in Neo4j.
    """
    print("Setting up global LLM and embedding settings...")
    llm = setup_global_settings()

    # Resolve the embedding model
    from llama_index.core.embeddings.utils import resolve_embed_model
    from llama_index.core import Settings
    embed_model = resolve_embed_model(Settings.embed_model)
    print(f"Embedding model loaded: {embed_model.model_name}")

    input_dir = "markdown_output"
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"Error: '{input_dir}' is empty or does not exist.")
        return

    # Find all markdown files recursively across all state directories
    md_files = glob.glob(os.path.join(input_dir, "**", "*.md"), recursive=True)
    if not md_files:
        print(f"No markdown files found in '{input_dir}'.")
        return
        
    print(f"Found {len(md_files)} markdown files to index.")

    # 1. Reset database
    print("Resetting Neo4j database...")
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        # Drop existing indexes to avoid conflicts
        try:
            session.run("DROP INDEX chunk_text_index IF EXISTS")
        except:
            pass
        try:
            session.run("DROP INDEX chunk_vector_index IF EXISTS")
        except:
            pass
    print("Database cleared.")

    # 2. Parse all documents into chunks
    all_chunks = []
    print("Parsing documents into table-aware chunks...")
    for filepath in md_files:
        path_obj = Path(filepath)
        filename = path_obj.stem
        
        parts = path_obj.parts
        try:
            base_idx = parts.index("markdown_output")
            state_raw = parts[base_idx + 1] if len(parts) > base_idx + 1 else "Unknown"
        except ValueError:
            state_raw = "Unknown"
            
        state = resolve_state_name(state_raw)
        year = parse_year_from_path(filepath)
        
        if year == "Unknown":
            for parent in path_obj.parents:
                py = parse_year_from_path(parent.name)
                if py != "Unknown":
                    year = py
                    break
        
        print(f"  - {filename} | State: {state} | Year: {year}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"    Failed to read: {e}")
            continue
            
        chunks = chunk_markdown(content, max_chunk_size=1000, state=state, year=year, doc_name=filename)
        print(f"    {len(chunks)} chunks.")
        
        doc_id = str(uuid.uuid4())
        for idx, chunk in enumerate(chunks):
            chunk["state"] = state
            chunk["year"] = year
            chunk["document_name"] = filename
            chunk["doc_id"] = doc_id
            chunk["chunk_id"] = str(uuid.uuid4())
            all_chunks.append(chunk)
            
    if not all_chunks:
        print("No chunks parsed from documents. Exiting.")
        return
        
    print(f"\nTotal chunks: {len(all_chunks)}")

    # 3. Generate embeddings in batches
    print("Generating embeddings...")
    batch_size = 32
    texts = [c["text"] for c in all_chunks]
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = embed_model.get_text_embedding_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks")
    
    print("Embeddings generated successfully!")

    # 4. Insert chunks into Neo4j directly
    print("Inserting chunks into Neo4j...")
    with driver.session() as session:
        for i, chunk in enumerate(all_chunks):
            session.run("""
                CREATE (c:Chunk {
                    id: $id,
                    text: $text,
                    chunk_type: $chunk_type,
                    header_path: $header_path,
                    state: $state,
                    year: $year,
                    document_name: $document_name,
                    doc_id: $doc_id,
                    embedding: $embedding
                })
            """, {
                "id": chunk["chunk_id"],
                "text": chunk["text"],
                "chunk_type": chunk["type"],
                "header_path": chunk["header_path"],
                "state": chunk["state"],
                "year": chunk["year"],
                "document_name": chunk["document_name"],
                "doc_id": chunk["doc_id"],
                "embedding": all_embeddings[i]
            })
            
            if (i + 1) % 50 == 0 or i == len(all_chunks) - 1:
                print(f"  Inserted {i+1}/{len(all_chunks)} chunks")
    
    print("All chunks inserted!")

    # 5. Create structured metadata nodes and relationships
    print("Creating structured metadata graph...")
    with driver.session() as session:
        # Document nodes
        print("  - Creating Document nodes...")
        session.run("""
            MATCH (c:Chunk)
            WITH DISTINCT c.document_name AS name, c.doc_id AS doc_id, c.state AS state, c.year AS year
            MERGE (d:Document {name: name})
            ON CREATE SET d.id = doc_id, d.state = state, d.year = year
        """)
        
        # Link Documents -> Chunks
        print("  - Linking Documents to Chunks...")
        session.run("""
            MATCH (c:Chunk), (d:Document {name: c.document_name})
            MERGE (d)-[:HAS_CHUNK]->(c)
        """)
        
        # State nodes
        print("  - Creating State nodes...")
        session.run("""
            MATCH (d:Document)
            WHERE d.state IS NOT NULL
            MERGE (s:State {name: d.state})
            MERGE (d)-[:FOR_STATE]->(s)
            MERGE (s)-[:HAS_DOCUMENT]->(d)
        """)
        
        # Year nodes
        print("  - Creating Year nodes...")
        session.run("""
            MATCH (d:Document)
            WHERE d.year IS NOT NULL
            MERGE (y:Year {year: d.year})
            MERGE (d)-[:FOR_YEAR]->(y)
            MERGE (y)-[:HAS_DOCUMENT]->(d)
        """)
        
        # Full-text index
        print("  - Creating full-text index...")
        session.run("""
            CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS
            FOR (n:Chunk)
            ON EACH [n.text]
        """)
        
        # Vector index
        print("  - Creating vector index...")
        session.run("""
            CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
            FOR (n:Chunk)
            ON (n.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
        """)

    print("\n=== Graph building completed successfully! ===")
    
    # 6. Verify
    with driver.session() as session:
        result = session.run("MATCH (c:Chunk) RETURN count(c) as cnt").single()
        print(f"  Chunks: {result['cnt']}")
        result = session.run("MATCH (d:Document) RETURN count(d) as cnt").single()
        print(f"  Documents: {result['cnt']}")
        result = session.run("MATCH (s:State) RETURN count(s) as cnt").single()
        print(f"  States: {result['cnt']}")
        result = session.run("MATCH (y:Year) RETURN count(y) as cnt").single()
        print(f"  Years: {result['cnt']}")
        
        print("\n  Document summary:")
        results = session.run("MATCH (d:Document) RETURN d.name as name, d.state as state, d.year as year")
        for r in results:
            print(f"    - {r['name']} | {r['state']} | {r['year']}")

    driver.close()

if __name__ == "__main__":
    build_knowledge_graph()

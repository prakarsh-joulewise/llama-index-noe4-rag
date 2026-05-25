# Implementation Plan - Wheeling Charges Retrieval Optimization

This plan resolves the retrieval issue where querying for proposed and approved wheeling charges in Uttar Pradesh (UP) yields incorrect/empty answers. 

## Problem Analysis & Findings

Our analysis of the Neo4j database and search logs revealed two main bugs:
1. **Sticky Header Contamination**: In [utils.py](file:///c:/Joulewise/Llama%20index%20rag/utils.py), headers containing table captions (e.g., `TABLE 6-104`) are set as level-1 (`#`) headers. Because there are no subsequent level-1 headers in the document, this header remains active forever and is prepended as context to all downstream chunks (including Chapters 9 and 10). This contaminates the text of almost every chunk in the database with words like "APPROVED", "APPROVED FOR MVVNL", and "UP TO" (which contains "UP").
2. **Retrieval Text-Mismatch**: The actual wheeling charge tables in the document do not contain the tokens "UP" or "Uttar Pradesh" in their text. Consequently, when a user queries for "wheeling charges in UP", these tables score extremely low in both vector similarity and fulltext search compared to the contaminated chunks that accidentally contain the token "UP" in their headers.

---

## Proposed Changes

We propose the following three-part optimization strategy:

### 1. Robust Header Path Tracking
We will modify [utils.py](file:///c:/Joulewise/Llama%20index%20rag/utils.py) to:
- Expand header tracking up to level 4 (`####`).
- Reset the hierarchy correctly: a header of level `N` resets all tracked headers of level `>= N`.
- Identify and ignore non-structural headers (e.g., headers containing "TABLE", "Table", "LIST OF TABLES", etc.) so they do not become sticky context for downstream chunks.

### 2. Metadata Enrichment in Chunk Text
We will modify the chunking pipeline in [2_build_graph.py](file:///c:/Joulewise/Llama%20index%20rag/2_build_graph.py) and [utils.py](file:///c:/Joulewise/Llama%20index%20rag/utils.py) to prepend the state name and year directly to the text of the chunk before embedding/indexing:
`text = f"State: {state} | Year: {year} | Document: {doc_name} | Context: {header_path}\n\n{content_text}"`
This ensures the terms "UP", "Uttar Pradesh", and the fiscal year are embedded and indexed, making the target tables highly matching.

### 3. Query-Time State Filtering
We will update `HybridNeo4jRetriever` in [3_query_graph.py](file:///c:/Joulewise/Llama%20index%20rag/3_query_graph.py) to:
- Detect the target state (e.g., "Uttar Pradesh", "Andhra Pradesh") directly from the query string.
- If a state is detected, apply a Cypher metadata filter (`WHERE node.state = $state`) in the vector and full-text searches to restrict the search space, guaranteeing that only chunks from the relevant state are retrieved.
- Increase the candidate retrieval limit (e.g., limit of 100) before filtering to ensure we retrieve a sufficient number of high-quality nodes for the target state.

---

## Detailed File Changes

### [RAG Utilities & Chunking]

#### [MODIFY] [utils.py](file:///c:/Joulewise/Llama%20index%20rag/utils.py)
- Update `chunk_markdown` to support level-4 (`####`) headers and proper hierarchy resetting.
- Prevent table captions matching `\b(TABLE|Table|LIST OF TABLES|LIST OF FIGURES|CONTENTS)\b` from being stored as sticky headers.
- Update `chunk_markdown` parameters to accept `state` and `year` to prepend them to chunk text.

### [Data Ingestion & Indexing]

#### [MODIFY] [2_build_graph.py](file:///c:/Joulewise/Llama%20index%20rag/2_build_graph.py)
- Pass `state` and `year` to `chunk_markdown` so they are prepended to the text field before embedding.
- Clear the Neo4j database and re-ingest the documents to rebuild the vector and fulltext indexes.

### [Retrieval & Querying]

#### [MODIFY] [3_query_graph.py](file:///c:/Joulewise/Llama%20index%20rag/3_query_graph.py)
- Implement `detect_state_from_query` inside `HybridNeo4jRetriever`.
- Update `_vector_search` and `_fulltext_search` to support an optional `state` parameter and apply `WHERE node.state = $state` filter in Cypher.

---

## Verification Plan

### Automated Tests
- Run `2_build_graph.py` to regenerate the Neo4j database and indices.
- Run `3_query_graph.py` and query: `"tell me what are the proposed and approved wheeling charges applicable in UP"`
- Verify that the response contains:
  - **Proposed Wheeling Charge**: `1.03` Rs. / kWh (as submitted by petitioner).
  - **Approved Wheeling Charge**: `0.9985` Rs. / kWh (as approved by the Commission).
  - Citations of the UPPCL Tariff Order FY 2025-26.

import re

# Indian State Code to Full State Name mapping
STATE_CODE_MAPPING = {
    "AP": "Andhra Pradesh",
    "UP": "Uttar Pradesh",
    "MH": "Maharashtra",
    "DL": "Delhi",
    "KA": "Karnataka",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "GJ": "Gujarat",
    "RJ": "Rajasthan",
    "MP": "Madhya Pradesh",
    "WB": "West Bengal",
    "HR": "Haryana",
    "PB": "Punjab",
    "KL": "Kerala",
    "BR": "Bihar",
    "JH": "Jharkhand",
    "OD": "Odisha",
    "AS": "Assam",
    "HP": "Himachal Pradesh",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "JK": "Jammu and Kashmir",
    "GA": "Goa",
    "TR": "Tripura",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "SK": "Sikkim",
    "AR": "Arunachal Pradesh",
    "CH": "Chandigarh",
    "PY": "Puducherry",
    "AN": "Andaman and Nicobar",
    "LD": "Lakshadweep",
    "DN": "Dadra and Nagar Haveli",
    "DD": "Daman and Diu"
}

def resolve_state_name(state_input):
    """Resolves short-form state codes to full state names, case-insensitively.
    E.g., 'up' -> 'Uttar Pradesh', 'AP' -> 'Andhra Pradesh'.
    """
    if not state_input:
        return state_input
    
    clean_input = state_input.strip().upper()
    
    # Try direct mapping lookup
    if clean_input in STATE_CODE_MAPPING:
        return STATE_CODE_MAPPING[clean_input]
        
    # Check if input matches full name in mapping (case-insensitive)
    for code, name in STATE_CODE_MAPPING.items():
        if clean_input == name.upper():
            return name
            
    return state_input.title()

def clean_cell(cell_text):
    """Removes HTML tags, markdown bold/italic tags, and normalizes spacing in table cells."""
    # Remove HTML tags (e.g. <br>, <span ...></span>)
    text = re.sub(r'<[^>]*>', ' ', cell_text)
    # Remove markdown formatting stars and underscores
    text = re.sub(r'[\*\_]', '', text)
    # Normalize multiple spaces/newlines into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_table_narratives(table_lines):
    """Parses a Markdown table into rows and generates row-by-row narrative sentences.
    Links the row key (col 0) with column headers and cell values.
    Propagates category values for column 0 and column 1 across merged (empty) cells.
    """
    rows = []
    for line in table_lines:
        stripped = line.strip()
        if not stripped.startswith('|'):
            continue
        # Split by | and strip whitespaces. Drop first and last since they are empty due to leading/trailing |
        cells = [c.strip() for c in stripped.split('|')[1:-1]]
        rows.append(cells)
        
    if len(rows) < 2:
        return []
        
    # Find the header row (first non-separator row)
    header_idx = -1
    for idx, row in enumerate(rows):
        is_sep = all(re.match(r'^[\-\:\s]+$', c) for c in row) if row else False
        if not is_sep:
            header_idx = idx
            break
            
    if header_idx == -1 or header_idx >= len(rows) - 1:
        return []
        
    headers = [clean_cell(c) for c in rows[header_idx]]
    if not headers:
        return []
        
    # Track last seen non-empty values for columns 0 and 1 to handle merged/hierarchical cells
    last_val_col0 = ""
    last_val_col1 = ""
    
    narratives = []
    # Process remaining rows
    for idx in range(header_idx + 1, len(rows)):
        row = rows[idx]
        is_sep = all(re.match(r'^[\-\:\s]+$', c) for c in row) if row else False
        if is_sep or not row:
            continue
            
        cleaned_row = [clean_cell(c) for c in row]
        if not cleaned_row:
            continue
            
        # Resolve column 0 hierarchy
        if len(cleaned_row) > 0:
            val0 = cleaned_row[0]
            if val0 and val0 not in ['-', '—', 'None', 'nil', '']:
                last_val_col0 = val0
            else:
                cleaned_row[0] = last_val_col0
                
        # Resolve column 1 hierarchy
        if len(cleaned_row) > 1:
            val1 = cleaned_row[1]
            if val1 and val1 not in ['-', '—', 'None', 'nil', '']:
                last_val_col1 = val1
            else:
                cleaned_row[1] = last_val_col1
                
        if not cleaned_row or not cleaned_row[0]:
            continue
            
        row_key = cleaned_row[0]
        col_descriptions = []
        for j in range(1, min(len(headers), len(cleaned_row))):
            col_name = headers[j]
            val = cleaned_row[j]
            # Skip empty cells or simple dash separators
            if val and val not in ['-', '—', 'None', 'nil', '']:
                col_descriptions.append(f"{col_name} is {val}")
                
        if col_descriptions:
            desc_sentence = f"- For \"{row_key}\": " + ", ".join(col_descriptions) + "."
            narratives.append(desc_sentence)
            
    return narratives

def is_structural_header(header_text):
    """Checks if a header is structural or just a table/list caption/toc."""
    clean = header_text.upper()
    # Match against common patterns for non-structural header sections
    if any(pattern in clean for pattern in ["TABLE", "LIST OF TABLES", "LIST OF FIGURES", "CONTENTS"]):
        return False
    return True

def format_chunk_text(text, state=None, year=None, doc_name=None, header_path=None):
    """Prepends a metadata line to the chunk text to ensure state, year, and context
    are indexable and searchable.
    """
    metadata_parts = []
    if state:
        metadata_parts.append(f"State: {state}")
    if year:
        metadata_parts.append(f"Year: {year}")
    if doc_name:
        metadata_parts.append(f"Document: {doc_name}")
    if header_path and header_path != "/":
        metadata_parts.append(f"Context: {header_path}")
        
    if metadata_parts:
        prefix = " | ".join(metadata_parts)
        return f"{prefix}\n\n{text}"
    return text

def chunk_markdown(content, max_chunk_size=1000, state=None, year=None, doc_name=None):
    """Splits a markdown document into chunks, keeping tables atomic,
    merging multi-page tables, and generating row narrative text for tables. 
    Tracks header hierarchy to assign correct header_path metadata.
    Uses semantic sentence-boundary chunking and sliding sentence overlap.
    """
    lines = content.split('\n')
    chunks = []
    
    current_text_block = []
    current_text_size = 0
    
    in_table = False
    table_lines = []
    
    # Header hierarchy tracking
    h1, h2, h3, h4 = "", "", "", ""
    
    def get_header_path():
        parts = []
        if h1: parts.append(h1.strip('*# '))
        if h2: parts.append(h2.strip('*# '))
        if h3: parts.append(h3.strip('*# '))
        if h4: parts.append(h4.strip('*# '))
        return "/" + "/".join(parts) + "/" if parts else "/"

    def flush_text_block(carry_overlap=True):
        nonlocal current_text_block, current_text_size
        if current_text_block:
            text = "\n".join(current_text_block).strip()
            if text:
                header_path = get_header_path()
                full_text = format_chunk_text(text, state=state, year=year, doc_name=doc_name, header_path=header_path)
                chunks.append({
                    "text": full_text,
                    "type": "text",
                    "header_path": header_path
                })
            
            # Sentence-boundary overlap: carry forward the last ~180 characters (ending on sentence lines)
            if carry_overlap and len(current_text_block) > 1:
                max_overlap_idx = len(current_text_block) // 2
                overlap_lines = []
                overlap_size = 0
                for i in range(len(current_text_block) - 1, -1, -1):
                    line = current_text_block[i]
                    overlap_lines.insert(0, line)
                    overlap_size += len(line) + 1
                    if overlap_size >= 180 or len(overlap_lines) >= max_overlap_idx:
                        break
                current_text_block = overlap_lines
                current_text_size = overlap_size
            else:
                current_text_block = []
                current_text_size = 0

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        
        # Track active headers to determine structural path context
        if stripped.startswith('# '):
            flush_text_block(carry_overlap=False)
            header_text = stripped[2:].strip()
            if is_structural_header(header_text):
                h1 = header_text
                h2, h3, h4 = "", "", ""
        elif stripped.startswith('## '):
            flush_text_block(carry_overlap=False)
            header_text = stripped[3:].strip()
            if is_structural_header(header_text):
                h2 = header_text
                h3, h4 = "", ""
        elif stripped.startswith('### '):
            flush_text_block(carry_overlap=False)
            header_text = stripped[4:].strip()
            if is_structural_header(header_text):
                h3 = header_text
                h4 = ""
        elif stripped.startswith('#### '):
            flush_text_block(carry_overlap=False)
            header_text = stripped[5:].strip()
            if is_structural_header(header_text):
                h4 = header_text
        
        # Detect table lines
        is_table_line = stripped.startswith('|')
        if is_table_line:
            if not in_table:
                flush_text_block(carry_overlap=False)
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
        else:
            if in_table:
                # Exiting table mode? Perform lookahead to check for multi-page table continuation
                is_continuation = False
                lookahead_lines = []
                lookahead_idx = idx
                
                # Check next 8 lines for a table continuation
                while lookahead_idx < len(lines) and (lookahead_idx - idx) < 9:
                    next_line = lines[lookahead_idx]
                    next_stripped = next_line.strip()
                    if next_stripped.startswith('|'):
                        is_continuation = True
                        break
                    lookahead_lines.append(next_line)
                    lookahead_idx += 1
                    
                if is_continuation:
                    # Table continues after page breaks/noise. Skip noise and consume continuation line.
                    idx = lookahead_idx
                    cont_line = lines[idx]
                    
                    # If repeating column headers are found, skip them
                    is_header_repeat = False
                    if idx + 1 < len(lines):
                        next_cont_line = lines[idx + 1]
                        next_cells = [c.strip() for c in next_cont_line.strip().split('|')[1:-1]]
                        if next_cells and all(re.match(r'^[\-\:\s]+$', c) for c in next_cells):
                            is_header_repeat = True
                            
                    if is_header_repeat:
                        idx += 2  # Skip header and separator row
                    else:
                        table_lines.append(cont_line)
                        idx += 1
                    continue
                else:
                    # No continuation. Flush table chunk
                    in_table = False
                    table_text = "\n".join(table_lines)
                    narratives = generate_table_narratives(table_lines)
                    full_chunk_text = table_text
                    if narratives:
                        full_chunk_text += "\n\nTable Narratives:\n" + "\n".join(narratives)
                    
                    header_path = get_header_path()
                    full_chunk_text = format_chunk_text(full_chunk_text, state=state, year=year, doc_name=doc_name, header_path=header_path)
                    
                    chunks.append({
                        "text": full_chunk_text,
                        "type": "table",
                        "header_path": header_path
                    })
                    table_lines = []
            
            # Normal text block accumulation
            current_text_block.append(line)
            current_text_size += len(line) + 1
            
            # Check for sentence/paragraph boundaries when threshold is met
            if current_text_size >= 800:
                has_boundary = False
                if stripped.endswith(('.', '?', '!', '"', "'")):
                    has_boundary = True
                elif idx + 1 < len(lines) and not lines[idx+1].strip():
                    # Next line is empty (paragraph boundary)
                    has_boundary = True
                elif current_text_size >= 1200:
                    # Hard character limit fallback
                    has_boundary = True
                    
                if has_boundary:
                    flush_text_block(carry_overlap=True)
            
        idx += 1
                
    # Flush any remaining items at EOF
    if in_table:
        table_text = "\n".join(table_lines)
        narratives = generate_table_narratives(table_lines)
        full_chunk_text = table_text
        if narratives:
            full_chunk_text += "\n\nTable Narratives:\n" + "\n".join(narratives)
            
        header_path = get_header_path()
        full_chunk_text = format_chunk_text(full_chunk_text, state=state, year=year, doc_name=doc_name, header_path=header_path)
            
        chunks.append({
            "text": full_chunk_text,
            "type": "table",
            "header_path": header_path
        })
    else:
        flush_text_block(carry_overlap=False)
        
    return chunks

import os
import subprocess
import glob
from pathlib import Path

INPUT_DIR = "input_docs"
OUTPUT_DIR = "markdown_output"

def convert_all_pdfs():
    """Converts all PDFs in the input directory to Markdown using marker."""
    
    # Ensure directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    input_path = Path(INPUT_DIR)
    pdf_files = list(input_path.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}. Please place some legal documents there.")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting conversion...")

    for pdf_path in pdf_files:
        # Determine the relative folder structure (e.g., UP/2025)
        rel_dir = pdf_path.parent.relative_to(input_path)
        
        # We don't want to create markdown_output/UP/2025/filename. 
        # Marker creates its own folder for the output, so we just pass markdown_output/UP/2025 as the output_dir.
        target_out_dir = Path(OUTPUT_DIR) / rel_dir
        os.makedirs(target_out_dir, exist_ok=True)
        
        filename = pdf_path.stem
        print(f"Converting {filename} from {rel_dir}...")
        
        # Marker CLI command to convert a single PDF
        try:
            subprocess.run(
                ["marker_single", str(pdf_path), "--output_dir", str(target_out_dir), "--output_format", "markdown"],
                check=True
            )
            print(f"Successfully converted {filename}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {filename}. Error: {e}")
        except FileNotFoundError:
            print("Error: 'marker_single' command not found. Ensure marker-pdf is installed correctly.")
            break

if __name__ == "__main__":
    convert_all_pdfs()

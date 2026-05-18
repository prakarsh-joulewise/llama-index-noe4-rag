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
    
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}. Please place some legal documents there.")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting conversion...")

    for pdf_path in pdf_files:
        filename = Path(pdf_path).stem
        output_folder = os.path.join(OUTPUT_DIR, filename)
        
        print(f"Converting {filename}...")
        
        # Marker CLI command to convert a single PDF
        # Note: Depending on your exact marker-pdf version, the command might be 'marker_single'
        # Usage: marker_single /path/to/file.pdf /path/to/output/folder
        try:
            subprocess.run(
                ["marker_single", pdf_path, "--output_dir", OUTPUT_DIR, "--output_format", "markdown"],
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

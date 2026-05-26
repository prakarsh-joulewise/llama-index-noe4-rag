import os
import subprocess
from pathlib import Path

INPUT_DIR = "input_docs"
OUTPUT_DIR = "markdown_output_mineru"  # Suffix to distinguish from marker output

def convert_all_pdfs():
    """Converts all PDFs in the input directory to Markdown using MinerU (magic-pdf / mineru)."""
    
    # Ensure directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    input_path = Path(INPUT_DIR)
    pdf_files = list(input_path.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}. Please place some PDF documents there.")
        return

    # Check if magic-pdf.json exists in user's home directory
    config_path = Path.home() / "magic-pdf.json"
    if not config_path.exists():
        print("="*80)
        print(f"Error: MinerU configuration file not found at: {config_path}")
        print("MinerU requires downloading model weights and initializing this file before running.")
        print("\nPlease run the following commands on your server to download models and set it up:")
        print("  1. Download the helper script:")
        print("     wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py")
        print("  2. Run the download script (this will fetch weights and auto-generate magic-pdf.json):")
        print("     python download_models_hf.py")
        print("  3. Verify the 'models-dir' path in ~/magic-pdf.json points to the downloaded weights.")
        print("="*80)
        return

    print(f"Found {len(pdf_files)} PDF files. Starting MinerU conversion...")

    # Determine command to use: magic-pdf or mineru
    cmd_name = "magic-pdf"
    try:
        # Check if magic-pdf is available
        subprocess.run(["magic-pdf", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        try:
            # Fallback to mineru
            subprocess.run(["mineru", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            cmd_name = "mineru"
        except FileNotFoundError:
            print("Error: Neither 'magic-pdf' nor 'mineru' command was found.")
            print("Please ensure MinerU is installed (pip install magic-pdf[full] or mineru) and configured.")
            return

    print(f"Using CLI tool: '{cmd_name}'")

    for pdf_path in pdf_files:
        # Determine the relative folder structure (e.g., UP/2025)
        rel_dir = pdf_path.parent.relative_to(input_path)
        
        # We preserve the relative folder structure
        target_out_dir = Path(OUTPUT_DIR) / rel_dir
        os.makedirs(target_out_dir, exist_ok=True)
        
        filename = pdf_path.stem
        print(f"Converting {filename} from {rel_dir}...")
        
        # MinerU CLI command to convert a single PDF:
        # -p is input path
        # -o is output directory
        try:
            subprocess.run(
                [cmd_name, "-p", str(pdf_path), "-o", str(target_out_dir)],
                check=True
            )
            print(f"Successfully converted {filename}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {filename}. Error: {e}")

if __name__ == "__main__":
    convert_all_pdfs()

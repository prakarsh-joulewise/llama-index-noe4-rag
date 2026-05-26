import os
import json
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

def setup_models():
    home_dir = Path.home()
    # Save the models in the user's home directory
    models_dir = home_dir / "magic-pdf-models"
    config_file = home_dir / "magic-pdf.json"
    
    print(f"Target models directory: {models_dir}")
    print("Downloading model weights from Hugging Face (opendatalab/PDF-Extract-Kit)...")
    
    # Download the weights using huggingface_hub
    snapshot_download(
        repo_id="opendatalab/PDF-Extract-Kit",
        local_dir=str(models_dir),
        max_workers=8
    )
    
    print("Download completed successfully!")
    
    # Automatically detect if CUDA (GPU) is available and configure device mode
    device_mode = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Detected torch.cuda.is_available() = {torch.cuda.is_available()}. Setting device-mode to '{device_mode}'.")
    
    # Structure of magic-pdf.json configuration
    config_data = {
        "bucket_info": {
            "bucket-name-1": ["ak", "sk", "endpoint"],
            "bucket-name-2": ["ak", "sk", "endpoint"]
        },
        "models-dir": str(models_dir),
        "device-mode": device_mode,
        "layout-config": {
            "model": "doclayout_yolo"
        },
        "formula-config": {
            "mfd_model": "yolo_v8_mfd",
            "mfr_model": "unimernet_small",
            "enable": True
        },
        "table-config": {
            "model": "rapid_table",
            "enable": False,
            "max_time": 400
        },
        "config_version": "1.0.0"
    }
    
    # Write the configuration file
    print(f"Writing configuration to {config_file}...")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
        
    print("="*80)
    print("MinerU Models and Configuration setup is COMPLETE!")
    print(f"Configuration written to: {config_file}")
    print(f"Model weights saved at: {models_dir}")
    print("="*80)

if __name__ == "__main__":
    setup_models()

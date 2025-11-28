#!/usr/bin/env python3
"""
Train EOLE NMT Model for Lexical and Syntactic Transfer
"""

import subprocess
import sys
from pathlib import Path

def train_eole_model():
    """Train EOLE model using the configuration file"""
    print("="*50)
    print("Training EOLE NMT Model")
    print("="*50)
    
    config_file = "configs/eole_config.yml"
    
    if not Path(config_file).exists():
        print(f"Error: Config file {config_file} not found!")
        sys.exit(1)
    
    # Check if SentencePiece models exist
    if not Path("models/spm.en.model").exists():
        print("Error: SentencePiece models not found!")
        print("Please run: python scripts/03_train_sentencepiece.py")
        sys.exit(1)
    
    # Create log directory
    Path("logs").mkdir(exist_ok=True)
    
    print("\nStarting EOLE training...")
    print("This may take several hours depending on your hardware.")
    print("Monitor training with: tensorboard --logdir logs/tensorboard\n")
    
    # Train using EOLE
    cmd = [
        "eole",
        "train",
        "--config", config_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*50)
        print("EOLE training complete!")
        print("="*50)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        sys.exit(1)

def translate_with_eole(input_file, output_file, model_path):
    """Translate using trained EOLE model"""
    print(f"\nTranslating: {input_file} -> {output_file}")
    
    cmd = [
        "eole",
        "predict",
        "--model", model_path,
        "--src", input_file,
        "--output", output_file,
        "--gpu", "0",
        "--batch_size", "32",
        "--beam_size", "5"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Translation saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during translation: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or use EOLE NMT model")
    parser.add_argument("--mode", choices=["train", "translate"], default="train",
                       help="Mode: train or translate")
    parser.add_argument("--input", type=str, help="Input file for translation")
    parser.add_argument("--output", type=str, help="Output file for translation")
    parser.add_argument("--model", type=str, default="models/eole_model_step_100000.pt",
                       help="Model checkpoint path")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_eole_model()
    elif args.mode == "translate":
        if not args.input or not args.output:
            print("Error: --input and --output required for translation mode")
            sys.exit(1)
        translate_with_eole(args.input, args.output, args.model)

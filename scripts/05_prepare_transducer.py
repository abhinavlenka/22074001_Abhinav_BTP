#!/usr/bin/env python3
"""
Prepare data for Neural Transducer (Morphological Transfer)
Format: character-level alignment for morphological generation
"""

import os
import sys
from pathlib import Path

def prepare_transducer_data():
    """
    Clone and setup Neural Transducer repository
    Prepare character-level data for morphological learning
    """
    print("="*50)
    print("Setting up Neural Transducer")
    print("="*50)
    
    transducer_dir = Path("neural-transducer")
    
    # Clone repository if not exists
    if not transducer_dir.exists():
        print("\nCloning Neural Transducer repository...")
        os.system("git clone https://github.com/shijie-wu/neural-transducer.git")
    else:
        print("\nNeural Transducer repository already exists.")
    
    # Create data directory for transducer
    trans_data_dir = transducer_dir / "data" / "en-hi"
    trans_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nPreparing morphological transfer data...")
    
    # Read source data
    data_dir = Path("data")
    
    for split in ["train", "dev", "test"]:
        print(f"Processing {split} split...")
        
        # Read English and Hindi
        with open(data_dir / f"{split}.en", "r", encoding="utf-8") as f:
            en_lines = f.readlines()
        
        with open(data_dir / f"{split}.hi", "r", encoding="utf-8") as f:
            hi_lines = f.readlines()
        
        # Character-level format for transducer
        # Format: source_char_sequence ||| target_char_sequence
        
        with open(trans_data_dir / f"{split}.txt", "w", encoding="utf-8") as f:
            for en, hi in zip(en_lines, hi_lines):
                en = en.strip()
                hi = hi.strip()
                
                if en and hi:
                    # Character-level with space separation
                    en_chars = " ".join(list(en))
                    hi_chars = " ".join(list(hi))
                    
                    f.write(f"{en_chars} ||| {hi_chars}\n")
        
        print(f"Saved {split}.txt with {len(en_lines)} examples")
    
    print("\n" + "="*50)
    print("Neural Transducer data preparation complete!")
    print("="*50)
    
    print("\nNext steps:")
    print("1. cd neural-transducer")
    print("2. Follow the training instructions in the repository")
    print("3. Train the model: python train.py --config configs/en-hi.yaml")

def create_transducer_config():
    """Create configuration file for neural transducer"""
    config_content = """# Neural Transducer Configuration for English-Hindi
# Character-level morphological transfer

# Data
data_dir: data/en-hi
train_file: train.txt
dev_file: dev.txt
test_file: test.txt

# Model architecture
model_type: transformer
hidden_size: 256
num_layers: 4
num_heads: 8
ff_size: 1024
dropout: 0.3

# Training
batch_size: 64
learning_rate: 0.001
max_epochs: 50
patience: 10
gradient_clip: 5.0

# Decoding
beam_size: 10
max_decode_len: 200

# Paths
save_dir: models/transducer
log_dir: logs/transducer
"""
    
    config_dir = Path("neural-transducer/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_dir / "en-hi.yaml", "w") as f:
        f.write(config_content)
    
    print("Created transducer config: neural-transducer/configs/en-hi.yaml")

if __name__ == "__main__":
    prepare_transducer_data()
    create_transducer_config()

#!/usr/bin/env python3
"""
Train SentencePiece models for English and Hindi
Required for EOLE-based NMT
"""

import sentencepiece as spm
from pathlib import Path

def train_sentencepiece(input_file, model_prefix, vocab_size=32000):
    """Train a SentencePiece model"""
    print(f"\nTraining SentencePiece model: {model_prefix}")
    print(f"Input: {input_file}")
    print(f"Vocab size: {vocab_size}")
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,  # Higher for Hindi script
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        user_defined_symbols=['<mask>'],
        num_threads=4
    )
    
    print(f"Model saved: {model_prefix}.model")
    print(f"Vocab saved: {model_prefix}.vocab")

def main():
    print("="*50)
    print("SentencePiece Training")
    print("="*50)
    
    data_dir = Path("data")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Train English model
    train_sentencepiece(
        input_file=str(data_dir / "train.en"),
        model_prefix=str(model_dir / "spm.en"),
        vocab_size=32000
    )
    
    # Train Hindi model
    train_sentencepiece(
        input_file=str(data_dir / "train.hi"),
        model_prefix=str(model_dir / "spm.hi"),
        vocab_size=32000
    )
    
    print("\n" + "="*50)
    print("SentencePiece training complete!")
    print("="*50)

if __name__ == "__main__":
    main()

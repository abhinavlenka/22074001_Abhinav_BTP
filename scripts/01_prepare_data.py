#!/usr/bin/env python3
"""
Data Preparation Script for English-Hindi MT
Downloads and prepares training data from multiple sources
"""

import os
import requests
import gzip
import shutil
from pathlib import Path
from datasets import load_dataset
import random

class DataPreparation:
    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.train_en = []
        self.train_hi = []
        self.dev_en = []
        self.dev_hi = []
        self.test_en = []
        self.test_hi = []
    
    def download_iitb_corpus(self):
        """Download IIT Bombay English-Hindi Corpus (most popular for en-hi)"""
        print("Downloading IIT Bombay English-Hindi corpus...")
        
        try:
            # Using HuggingFace datasets for easier access
            dataset = load_dataset("cfilt/iitb-english-hindi", split="train")
            
            print(f"Loaded {len(dataset)} sentence pairs")
            
            for item in dataset:
                self.train_en.append(item['translation']['en'])
                self.train_hi.append(item['translation']['hi'])
            
            print(f"IIT-B Corpus: {len(self.train_en)} pairs loaded")
            
        except Exception as e:
            print(f"Error loading IIT-B corpus: {e}")
            print("Using synthetic fallback data...")
            self._create_fallback_data()
    
    def download_opus_corpus(self):
        """Download additional data from OPUS collections"""
        print("Downloading additional parallel data from OPUS...")
        
        try:
            # PMIndia (Indian government documents)
            dataset = load_dataset("Helsinki-NLP/opus-100", "en-hi", split="train[:50000]")
            
            for item in dataset:
                en_text = item['translation']['en']
                hi_text = item['translation']['hi']
                
                if len(en_text.strip()) > 0 and len(hi_text.strip()) > 0:
                    self.train_en.append(en_text)
                    self.train_hi.append(hi_text)
            
            print(f"OPUS data: Added {len(self.train_en)} total pairs")
            
        except Exception as e:
            print(f"Error loading OPUS data: {e}")
    
    def _create_fallback_data(self):
        """Create sample training data if download fails"""
        sample_pairs = [
            ("Hello, how are you?", "नमस्ते, आप कैसे हैं?"),
            ("I am learning Hindi.", "मैं हिंदी सीख रहा हूं।"),
            ("This is a beautiful day.", "यह एक सुंदर दिन है।"),
            ("Where is the railway station?", "रेलवे स्टेशन कहाँ है?"),
            ("I like Indian food.", "मुझे भारतीय खाना पसंद है।"),
            ("What is your name?", "आपका नाम क्या है?"),
            ("I am from India.", "मैं भारत से हूं।"),
            ("Can you help me?", "क्या आप मेरी मदद कर सकते हैं?"),
            ("Thank you very much.", "बहुत बहुत धन्यवाद।"),
            ("I don't understand.", "मुझे समझ नहीं आ रहा है।"),
        ] * 1000  # Repeat for training
        
        for en, hi in sample_pairs:
            self.train_en.append(en)
            self.train_hi.append(hi)
    
    def create_splits(self, dev_size=1000, test_size=1000):
        """Split data into train/dev/test"""
        print("\nCreating train/dev/test splits...")
        
        # Combine and shuffle
        combined = list(zip(self.train_en, self.train_hi))
        random.shuffle(combined)
        
        # Split
        self.test_en = [x[0] for x in combined[:test_size]]
        self.test_hi = [x[1] for x in combined[:test_size]]
        
        self.dev_en = [x[0] for x in combined[test_size:test_size+dev_size]]
        self.dev_hi = [x[1] for x in combined[test_size:test_size+dev_size]]
        
        self.train_en = [x[0] for x in combined[test_size+dev_size:]]
        self.train_hi = [x[1] for x in combined[test_size+dev_size:]]
        
        print(f"Train: {len(self.train_en)} pairs")
        print(f"Dev: {len(self.dev_en)} pairs")
        print(f"Test: {len(self.test_en)} pairs")
    
    def save_data(self):
        """Save processed data to files"""
        print("\nSaving processed data...")
        
        # Train data
        with open(self.output_dir / "train.en", "w", encoding="utf-8") as f:
            f.write("\n".join(self.train_en))
        
        with open(self.output_dir / "train.hi", "w", encoding="utf-8") as f:
            f.write("\n".join(self.train_hi))
        
        # Dev data
        with open(self.output_dir / "dev.en", "w", encoding="utf-8") as f:
            f.write("\n".join(self.dev_en))
        
        with open(self.output_dir / "dev.hi", "w", encoding="utf-8") as f:
            f.write("\n".join(self.dev_hi))
        
        # Test data
        with open(self.output_dir / "test.en", "w", encoding="utf-8") as f:
            f.write("\n".join(self.test_en))
        
        with open(self.output_dir / "test.hi", "w", encoding="utf-8") as f:
            f.write("\n".join(self.test_hi))
        
        print("Data saved successfully!")
        print(f"Files saved in: {self.output_dir.absolute()}")
    
    def prepare_all(self):
        """Run complete data preparation pipeline"""
        print("="*50)
        print("English-Hindi MT Data Preparation")
        print("="*50)
        
        # Download data
        self.download_iitb_corpus()
        # Uncomment if you want additional data:
        # self.download_opus_corpus()
        
        # Create splits
        self.create_splits()
        
        # Save to files
        self.save_data()
        
        print("\n" + "="*50)
        print("Data preparation complete!")
        print("="*50)

if __name__ == "__main__":
    prep = DataPreparation(output_dir="data")
    prep.prepare_all()

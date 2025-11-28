#!/usr/bin/env python3
"""
SpaCy Preprocessing and Syntactic Analysis
Extracts syntactic features for transfer-based MT
"""

import spacy
import json
from pathlib import Path
from tqdm import tqdm
import pickle

class SpacyPreprocessor:
    def __init__(self):
        print("Loading SpaCy models...")
        # Load English model
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
        except:
            print("Downloading English model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp_en = spacy.load("en_core_web_sm")
    
    def analyze_sentence(self, text):
        """Perform syntactic analysis on a sentence"""
        doc = self.nlp_en(text)
        
        analysis = {
            "text": text,
            "tokens": [],
            "pos_tags": [],
            "dep_tags": [],
            "lemmas": [],
            "tree": []
        }
        
        for token in doc:
            analysis["tokens"].append(token.text)
            analysis["pos_tags"].append(token.pos_)
            analysis["dep_tags"].append(token.dep_)
            analysis["lemmas"].append(token.lemma_)
            
            # Dependency tree structure
            analysis["tree"].append({
                "id": token.i,
                "text": token.text,
                "pos": token.pos_,
                "dep": token.dep_,
                "head": token.head.i,
                "children": [child.i for child in token.children]
            })
        
        return analysis
    
    def preprocess_file(self, input_file, output_file, limit=None):
        """Process a file and save syntactic analyses"""
        print(f"Processing {input_file}...")
        
        with open(input_file, "r", encoding="utf-8") as f:
            sentences = f.readlines()
        
        if limit:
            sentences = sentences[:limit]
        
        analyses = []
        for sent in tqdm(sentences, desc="Analyzing"):
            sent = sent.strip()
            if sent:
                analysis = self.analyze_sentence(sent)
                analyses.append(analysis)
        
        # Save as both JSON and pickle
        output_path = Path(output_file)
        
        # Save JSON for inspection
        with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(analyses, f, ensure_ascii=False, indent=2)
        
        # Save pickle for faster loading
        with open(output_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(analyses, f)
        
        print(f"Saved {len(analyses)} analyses to {output_path.stem}.*")
        return analyses
    
    def extract_transfer_features(self, analysis):
        """Extract features useful for transfer"""
        features = {
            "surface_form": analysis["tokens"],
            "lemma_form": analysis["lemmas"],
            "pos_sequence": analysis["pos_tags"],
            "dependency_structure": analysis["tree"],
            "verb_positions": [i for i, pos in enumerate(analysis["pos_tags"]) if pos == "VERB"],
            "noun_positions": [i for i, pos in enumerate(analysis["pos_tags"]) if pos == "NOUN"],
            "adj_positions": [i for i, pos in enumerate(analysis["pos_tags"]) if pos == "ADJ"],
        }
        return features

def main():
    preprocessor = SpacyPreprocessor()
    
    # Process training data
    print("\n" + "="*50)
    print("Syntactic Analysis with SpaCy")
    print("="*50 + "\n")
    
    data_dir = Path("data")
    output_dir = Path("data/spacy_features")
    output_dir.mkdir(exist_ok=True)
    
    # Process English source files
    for split in ["train", "dev", "test"]:
        input_file = data_dir / f"{split}.en"
        output_file = output_dir / f"{split}_en_features"
        
        if input_file.exists():
            # Limit training data processing for speed (remove limit for full training)
            limit = 10000 if split == "train" else None
            preprocessor.preprocess_file(input_file, output_file, limit=limit)
    
    print("\n" + "="*50)
    print("Syntactic analysis complete!")
    print("="*50)

if __name__ == "__main__":
    main()

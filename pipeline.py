#!/usr/bin/env python3
"""
Complete English-Hindi Machine Translation Pipeline
Integrates: SpaCy -> EOLE -> Neural Transducer -> LLM Refinement
"""

import sys
import pickle
import torch
from pathlib import Path
import subprocess
import tempfile
import warnings
import os

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class EnglishHindiMTPipeline:
    """Complete MT pipeline orchestrator"""
    
    def __init__(self, models_dir="models", verbose=True):
        self.models_dir = Path(models_dir)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.verbose = verbose

        if self.verbose:
            print("Initializing English-Hindi MT Pipeline...")
        self._load_components()
    
    def _load_components(self):
        """Load all pipeline components"""
        # 1. SpaCy for syntactic analysis
        if self.verbose:
            print("Loading SpaCy...")
        import spacy
        try:
            self.spacy_model = spacy.load("en_core_web_sm")
        except:
            if self.verbose:
                print("Downloading SpaCy model...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.spacy_model = spacy.load("en_core_web_sm")
        
        # 2. EOLE model (if trained)
        self.eole_model = None
        eole_path = self.models_dir / "eole_model"
        if eole_path.exists() and (eole_path / "config.json").exists():
            self.eole_model = str(eole_path)
        
        # 3. Neural Transducer (if trained)
        self.transducer = None
        transducer_path = self.models_dir / "transducer" / "best_model.pt"
        vocab_path = self.models_dir / "transducer" / "vocab.pkl"
        
        if transducer_path.exists() and vocab_path.exists():
            self._load_transducer(transducer_path, vocab_path)
        
        # 4. LLM Refiner
        if self.verbose:
            print("Initializing LLM Refiner...")
        import importlib.util
        spec = importlib.util.spec_from_file_location("llm_refinement",
                                                       Path(__file__).parent / "scripts" / "07_llm_refinement.py")
        llm_refinement = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llm_refinement)
        self.llm_refiner = llm_refinement.LLMRefiner(api_provider="gemini", verbose=self.verbose)

        if self.verbose:
            print("Pipeline initialization complete!\n")
    
    def _load_transducer(self, model_path, vocab_path):
        """Load neural transducer model"""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_transducer",
                                                       Path(__file__).parent / "scripts" / "06_train_transducer.py")
        train_transducer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_transducer)
        Seq2SeqTransducer = train_transducer.Seq2SeqTransducer
        CharacterVocab = train_transducer.CharacterVocab

        # Load vocabularies (need to add CharacterVocab to globals for unpickling)
        import __main__
        __main__.CharacterVocab = CharacterVocab

        with open(vocab_path, 'rb') as f:
            vocabs = pickle.load(f)
            self.src_vocab = vocabs['src_vocab']
            self.tgt_vocab = vocabs['tgt_vocab']
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.transducer = Seq2SeqTransducer(
            src_vocab_size=len(self.src_vocab.char2idx),
            tgt_vocab_size=len(self.tgt_vocab.char2idx)
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.transducer.load_state_dict(checkpoint['model_state_dict'])
        self.transducer.eval()
        
        self.device = device
    
    def stage1_syntactic_analysis(self, text):
        """Stage 1: SpaCy syntactic analysis"""
        doc = self.spacy_model(text)
        
        analysis = {
            "tokens": [token.text for token in doc],
            "pos_tags": [token.pos_ for token in doc],
            "dep_tags": [token.dep_ for token in doc],
            "lemmas": [token.lemma_ for token in doc]
        }
        
        return analysis
    
    def stage2_lexical_transfer(self, text):
        """Stage 2: EOLE lexical and syntactic transfer"""
        if self.eole_model is None:
            return text  # Fallback to original text

        # Write input to temp file
        input_file = self.temp_dir / "input.txt"
        output_file = self.temp_dir / "eole_output.txt"

        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text)

        # Find eole command - try venv first, then system PATH
        eole_cmd = None
        venv_eole = Path(sys.executable).parent / "eole"
        if venv_eole.exists():
            eole_cmd = str(venv_eole)
        else:
            # Try to find in PATH
            import shutil
            eole_cmd = shutil.which("eole")

        if not eole_cmd:
            return text

        # Run EOLE translation
        cmd = [
            eole_cmd, "predict",
            "--model", self.eole_model,
            "--src", str(input_file),
            "--output", str(output_file),
            "--gpu", "0" if torch.cuda.is_available() else "-1",
            "--batch_size", "1",
            "--beam_size", "5"
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, stderr=subprocess.DEVNULL)
            with open(output_file, 'r', encoding='utf-8') as f:
                result = f.read().strip()
            return result
        except:
            # Silently fall back to original text on any error
            return text
    
    def stage3_morphological_transfer(self, text):
        """Stage 3: Neural Transducer morphological generation"""
        if self.transducer is None:
            return text
        
        # Encode input
        input_ids = [self.src_vocab.char2idx['<SOS>']] + \
                   self.src_vocab.encode(text) + \
                   [self.src_vocab.char2idx['<EOS>']]
        
        input_tensor = torch.LongTensor([input_ids]).to(self.device)
        
        # Generate with beam search (simplified)
        with torch.no_grad():
            # Encoder
            src_emb = self.transducer.src_embedding(input_tensor)
            encoder_outputs, (hidden, cell) = self.transducer.encoder(src_emb)

            # Reshape bidirectional encoder hidden states for decoder
            # hidden/cell: [num_layers * 2, batch, hidden_dim] -> [num_layers, batch, hidden_dim * 2]
            num_layers = hidden.size(0) // 2
            batch_size = hidden.size(1)
            hidden_dim = hidden.size(2)

            # Combine forward and backward hidden states
            hidden = hidden.view(num_layers, 2, batch_size, hidden_dim)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)

            cell = cell.view(num_layers, 2, batch_size, hidden_dim)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

            # Decoder (greedy decoding)
            decoder_input = torch.LongTensor([[self.tgt_vocab.char2idx['<SOS>']]]).to(self.device)
            decoder_hidden = (hidden, cell)
            
            output_ids = []
            max_length = len(text) * 3  # Allow for expansion
            
            for _ in range(max_length):
                tgt_emb = self.transducer.tgt_embedding(decoder_input)
                decoder_output, decoder_hidden = self.transducer.decoder(tgt_emb, decoder_hidden)
                logits = self.transducer.output_layer(decoder_output)
                
                pred_id = logits.argmax(dim=-1).item()
                
                if pred_id == self.tgt_vocab.char2idx['<EOS>']:
                    break
                
                output_ids.append(pred_id)
                decoder_input = torch.LongTensor([[pred_id]]).to(self.device)
            
            result = self.tgt_vocab.decode(output_ids)
            return result
    
    def stage4_llm_refinement(self, english_text, mt_output):
        """Stage 4: LLM-based few-shot refinement"""
        refined = self.llm_refiner.refine(english_text, mt_output)
        return refined
    
    def translate(self, text, use_stages=[1, 2, 3, 4]):
        """
        Complete translation pipeline

        Args:
            text: English input text
            use_stages: List of stages to use (1=SpaCy, 2=EOLE, 3=Transducer, 4=LLM)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Translating: {text}")
            print(f"{'='*60}\n")

        current_output = text

        # Stage 1: Syntactic Analysis
        if 1 in use_stages:
            if self.verbose:
                print("Stage 1: Syntactic Analysis (SpaCy)")
            analysis = self.stage1_syntactic_analysis(text)
            if self.verbose:
                print(f"  Tokens: {analysis['tokens']}")
                print(f"  POS: {analysis['pos_tags']}")
                print()

        # Stage 2: Lexical Transfer
        if 2 in use_stages:
            if self.verbose:
                print("Stage 2: Lexical & Syntactic Transfer (EOLE)")
            current_output = self.stage2_lexical_transfer(text)
            if self.verbose:
                print(f"  Output: {current_output}")
                print()

        # Stage 3: Morphological Transfer
        if 3 in use_stages:
            if self.verbose:
                print("Stage 3: Morphological Transfer (Neural Transducer)")
            current_output = self.stage3_morphological_transfer(current_output)
            if self.verbose:
                print(f"  Output: {current_output}")
                print()

        # Stage 4: LLM Refinement
        if 4 in use_stages:
            if self.verbose:
                print("Stage 4: LLM Refinement")
            current_output = self.stage4_llm_refinement(text, current_output)
            if self.verbose:
                print(f"  Output: {current_output}")
                print()

        if self.verbose:
            print(f"{'='*60}")
            print(f"Final Translation: {current_output}")
            print(f"{'='*60}\n")
        else:
            print(current_output)

        return current_output
    
    def translate_file(self, input_file, output_file, use_stages=[1, 2, 3, 4]):
        """Translate entire file"""
        print(f"Translating file: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        translations = []
        for line in lines:
            translation = self.translate(line, use_stages=use_stages)
            translations.append(translation)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(translations))
        
        print(f"Translations saved to: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="English-Hindi MT Pipeline")
    parser.add_argument("--text", type=str, help="Text to translate")
    parser.add_argument("--input_file", type=str, help="Input file to translate")
    parser.add_argument("--output_file", type=str, help="Output file for translations")
    parser.add_argument("--stages", nargs='+', type=int, default=[1, 2, 3, 4],
                       help="Stages to use (1=SpaCy, 2=EOLE, 3=Transducer, 4=LLM)")
    parser.add_argument("--quiet", action="store_true", help="Suppress all output except final translation")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = EnglishHindiMTPipeline(verbose=not args.quiet)
    
    if args.text:
        # Translate single text
        pipeline.translate(args.text, use_stages=args.stages)
    elif args.input_file and args.output_file:
        # Translate file
        pipeline.translate_file(args.input_file, args.output_file, use_stages=args.stages)
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("English-Hindi MT Pipeline - Interactive Mode")
        print("="*60)
        print("Enter English text to translate (or 'quit' to exit)")
        print()
        
        while True:
            text = input("English: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text:
                pipeline.translate(text, use_stages=args.stages)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download and prepare IIT Bombay English-Hindi Parallel Corpus
Using HuggingFace datasets for easy access
"""

import sys
from pathlib import Path
from tqdm import tqdm

def prepare_iitb_corpus(data_dir="data"):
    """
    Download and prepare IIT Bombay corpus using HuggingFace datasets
    Creates test set of 50k sentences
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    print("="*80)
    print("IIT Bombay English-Hindi Parallel Corpus Setup")
    print("="*80)
    print("Using HuggingFace datasets (cfilt/iitb-english-hindi)")
    print()

    # Import datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    # Download IIT Bombay corpus from HuggingFace
    print("Downloading IIT Bombay corpus from HuggingFace...")
    print("Note: First download may take a few minutes")
    print()

    try:
        dataset = load_dataset("cfilt/iitb-english-hindi", split="train")
        print(f"\n✓ Loaded {len(dataset):,} sentence pairs")
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nTrying alternative method...")
        try:
            # Try loading with specific revision
            dataset = load_dataset("cfilt/iitb-english-hindi", split="train", trust_remote_code=True)
            print(f"\n✓ Loaded {len(dataset):,} sentence pairs")
        except Exception as e2:
            print(f"\nError: {e2}")
            print("\nPlease check your internet connection and try again.")
            print("Alternatively, download manually from:")
            print("https://huggingface.co/datasets/cfilt/iitb-english-hindi")
            sys.exit(1)

    # Get English and Hindi texts
    print("\nExtracting English and Hindi sentences...")
    english_sentences = []
    hindi_sentences = []

    for item in tqdm(dataset, desc="Processing"):
        # Handle different possible field names
        if 'translation' in item:
            translation = item['translation']
            en_text = translation.get('en', translation.get('english', ''))
            hi_text = translation.get('hi', translation.get('hindi', ''))
        else:
            en_text = item.get('en', item.get('english', ''))
            hi_text = item.get('hi', item.get('hindi', ''))

        # Clean and filter
        en_text = en_text.strip()
        hi_text = hi_text.strip()

        if en_text and hi_text:
            english_sentences.append(en_text)
            hindi_sentences.append(hi_text)

    total_lines = len(english_sentences)
    print(f"\n✓ Extracted {total_lines:,} valid sentence pairs")

    # Create 50k test set
    print("\nCreating 50k sentence test set...")

    output_en = data_dir / "iitb_test_50k.en"
    output_hi = data_dir / "iitb_test_50k.hi"

    # Take last 50k sentences (avoid training data overlap)
    start_idx = max(0, total_lines - 50000)
    end_idx = total_lines

    # Write 50k test set
    count = 0
    with open(output_en, 'w', encoding='utf-8') as out_en, \
         open(output_hi, 'w', encoding='utf-8') as out_hi:

        for i in range(start_idx, end_idx):
            out_en.write(english_sentences[i] + '\n')
            out_hi.write(hindi_sentences[i] + '\n')
            count += 1

    print(f"\n✓ Created test set with {count:,} sentences")
    print(f"  English: {output_en}")
    print(f"  Hindi:   {output_hi}")

    # Also create smaller subsets for faster testing
    print("\nCreating smaller subsets...")

    for size in [1000, 5000, 10000]:
        subset_en = data_dir / f"iitb_test_{size}.en"
        subset_hi = data_dir / f"iitb_test_{size}.hi"

        with open(subset_en, 'w', encoding='utf-8') as out_en, \
             open(subset_hi, 'w', encoding='utf-8') as out_hi:

            for i in range(start_idx, min(start_idx + size, end_idx)):
                out_en.write(english_sentences[i] + '\n')
                out_hi.write(hindi_sentences[i] + '\n')

        print(f"  ✓ {size:>5,} sentences: {subset_en.name}")

    print("\n" + "="*80)
    print("Setup Complete!")
    print("="*80)
    print("\nCreated test files:")
    print(f"  • 1k sentences:  {data_dir}/iitb_test_1000.en/hi")
    print(f"  • 5k sentences:  {data_dir}/iitb_test_5000.en/hi")
    print(f"  • 10k sentences: {data_dir}/iitb_test_10000.en/hi")
    print(f"  • 50k sentences: {data_dir}/iitb_test_50k.en/hi")
    print("\nYou can now run evaluation:")
    print("  # Quick test (1k sentences)")
    print("  python evaluate_metrics.py --evaluate-pipeline \\")
    print("    --source data/iitb_test_1000.en --reference data/iitb_test_1000.hi")
    print()
    print("  # Full evaluation (50k sentences)")
    print("  python evaluate_large_dataset.py")
    print()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download IIT Bombay Corpus from HuggingFace")
    parser.add_argument("--data-dir", default="data", help="Data directory")

    args = parser.parse_args()

    try:
        prepare_iitb_corpus(args.data_dir)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

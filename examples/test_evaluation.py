#!/usr/bin/env python3
"""
Quick test example for MT evaluation metrics
Demonstrates all three metrics with sample data
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_metrics import MTEvaluator


def demo_basic_evaluation():
    """Demo basic evaluation with simple examples"""

    print("\n" + "="*80)
    print("DEMO 1: Basic Evaluation with Sample Translations")
    print("="*80)

    # Sample translations
    hypotheses = [
        "मैं घर जा रहा हूं",
        "यह एक अच्छी किताब है",
        "वह स्कूल में पढ़ता है",
        "आज मौसम बहुत अच्छा है",
        "मुझे हिंदी सीखना पसंद है"
    ]

    references = [
        "मैं घर जा रहा हूँ",
        "यह एक बहुत अच्छी किताब है",
        "वह स्कूल में पढ़ाई करता है",
        "आज का मौसम बहुत सुहावना है",
        "मुझे हिन्दी सीखना पसंद है"
    ]

    evaluator = MTEvaluator(verbose=True)
    results = evaluator.evaluate_all(hypotheses, references)

    return results


def demo_perfect_match():
    """Demo with perfect translation match"""

    print("\n" + "="*80)
    print("DEMO 2: Perfect Match (100% scores)")
    print("="*80)

    hypotheses = [
        "मैं खुश हूँ",
        "यह अच्छा है",
    ]

    references = [
        "मैं खुश हूँ",
        "यह अच्छा है",
    ]

    evaluator = MTEvaluator(verbose=True)
    results = evaluator.evaluate_all(hypotheses, references)

    return results


def demo_poor_match():
    """Demo with poor translation match"""

    print("\n" + "="*80)
    print("DEMO 3: Poor Match (low scores)")
    print("="*80)

    hypotheses = [
        "cat dog bird",
        "hello world",
    ]

    references = [
        "मैं खुश हूँ",
        "यह अच्छा है",
    ]

    evaluator = MTEvaluator(verbose=True)
    results = evaluator.evaluate_all(hypotheses, references)

    return results


def demo_multiple_references():
    """Demo with multiple reference translations"""

    print("\n" + "="*80)
    print("DEMO 4: Multiple Reference Translations")
    print("="*80)
    print("Note: Multiple references improve BLEU scores (more ways to be correct)")
    print()

    hypotheses = [
        "मैं घर जा रहा हूं",
        "यह किताब अच्छी है"
    ]

    # Multiple references per hypothesis
    references = [
        [
            "मैं घर जा रहा हूँ",
            "मैं अपने घर जा रहा हूँ",
            "मैं घर की ओर जा रहा हूँ"
        ],
        [
            "यह एक अच्छी किताब है",
            "यह किताब बहुत अच्छी है",
            "यह बहुत अच्छी पुस्तक है"
        ]
    ]

    evaluator = MTEvaluator(verbose=True)
    results = evaluator.calculate_bleu(hypotheses, references)

    return results


def demo_individual_metrics():
    """Demo calculating metrics individually"""

    print("\n" + "="*80)
    print("DEMO 5: Individual Metric Calculation")
    print("="*80)

    hypotheses = ["मैं घर जा रहा हूं"]
    references = ["मैं घर जा रहा हूँ"]

    evaluator = MTEvaluator(verbose=False)

    print("\n--- BLEU Only ---")
    bleu = evaluator.calculate_bleu(hypotheses, references)
    print(f"BLEU: {bleu['bleu']:.2f}")

    print("\n--- Character F2 Only ---")
    char_f2 = evaluator.calculate_char_f2(hypotheses, references)
    print(f"Char F2: {char_f2['char_f2']:.2f}%")
    print(f"Char Precision: {char_f2['char_precision']:.2f}%")
    print(f"Char Recall: {char_f2['char_recall']:.2f}%")

    print("\n--- WER Only ---")
    wer = evaluator.calculate_wer(hypotheses, references)
    print(f"WER: {wer['wer']:.2f}%")
    print(f"Avg Ref Length: {wer['avg_ref_length']:.1f} words")
    print(f"Avg Hyp Length: {wer['avg_hyp_length']:.1f} words")


def main():
    """Run all demos"""

    print("\n" + "="*80)
    print("MT EVALUATION METRICS - INTERACTIVE DEMOS")
    print("="*80)
    print("\nThis script demonstrates all three evaluation metrics:")
    print("  1. BLEU Score (SacreBLEU)")
    print("  2. Character F2 Score")
    print("  3. Word Error Rate (WER)")
    print()
    input("Press Enter to start demos...")

    # Run demos
    demo_basic_evaluation()
    input("\nPress Enter for next demo...")

    demo_perfect_match()
    input("\nPress Enter for next demo...")

    demo_poor_match()
    input("\nPress Enter for next demo...")

    demo_multiple_references()
    input("\nPress Enter for next demo...")

    demo_individual_metrics()

    print("\n" + "="*80)
    print("DEMOS COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  - See EVALUATION_GUIDE.md for full documentation")
    print("  - Run: python evaluate_metrics.py --help")
    print("  - Evaluate your pipeline: python evaluate_metrics.py --evaluate-pipeline \\")
    print("      --source data/test.en --reference data/test.hi")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluation Metrics for English-Hindi MT Pipeline
Calculates: BLEU Score, Character F2 Score, Word Error Rate (WER)
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import metrics libraries
try:
    import sacrebleu
    from sacrebleu import BLEU
except ImportError:
    print("Installing sacrebleu...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sacrebleu>=2.3.0"])
    import sacrebleu
    from sacrebleu import BLEU

import numpy as np


class MTEvaluator:
    """Comprehensive MT evaluation with multiple metrics"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.bleu = BLEU(effective_order=True)  # Latest BLEU implementation

    def calculate_bleu(self, hypotheses: List[str], references: List[List[str]]) -> Dict:
        """
        Calculate BLEU score using SacreBLEU (recommended latest standard)

        Args:
            hypotheses: List of translation hypotheses
            references: List of reference translations (can be multiple refs per hypothesis)

        Returns:
            Dictionary with BLEU scores and details
        """
        # SacreBLEU expects references as list of lists (one list per reference set)
        # If single reference per hypothesis, transpose the structure
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        # Transpose references for sacrebleu format
        # From: [[ref1_hyp1, ref2_hyp1], [ref1_hyp2, ref2_hyp2]]
        # To: [[ref1_hyp1, ref1_hyp2], [ref2_hyp1, ref2_hyp2]]
        if len(references) > 0 and isinstance(references[0], list):
            num_refs = len(references[0])
            transposed_refs = []
            for ref_idx in range(num_refs):
                transposed_refs.append([refs[ref_idx] for refs in references])
            references = transposed_refs

        # Calculate BLEU
        bleu_score = self.bleu.corpus_score(hypotheses, references)

        results = {
            'bleu': bleu_score.score,
            'bleu_precision': bleu_score.precisions,
            'bleu_bp': bleu_score.bp,  # Brevity penalty
            'bleu_ratio': bleu_score.sys_len / bleu_score.ref_len,
            'bleu_details': str(bleu_score)
        }

        if self.verbose:
            print("\n" + "="*70)
            print("BLEU SCORE (SacreBLEU)")
            print("="*70)
            print(f"BLEU Score: {bleu_score.score:.2f}")
            print(f"Precisions: {bleu_score.precisions}")
            print(f"Brevity Penalty: {bleu_score.bp:.4f}")
            print(f"Length Ratio: {bleu_score.sys_len}/{bleu_score.ref_len} = {bleu_score.ratio:.4f}")
            print(f"Full: {bleu_score}")

        return results

    def calculate_char_f2(self, hypotheses: List[str], references: List[str]) -> Dict:
        """
        Calculate Character-level F2 Score
        F2 weighs recall higher than precision (beta=2)

        Args:
            hypotheses: List of translation hypotheses
            references: List of reference translations

        Returns:
            Dictionary with F2 score and related metrics
        """
        total_precision = 0.0
        total_recall = 0.0
        total_f2 = 0.0
        n = len(hypotheses)

        for hyp, ref in zip(hypotheses, references):
            hyp_chars = set(hyp)
            ref_chars = set(ref)

            if len(hyp_chars) == 0 and len(ref_chars) == 0:
                precision = recall = f2 = 1.0
            elif len(hyp_chars) == 0 or len(ref_chars) == 0:
                precision = recall = f2 = 0.0
            else:
                # Character-level metrics
                true_positives = len(hyp_chars & ref_chars)

                precision = true_positives / len(hyp_chars) if len(hyp_chars) > 0 else 0
                recall = true_positives / len(ref_chars) if len(ref_chars) > 0 else 0

                # F2 score (beta=2, weighs recall higher)
                beta = 2
                if precision + recall > 0:
                    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
                else:
                    f2 = 0.0

            total_precision += precision
            total_recall += recall
            total_f2 += f2

        avg_precision = total_precision / n
        avg_recall = total_recall / n
        avg_f2 = total_f2 / n

        results = {
            'char_f2': avg_f2 * 100,  # Convert to percentage
            'char_precision': avg_precision * 100,
            'char_recall': avg_recall * 100
        }

        if self.verbose:
            print("\n" + "="*70)
            print("CHARACTER F2 SCORE")
            print("="*70)
            print(f"Character F2 Score: {avg_f2*100:.2f}%")
            print(f"Character Precision: {avg_precision*100:.2f}%")
            print(f"Character Recall: {avg_recall*100:.2f}%")
            print(f"(F2 score weighs recall 2x more than precision)")

        return results

    def calculate_wer(self, hypotheses: List[str], references: List[str]) -> Dict:
        """
        Calculate Word Error Rate (WER)
        WER = (Substitutions + Deletions + Insertions) / Total Words in Reference

        Args:
            hypotheses: List of translation hypotheses
            references: List of reference translations

        Returns:
            Dictionary with WER and related metrics
        """
        total_wer = 0.0
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_ref_words = 0
        n = len(hypotheses)

        for hyp, ref in zip(hypotheses, references):
            hyp_words = hyp.split()
            ref_words = ref.split()

            # Calculate edit distance using dynamic programming
            m, n_words = len(ref_words), len(hyp_words)
            dp = np.zeros((m + 1, n_words + 1))

            # Initialize base cases
            for i in range(m + 1):
                dp[i][0] = i  # Deletions
            for j in range(n_words + 1):
                dp[0][j] = j  # Insertions

            # Fill DP table
            for i in range(1, m + 1):
                for j in range(1, n_words + 1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        dp[i][j] = dp[i-1][j-1]  # No operation needed
                    else:
                        dp[i][j] = 1 + min(
                            dp[i-1][j],      # Deletion
                            dp[i][j-1],      # Insertion
                            dp[i-1][j-1]     # Substitution
                        )

            edit_distance = dp[m][n_words]

            # Calculate WER for this sentence
            if m > 0:
                wer = edit_distance / m
            else:
                wer = 0.0 if n_words == 0 else float('inf')

            total_wer += wer
            total_ref_words += m

            # Approximate operation counts (simplified)
            total_substitutions += min(m, n_words)
            total_deletions += max(0, m - n_words)
            total_insertions += max(0, n_words - m)

        avg_wer = total_wer / len(hypotheses)

        results = {
            'wer': avg_wer * 100,  # Convert to percentage
            'total_operations': total_substitutions + total_deletions + total_insertions,
            'total_ref_words': total_ref_words,
            'avg_ref_length': total_ref_words / len(hypotheses),
            'avg_hyp_length': sum(len(h.split()) for h in hypotheses) / len(hypotheses)
        }

        if self.verbose:
            print("\n" + "="*70)
            print("WORD ERROR RATE (WER)")
            print("="*70)
            print(f"Word Error Rate: {avg_wer*100:.2f}%")
            print(f"Average Reference Length: {results['avg_ref_length']:.1f} words")
            print(f"Average Hypothesis Length: {results['avg_hyp_length']:.1f} words")
            print(f"Lower WER is better (0% = perfect match)")

        return results

    def evaluate_all(self, hypotheses: List[str], references: List[str]) -> Dict:
        """
        Calculate all metrics at once

        Args:
            hypotheses: List of translation hypotheses
            references: List of reference translations (or list of lists for multiple refs)

        Returns:
            Dictionary with all metrics
        """
        if len(hypotheses) != len(references):
            raise ValueError(f"Number of hypotheses ({len(hypotheses)}) must match references ({len(references)})")

        # Handle multiple references per hypothesis
        single_refs = references
        if len(references) > 0 and isinstance(references[0], list):
            # Use first reference for char F2 and WER
            single_refs = [refs[0] for refs in references]

        results = {}
        results.update(self.calculate_bleu(hypotheses, references))
        results.update(self.calculate_char_f2(hypotheses, single_refs))
        results.update(self.calculate_wer(hypotheses, single_refs))

        if self.verbose:
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            print(f"BLEU Score:       {results['bleu']:.2f}")
            print(f"Character F2:     {results['char_f2']:.2f}%")
            print(f"Word Error Rate:  {results['wer']:.2f}%")
            print("="*70 + "\n")

        return results


def evaluate_pipeline_on_dataset(pipeline, test_data_path: str, reference_path: str,
                                 stages=[1, 2, 3, 4]) -> Dict:
    """
    Evaluate the MT pipeline on a test dataset

    Args:
        pipeline: Initialized EnglishHindiMTPipeline instance
        test_data_path: Path to source language test file (English)
        reference_path: Path to reference translations (Hindi)
        stages: Pipeline stages to use

    Returns:
        Dictionary with all evaluation metrics
    """
    # Read test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        source_texts = [line.strip() for line in f if line.strip()]

    with open(reference_path, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f if line.strip()]

    if len(source_texts) != len(references):
        print(f"Warning: Source ({len(source_texts)}) and reference ({len(references)}) counts don't match")
        min_len = min(len(source_texts), len(references))
        source_texts = source_texts[:min_len]
        references = references[:min_len]

    print(f"\nEvaluating pipeline on {len(source_texts)} sentences...")
    print(f"Using stages: {stages}")

    # Generate translations
    hypotheses = []
    original_verbose = pipeline.verbose
    pipeline.verbose = False  # Suppress per-sentence output

    from tqdm import tqdm
    for source in tqdm(source_texts, desc="Translating"):
        translation = pipeline.translate(source, use_stages=stages)
        hypotheses.append(translation)

    pipeline.verbose = original_verbose

    # Evaluate
    evaluator = MTEvaluator(verbose=True)
    results = evaluator.evaluate_all(hypotheses, references)

    return results


def evaluate_from_files(hypothesis_file: str, reference_file: str) -> Dict:
    """
    Evaluate pre-generated translations

    Args:
        hypothesis_file: Path to hypothesis translations
        reference_file: Path to reference translations

    Returns:
        Dictionary with all evaluation metrics
    """
    with open(hypothesis_file, 'r', encoding='utf-8') as f:
        hypotheses = [line.strip() for line in f if line.strip()]

    with open(reference_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f if line.strip()]

    if len(hypotheses) != len(references):
        print(f"Warning: Hypothesis ({len(hypotheses)}) and reference ({len(references)}) counts don't match")
        min_len = min(len(hypotheses), len(references))
        hypotheses = hypotheses[:min_len]
        references = references[:min_len]

    print(f"\nEvaluating {len(hypotheses)} translations...")

    evaluator = MTEvaluator(verbose=True)
    results = evaluator.evaluate_all(hypotheses, references)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MT Evaluation Metrics")
    parser.add_argument("--hypothesis", type=str, help="Hypothesis translations file")
    parser.add_argument("--reference", type=str, help="Reference translations file")
    parser.add_argument("--source", type=str, help="Source text file (for pipeline evaluation)")
    parser.add_argument("--stages", nargs='+', type=int, default=[1, 2, 3, 4],
                       help="Pipeline stages to use (for pipeline evaluation)")
    parser.add_argument("--evaluate-pipeline", action="store_true",
                       help="Evaluate the pipeline directly (requires --source and --reference)")

    args = parser.parse_args()

    if args.evaluate_pipeline:
        # Evaluate pipeline directly
        if not args.source or not args.reference:
            print("Error: --source and --reference required for pipeline evaluation")
            sys.exit(1)

        # Import and initialize pipeline
        from pipeline import EnglishHindiMTPipeline
        print("Initializing pipeline...")
        pipeline = EnglishHindiMTPipeline(verbose=False)

        results = evaluate_pipeline_on_dataset(
            pipeline,
            args.source,
            args.reference,
            stages=args.stages
        )

    elif args.hypothesis and args.reference:
        # Evaluate pre-generated translations
        results = evaluate_from_files(args.hypothesis, args.reference)

    else:
        # Demo mode with example
        print("\n" + "="*70)
        print("MT EVALUATION DEMO")
        print("="*70)
        print("\nUsage:")
        print("  1. Evaluate pre-generated translations:")
        print("     python evaluate_metrics.py --hypothesis hyp.txt --reference ref.txt")
        print("\n  2. Evaluate pipeline directly:")
        print("     python evaluate_metrics.py --evaluate-pipeline --source src.txt --reference ref.txt")
        print("\n  3. With specific stages:")
        print("     python evaluate_metrics.py --evaluate-pipeline --source src.txt --reference ref.txt --stages 2 3 4")
        print("\n" + "="*70)

        # Run demo with example data
        print("\nRunning demo with example translations...\n")

        hypotheses = [
            "मैं घर जा रहा हूं",
            "यह एक अच्छी किताब है",
            "वह स्कूल में पढ़ता है"
        ]

        references = [
            "मैं घर जा रहा हूँ",
            "यह एक बहुत अच्छी किताब है",
            "वह स्कूल में पढ़ाई करता है"
        ]

        evaluator = MTEvaluator(verbose=True)
        results = evaluator.evaluate_all(hypotheses, references)


if __name__ == "__main__":
    main()

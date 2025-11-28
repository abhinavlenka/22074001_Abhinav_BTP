#!/usr/bin/env python3
"""
Large-Scale Evaluation for English-Hindi MT Pipeline
Optimized for evaluating 50k+ sentences with progress tracking and batch processing
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import numpy as np

from evaluate_metrics import MTEvaluator
from pipeline import EnglishHindiMTPipeline


class LargeScaleEvaluator:
    """Evaluator optimized for large datasets"""

    def __init__(self, pipeline, batch_size=100, save_translations=True):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.save_translations = save_translations
        self.evaluator = MTEvaluator(verbose=False)

    def translate_dataset(self, source_file, stages=[1, 2, 3, 4]):
        """
        Translate entire dataset with progress tracking

        Args:
            source_file: Path to source sentences
            stages: Pipeline stages to use

        Returns:
            List of translations
        """
        print(f"\nTranslating {source_file}")
        print(f"Using stages: {stages}")
        print()

        # Read source sentences
        with open(source_file, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f if line.strip()]

        total = len(sources)
        print(f"Total sentences: {total:,}")

        # Translate with progress bar
        translations = []
        start_time = time.time()

        # Suppress pipeline output
        original_verbose = self.pipeline.verbose
        self.pipeline.verbose = False

        with tqdm(total=total, desc="Translating", unit="sent") as pbar:
            for i, source in enumerate(sources):
                try:
                    translation = self.pipeline.translate(source, use_stages=stages)
                    translations.append(translation)
                except Exception as e:
                    print(f"\nError on sentence {i}: {e}")
                    translations.append("")  # Add empty translation on error

                pbar.update(1)

                # Update ETA every 100 sentences
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = total - (i + 1)
                    eta = remaining / rate if rate > 0 else 0
                    pbar.set_postfix({
                        'rate': f'{rate:.1f} sent/s',
                        'eta': f'{timedelta(seconds=int(eta))}'
                    })

        self.pipeline.verbose = original_verbose

        elapsed = time.time() - start_time
        rate = total / elapsed
        print(f"\n✓ Translation complete!")
        print(f"  Time: {timedelta(seconds=int(elapsed))}")
        print(f"  Rate: {rate:.2f} sentences/second")

        return translations

    def evaluate_large_dataset(self, source_file, reference_file, stages=[1, 2, 3, 4],
                               output_dir="evaluation_results"):
        """
        Complete evaluation pipeline for large datasets

        Args:
            source_file: Path to source sentences
            reference_file: Path to reference translations
            stages: Pipeline stages to use
            output_dir: Directory to save results

        Returns:
            Dictionary with all metrics and metadata
        """
        source_file = Path(source_file)
        reference_file = Path(reference_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"eval_{source_file.stem}_{timestamp}"
        run_dir = output_dir / run_name
        run_dir.mkdir(exist_ok=True)

        print("="*80)
        print("LARGE-SCALE MT EVALUATION")
        print("="*80)
        print(f"Source:    {source_file}")
        print(f"Reference: {reference_file}")
        print(f"Stages:    {stages}")
        print(f"Output:    {run_dir}")
        print("="*80)

        # Read reference
        print("\nLoading reference translations...")
        with open(reference_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip()]

        # Translate
        print("\n" + "="*80)
        print("STEP 1: Translation")
        print("="*80)
        translations = self.translate_dataset(source_file, stages)

        # Save translations
        if self.save_translations:
            trans_file = run_dir / "translations.txt"
            print(f"\nSaving translations to {trans_file}")
            with open(trans_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(translations))

        # Evaluate
        print("\n" + "="*80)
        print("STEP 2: Metric Calculation")
        print("="*80)

        # Ensure same length
        min_len = min(len(translations), len(references))
        if len(translations) != len(references):
            print(f"Warning: Length mismatch - using first {min_len} sentences")
            translations = translations[:min_len]
            references = references[:min_len]

        print(f"\nCalculating metrics for {len(translations):,} sentence pairs...")
        print()

        # Calculate metrics with progress
        print("Calculating BLEU score...")
        bleu_results = self.evaluator.calculate_bleu(translations, references)

        print("Calculating Character F2 score...")
        char_results = self.evaluator.calculate_char_f2(translations, references)

        print("Calculating Word Error Rate...")
        wer_results = self.evaluator.calculate_wer(translations, references)

        # Combine results
        results = {
            'metadata': {
                'source_file': str(source_file),
                'reference_file': str(reference_file),
                'num_sentences': len(translations),
                'stages': stages,
                'timestamp': timestamp,
                'date': datetime.now().isoformat()
            },
            'metrics': {
                'bleu': bleu_results['bleu'],
                'bleu_precisions': bleu_results['bleu_precision'],
                'bleu_bp': bleu_results['bleu_bp'],
                'bleu_ratio': bleu_results['bleu_ratio'],
                'char_f2': char_results['char_f2'],
                'char_precision': char_results['char_precision'],
                'char_recall': char_results['char_recall'],
                'wer': wer_results['wer'],
                'avg_ref_length': wer_results['avg_ref_length'],
                'avg_hyp_length': wer_results['avg_hyp_length']
            }
        }

        # Save results
        results_file = run_dir / "results.json"
        print(f"\nSaving results to {results_file}")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        # Create human-readable report
        report_file = run_dir / "report.txt"
        self._generate_report(results, report_file)

        # Display results
        self._display_results(results)

        print(f"\n✓ Evaluation complete! Results saved to: {run_dir}")

        return results

    def _generate_report(self, results, report_file):
        """Generate human-readable report"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ENGLISH-HINDI MT EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            # Metadata
            meta = results['metadata']
            f.write("Evaluation Details:\n")
            f.write(f"  Date:          {meta['date']}\n")
            f.write(f"  Source File:   {meta['source_file']}\n")
            f.write(f"  Reference:     {meta['reference_file']}\n")
            f.write(f"  Sentences:     {meta['num_sentences']:,}\n")
            f.write(f"  Stages Used:   {meta['stages']}\n")
            f.write("\n")

            # Metrics
            m = results['metrics']
            f.write("="*80 + "\n")
            f.write("EVALUATION METRICS\n")
            f.write("="*80 + "\n\n")

            f.write("BLEU Score (SacreBLEU):\n")
            f.write(f"  Overall Score:     {m['bleu']:.2f}\n")
            f.write(f"  1-gram Precision:  {m['bleu_precisions'][0]:.2f}\n")
            f.write(f"  2-gram Precision:  {m['bleu_precisions'][1]:.2f}\n")
            f.write(f"  3-gram Precision:  {m['bleu_precisions'][2]:.2f}\n")
            f.write(f"  4-gram Precision:  {m['bleu_precisions'][3]:.2f}\n")
            f.write(f"  Brevity Penalty:   {m['bleu_bp']:.4f}\n")
            f.write(f"  Length Ratio:      {m['bleu_ratio']:.4f}\n")
            f.write("\n")

            f.write("Character F2 Score:\n")
            f.write(f"  F2 Score:          {m['char_f2']:.2f}%\n")
            f.write(f"  Precision:         {m['char_precision']:.2f}%\n")
            f.write(f"  Recall:            {m['char_recall']:.2f}%\n")
            f.write("\n")

            f.write("Word Error Rate:\n")
            f.write(f"  WER:               {m['wer']:.2f}%\n")
            f.write(f"  Avg Ref Length:    {m['avg_ref_length']:.1f} words\n")
            f.write(f"  Avg Hyp Length:    {m['avg_hyp_length']:.1f} words\n")
            f.write("\n")

            # Interpretation
            f.write("="*80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("="*80 + "\n\n")

            bleu = m['bleu']
            if bleu >= 70:
                quality = "Exceptional to very good"
            elif bleu >= 50:
                quality = "Good"
            elif bleu >= 30:
                quality = "Understandable"
            else:
                quality = "Poor"

            f.write(f"Overall Translation Quality: {quality}\n")
            f.write(f"  BLEU {bleu:.2f} indicates {quality.lower()} translation quality.\n\n")

            char_f2 = m['char_f2']
            if char_f2 >= 90:
                char_quality = "Excellent"
            elif char_f2 >= 75:
                char_quality = "Good"
            elif char_f2 >= 60:
                char_quality = "Moderate"
            else:
                char_quality = "Poor"

            f.write(f"Character-Level Accuracy: {char_quality}\n")
            f.write(f"  Char F2 {char_f2:.2f}% indicates {char_quality.lower()} character coverage.\n\n")

            wer = m['wer']
            if wer <= 10:
                wer_quality = "Excellent"
            elif wer <= 20:
                wer_quality = "Good"
            elif wer <= 40:
                wer_quality = "Moderate"
            elif wer <= 60:
                wer_quality = "Poor"
            else:
                wer_quality = "Very poor"

            f.write(f"Word-Level Accuracy: {wer_quality}\n")
            f.write(f"  WER {wer:.2f}% indicates {wer_quality.lower()} word-level precision.\n\n")

        print(f"✓ Report saved to {report_file}")

    def _display_results(self, results):
        """Display results in terminal"""
        m = results['metrics']

        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"\nDataset: {results['metadata']['num_sentences']:,} sentences")
        print(f"Stages:  {results['metadata']['stages']}")
        print()
        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│                        METRIC SUMMARY                            │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print(f"│  BLEU Score:        {m['bleu']:>6.2f}  / 100                            │")
        print(f"│  Character F2:      {m['char_f2']:>6.2f}% (Precision: {m['char_precision']:>5.2f}%)         │")
        print(f"│  Word Error Rate:   {m['wer']:>6.2f}% (Lower is better)              │")
        print("└─────────────────────────────────────────────────────────────────┘")
        print()
        print("Detailed BLEU Breakdown:")
        print(f"  1-gram: {m['bleu_precisions'][0]:>5.2f}  2-gram: {m['bleu_precisions'][1]:>5.2f}  " +
              f"3-gram: {m['bleu_precisions'][2]:>5.2f}  4-gram: {m['bleu_precisions'][3]:>5.2f}")
        print(f"  Brevity Penalty: {m['bleu_bp']:.4f}  Length Ratio: {m['bleu_ratio']:.4f}")
        print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Large-scale evaluation of English-Hindi MT pipeline"
    )
    parser.add_argument("--source", default="data/iitb_test_50k.en",
                       help="Source file (English)")
    parser.add_argument("--reference", default="data/iitb_test_50k.hi",
                       help="Reference file (Hindi)")
    parser.add_argument("--stages", nargs='+', type=int, default=[1, 2, 3, 4],
                       help="Pipeline stages to use")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing")
    parser.add_argument("--no-save-translations", action="store_true",
                       help="Don't save translation outputs")
    parser.add_argument("--output-dir", default="evaluation_results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Check files exist
    if not Path(args.source).exists():
        print(f"Error: Source file not found: {args.source}")
        print("\nRun this first to download IIT Bombay corpus:")
        print("  python download_iitb_corpus.py")
        sys.exit(1)

    if not Path(args.reference).exists():
        print(f"Error: Reference file not found: {args.reference}")
        print("\nRun this first to download IIT Bombay corpus:")
        print("  python download_iitb_corpus.py")
        sys.exit(1)

    # Initialize pipeline
    print("Initializing MT pipeline...")
    pipeline = EnglishHindiMTPipeline(verbose=False)

    # Create evaluator
    evaluator = LargeScaleEvaluator(
        pipeline,
        batch_size=args.batch_size,
        save_translations=not args.no_save_translations
    )

    # Run evaluation
    try:
        results = evaluator.evaluate_large_dataset(
            args.source,
            args.reference,
            stages=args.stages,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

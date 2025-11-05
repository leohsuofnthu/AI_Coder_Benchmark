"""
AI Coder Evaluation Script
Compares model-generated XML output against benchmark (ground truth) XML
for multi-label code assignment accuracy.

Author: AI Coder Metrics Team
Date: November 2025
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import json
from pathlib import Path


class CodebookEvaluator:
    """Evaluates AI model coding accuracy against human benchmark."""
    
    def __init__(self, benchmark_path: str, model_output_path: str):
        """
        Initialize evaluator with paths to XML files.
        
        Args:
            benchmark_path: Path to ground truth XML
            model_output_path: Path to model-generated XML
        """
        self.benchmark_path = benchmark_path
        self.model_output_path = model_output_path
        
        # Data structures
        self.codebook = {}  # CBCKey -> code details
        self.benchmark_responses = {}  # (respondent_id, question_id) -> set of CBCKeys
        self.model_responses = {}  # (respondent_id, question_id) -> set of CBCKeys
        
        # Metrics storage
        self.metrics = {}
        
    def parse_xml(self, xml_path: str) -> Tuple[Dict, Dict]:
        """
        Parse XML file and extract codebook and responses.
        
        Returns:
            Tuple of (codebook_dict, responses_dict)
        """
        print(f"Parsing {Path(xml_path).name}...")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        codebook = {}
        responses = defaultdict(set)
        
        # Parse CodeBook
        for codebook_elem in root.findall('.//CodeBook'):
            for code in codebook_elem.findall('.//CodeBookCode'):
                cbc_key = code.find('CBCKey')
                if cbc_key is not None:
                    key = cbc_key.text
                    codebook[key] = {
                        'key': key,
                        'description': self._get_text(code, 'CBCDescription'),
                        'depth': int(self._get_text(code, 'CBCDepth', '0')),
                        'is_net': self._get_text(code, 'CBCIsNet') == 'True',
                        'output_id': self._get_text(code, 'CBCOutputID'),
                        'code_id': self._get_text(code, 'CBCCodeID'),
                    }
        
        # Parse Questions and Responses
        for question in root.findall('.//Question'):
            question_id = self._get_text(question, 'QuestionID')
            
            for response in question.findall('.//Response'):
                respondent_id = self._get_text(response, 'DRORespondent')
                verbatim = self._get_text(response, 'DROVerbatim')
                
                # Extract assigned codes
                codes = set()
                for resp_code in response.findall('.//ResponseCode'):
                    cbc_key = self._get_text(resp_code, 'DCCBCKey')
                    if cbc_key:
                        codes.add(cbc_key)
                
                # Store with unique key
                key = (respondent_id, question_id)
                responses[key] = codes
        
        print(f"  Found {len(codebook)} codes and {len(responses)} responses")
        return codebook, responses
    
    def _get_text(self, element, tag, default=''):
        """Safely extract text from XML element."""
        child = element.find(tag)
        return child.text if child is not None and child.text else default
    
    def _validate_codebook(self, benchmark_codebook: Dict, model_codebook: Dict):
        """
        Validate that model output has the same codebook architecture as benchmark.
        
        Args:
            benchmark_codebook: Codebook from benchmark XML
            model_codebook: Codebook from model output XML
            
        Raises:
            ValueError: If codebooks don't match
        """
        print("\n🔍 Validating codebook architecture...")
        
        # Check if both have codebooks
        if not benchmark_codebook:
            raise ValueError("Benchmark XML has no codebook!")
        
        if not model_codebook:
            raise ValueError("Model output XML has no codebook!")
        
        # Check number of codes
        if len(benchmark_codebook) != len(model_codebook):
            raise ValueError(
                f"❌ Codebook size mismatch!\n"
                f"   Benchmark has {len(benchmark_codebook)} codes\n"
                f"   Model output has {len(model_codebook)} codes\n"
                f"   The codebook architecture must be identical."
            )
        
        # Check that all benchmark codes exist in model
        benchmark_keys = set(benchmark_codebook.keys())
        model_keys = set(model_codebook.keys())
        
        missing_in_model = benchmark_keys - model_keys
        extra_in_model = model_keys - benchmark_keys
        
        if missing_in_model:
            missing_codes = list(missing_in_model)[:5]  # Show first 5
            raise ValueError(
                f"❌ Codebook mismatch: Model output is missing {len(missing_in_model)} codes from benchmark!\n"
                f"   Examples of missing codes: {missing_codes}\n"
                f"   You must use the same codebook structure as the benchmark."
            )
        
        if extra_in_model:
            extra_codes = list(extra_in_model)[:5]  # Show first 5
            raise ValueError(
                f"❌ Codebook mismatch: Model output has {len(extra_in_model)} extra codes not in benchmark!\n"
                f"   Examples of extra codes: {extra_codes}\n"
                f"   You must use the same codebook structure as the benchmark."
            )
        
        # Check that code properties match (description, depth)
        mismatches = []
        for code_key in benchmark_keys:
            bench_code = benchmark_codebook[code_key]
            model_code = model_codebook[code_key]
            
            # Check description
            if bench_code.get('description') != model_code.get('description'):
                mismatches.append(
                    f"Code {code_key}: Description mismatch\n"
                    f"     Benchmark: '{bench_code.get('description')}'\n"
                    f"     Model: '{model_code.get('description')}'"
                )
            
            # Check depth (hierarchy level)
            if bench_code.get('depth') != model_code.get('depth'):
                mismatches.append(
                    f"Code {code_key}: Hierarchy depth mismatch\n"
                    f"     Benchmark: {bench_code.get('depth')}\n"
                    f"     Model: {model_code.get('depth')}"
                )
            
            # Check is_net flag
            if bench_code.get('is_net') != model_code.get('is_net'):
                mismatches.append(
                    f"Code {code_key}: Net code flag mismatch\n"
                    f"     Benchmark: {bench_code.get('is_net')}\n"
                    f"     Model: {model_code.get('is_net')}"
                )
        
        if mismatches:
            # Show first 3 mismatches
            error_details = "\n   ".join(mismatches[:3])
            more_text = f"\n   ... and {len(mismatches) - 3} more mismatches" if len(mismatches) > 3 else ""
            raise ValueError(
                f"❌ Codebook architecture mismatch: {len(mismatches)} code(s) have different properties!\n\n"
                f"   {error_details}{more_text}\n\n"
                f"   The codebook must be identical to the benchmark.\n"
                f"   Copy the entire <CodeBooks> section from the benchmark XML."
            )
        
        print("✓ Codebook validation passed - architectures match!")
    
    def load_data(self):
        """Load both benchmark and model output data."""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load benchmark (ground truth)
        self.codebook, self.benchmark_responses = self.parse_xml(self.benchmark_path)
        
        # Load model output
        model_codebook, self.model_responses = self.parse_xml(self.model_output_path)
        
        # Validate codebook architecture matches
        self._validate_codebook(self.codebook, model_codebook)
        
        print(f"\nCodebook contains {len(self.codebook)} unique codes")
        print(f"Benchmark contains {len(self.benchmark_responses)} responses")
        print(f"Model output contains {len(self.model_responses)} responses")
        
    def align_responses(self) -> List[Tuple[str, Set[str], Set[str]]]:
        """
        Align benchmark and model responses by (respondent_id, question_id).
        
        Returns:
            List of (key, ground_truth_codes, predicted_codes)
        """
        aligned = []
        
        benchmark_keys = set(self.benchmark_responses.keys())
        model_keys = set(self.model_responses.keys())
        
        # Find common responses
        common_keys = benchmark_keys & model_keys
        
        # Report mismatches
        missing_in_model = benchmark_keys - model_keys
        extra_in_model = model_keys - benchmark_keys
        
        if missing_in_model:
            print(f"\n⚠️  WARNING: {len(missing_in_model)} responses in benchmark but not in model output")
        if extra_in_model:
            print(f"⚠️  WARNING: {len(extra_in_model)} responses in model output but not in benchmark")
        
        # Align common responses
        for key in common_keys:
            ground_truth = self.benchmark_responses[key]
            predicted = self.model_responses[key]
            aligned.append((key, ground_truth, predicted))
        
        print(f"\n✓ Aligned {len(aligned)} responses for evaluation")
        return aligned
    
    def calculate_metrics(self, aligned_data: List[Tuple[str, Set[str], Set[str]]]):
        """Calculate comprehensive evaluation metrics."""
        print("\n" + "="*70)
        print("CALCULATING METRICS")
        print("="*70)
        
        # Initialize counters
        total_tp = 0  # True Positives (across all responses)
        total_fp = 0  # False Positives
        total_fn = 0  # False Negatives
        
        exact_matches = 0
        jaccard_scores = []
        
        # Per-response metrics
        response_metrics = []
        
        # Mismatched responses (for detailed analysis)
        mismatches = []
        
        # Per-code metrics
        code_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0})
        
        for key, ground_truth, predicted in aligned_data:
            # Response-level calculations
            tp = len(ground_truth & predicted)  # Intersection
            fp = len(predicted - ground_truth)  # Predicted but not actual
            fn = len(ground_truth - predicted)  # Actual but not predicted
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Exact match
            if ground_truth == predicted:
                exact_matches += 1
            
            # Jaccard similarity (IoU)
            union = len(ground_truth | predicted)
            jaccard = tp / union if union > 0 else 1.0  # 1.0 if both empty
            jaccard_scores.append(jaccard)
            
            # Per-code statistics
            for code in ground_truth:
                code_stats[code]['support'] += 1
                if code in predicted:
                    code_stats[code]['tp'] += 1
                else:
                    code_stats[code]['fn'] += 1
            
            for code in predicted:
                if code not in ground_truth:
                    code_stats[code]['fp'] += 1
            
            # Store response metrics
            response_metrics.append({
                'key': key,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'jaccard': jaccard,
                'ground_truth_count': len(ground_truth),
                'predicted_count': len(predicted),
            })
            
            # Track mismatches (responses that aren't perfect)
            if ground_truth != predicted:
                missed_codes = ground_truth - predicted  # False negatives
                extra_codes = predicted - ground_truth   # False positives
                
                mismatches.append({
                    'respondent_id': key[0],
                    'question_id': key[1],
                    'ground_truth': list(ground_truth),
                    'predicted': list(predicted),
                    'missed_codes': list(missed_codes),
                    'extra_codes': list(extra_codes),
                    'jaccard': jaccard,
                    'error_count': fp + fn
                })
        
        # Calculate overall metrics
        n_responses = len(aligned_data)
        
        # Precision, Recall, F1 (Micro - aggregate all TP/FP/FN)
        precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) \
                   if (precision_micro + recall_micro) > 0 else 0
        
        # Macro-averaged (average per-code metrics)
        code_precisions = []
        code_recalls = []
        code_f1s = []
        
        for code, stats in code_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            code_precisions.append(prec)
            code_recalls.append(rec)
            code_f1s.append(f1)
        
        precision_macro = sum(code_precisions) / len(code_precisions) if code_precisions else 0
        recall_macro = sum(code_recalls) / len(code_recalls) if code_recalls else 0
        f1_macro = sum(code_f1s) / len(code_f1s) if code_f1s else 0
        
        # Sort mismatches by error count (worst first)
        mismatches.sort(key=lambda x: x['error_count'], reverse=True)
        
        # Store metrics
        self.metrics = {
            'overall': {
                'n_responses': n_responses,
                'n_mismatches': len(mismatches),
                'exact_match_ratio': exact_matches / n_responses if n_responses > 0 else 0,
                'jaccard_mean': sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0,
                'jaccard_median': sorted(jaccard_scores)[len(jaccard_scores) // 2] if jaccard_scores else 0,
                'precision_micro': precision_micro,
                'recall_micro': recall_micro,
                'f1_micro': f1_micro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
            },
            'per_code': code_stats,
            'per_response': response_metrics,
            'mismatches': mismatches,
        }
        
        return self.metrics
    
    def print_summary(self):
        """Print evaluation summary to console."""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        overall = self.metrics['overall']
        
        print(f"\n📊 OVERALL METRICS (n={overall['n_responses']} responses)")
        print("-" * 70)
        print(f"Exact Match Ratio:      {overall['exact_match_ratio']:.2%}")
        print(f"Jaccard Similarity:     {overall['jaccard_mean']:.4f} (mean), {overall['jaccard_median']:.4f} (median)")
        print()
        print(f"Precision (Micro):      {overall['precision_micro']:.4f}")
        print(f"Recall (Micro):         {overall['recall_micro']:.4f}")
        print(f"F1-Score (Micro):       {overall['f1_micro']:.4f}")
        print()
        print(f"Precision (Macro):      {overall['precision_macro']:.4f}")
        print(f"Recall (Macro):         {overall['recall_macro']:.4f}")
        print(f"F1-Score (Macro):       {overall['f1_macro']:.4f}")
        print()
        print(f"True Positives:         {overall['total_tp']}")
        print(f"False Positives:        {overall['total_fp']}")
        print(f"False Negatives:        {overall['total_fn']}")
        
        # Top performing codes
        print(f"\n🎯 TOP 10 BEST PERFORMING CODES (by F1-Score)")
        print("-" * 70)
        self._print_top_codes(top_n=10, ascending=False)
        
        # Worst performing codes
        print(f"\n⚠️  TOP 10 WORST PERFORMING CODES (by F1-Score)")
        print("-" * 70)
        self._print_top_codes(top_n=10, ascending=True)
        
    def _print_top_codes(self, top_n=10, ascending=True):
        """Print top/worst performing codes."""
        code_stats = self.metrics['per_code']
        
        # Calculate F1 for each code
        code_f1 = []
        for code, stats in code_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            support = stats['support']
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            code_info = self.codebook.get(code, {})
            code_f1.append((code, f1, prec, rec, support, code_info.get('description', 'Unknown')))
        
        # Sort
        code_f1.sort(key=lambda x: x[1], reverse=not ascending)
        
        # Print top N
        for i, (code, f1, prec, rec, support, desc) in enumerate(code_f1[:top_n], 1):
            print(f"{i:2d}. [{code}] {desc[:40]:40s} | F1: {f1:.3f} | P: {prec:.3f} | R: {rec:.3f} | n={support}")
    
    def print_mismatches(self, top_n=20, include_verbatims=False):
        """
        Print worst mismatched responses.
        
        Args:
            top_n: Number of worst mismatches to show
            include_verbatims: If True, fetch and display verbatim text (requires re-parsing XML)
        """
        if 'mismatches' not in self.metrics or not self.metrics['mismatches']:
            print("\n✓ No mismatches found - all predictions are perfect!")
            return
        
        mismatches = self.metrics['mismatches']
        
        print(f"\n" + "="*70)
        print(f"TOP {min(top_n, len(mismatches))} WORST MISMATCHES")
        print("="*70)
        print(f"\nTotal mismatches: {len(mismatches)} / {self.metrics['overall']['n_responses']} responses")
        print(f"Exact match rate: {self.metrics['overall']['exact_match_ratio']:.1%}\n")
        
        # Optionally load verbatims
        verbatims = {}
        if include_verbatims:
            print("Loading verbatim texts...")
            import xml.etree.ElementTree as ET
            tree = ET.parse(self.benchmark_path)
            root = tree.getroot()
            
            for response in root.findall('.//Response'):
                respondent = response.find('DRORespondent')
                verbatim = response.find('DROVerbatim')
                if respondent is not None and verbatim is not None:
                    verbatims[respondent.text] = verbatim.text or ""
        
        for i, mismatch in enumerate(mismatches[:top_n], 1):
            print(f"\n{i}. Respondent: {mismatch['respondent_id']} | Question: {mismatch['question_id']}")
            print(f"   Errors: {mismatch['error_count']} | Jaccard: {mismatch['jaccard']:.3f}")
            
            if include_verbatims and mismatch['respondent_id'] in verbatims:
                verbatim = verbatims[mismatch['respondent_id']]
                print(f"   Verbatim: \"{verbatim[:100]}{'...' if len(verbatim) > 100 else ''}\"")
            
            # Show missed codes (False Negatives)
            if mismatch['missed_codes']:
                print(f"\n   ❌ Missed {len(mismatch['missed_codes'])} code(s) (should have predicted):")
                for code in mismatch['missed_codes']:
                    desc = self.codebook.get(code, {}).get('description', 'Unknown')
                    print(f"      - [{code}] {desc}")
            
            # Show extra codes (False Positives)
            if mismatch['extra_codes']:
                print(f"\n   ⚠️  Incorrectly added {len(mismatch['extra_codes'])} code(s):")
                for code in mismatch['extra_codes']:
                    desc = self.codebook.get(code, {}).get('description', 'Unknown')
                    print(f"      - [{code}] {desc}")
            
            # Show correct codes (True Positives) 
            correct_codes = set(mismatch['predicted']) & set(mismatch['ground_truth'])
            if correct_codes:
                print(f"\n   ✓ Correctly predicted {len(correct_codes)} code(s)")
        
        print(f"\n" + "="*70)
    
    def save_detailed_report(self, output_path: str = "evaluation_report.json"):
        """Save detailed metrics to JSON file."""
        # Convert defaultdict to regular dict for JSON serialization
        report = {
            'overall': self.metrics['overall'],
            'per_code': {
                code: {
                    **stats,
                    'description': self.codebook.get(code, {}).get('description', 'Unknown'),
                    'depth': self.codebook.get(code, {}).get('depth', 0),
                }
                for code, stats in self.metrics['per_code'].items()
            },
            'codebook': self.codebook,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed report saved to: {output_path}")
    
    def run(self, save_report: bool = True, show_mismatches: int = 0, with_verbatims: bool = False):
        """Run complete evaluation pipeline.
        
        Args:
            save_report: Whether to save JSON report
            show_mismatches: Number of worst mismatches to display (0 = don't show)
            with_verbatims: Include verbatim text in mismatch display
        """
        try:
            # Load data
            self.load_data()
            
            # Align responses
            aligned_data = self.align_responses()
            
            if not aligned_data:
                print("\n❌ ERROR: No aligned responses found. Cannot evaluate.")
                return None
            
            # Calculate metrics
            self.calculate_metrics(aligned_data)
            
            # Print summary
            self.print_summary()
            
            # Show mismatches if requested
            if show_mismatches > 0:
                self.print_mismatches(top_n=show_mismatches, include_verbatims=with_verbatims)
            
            # Save detailed report
            if save_report:
                self.save_detailed_report()
            
            print("\n" + "="*70)
            print("✓ EVALUATION COMPLETE")
            print("="*70)
            
            return self.metrics
            
        except Exception as e:
            print(f"\n❌ ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate AI model coding accuracy against benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py benchmark.xml model_output.xml
  python evaluate_model.py Benchmark/Kids_Meal.xml ModelOutput/Kids_Meal_predicted.xml
        """
    )
    
    parser.add_argument('benchmark', help='Path to benchmark (ground truth) XML file')
    parser.add_argument('model_output', help='Path to model-generated XML file')
    parser.add_argument('--output', '-o', default='evaluation_report.json',
                       help='Output path for detailed JSON report (default: evaluation_report.json)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save detailed report to file')
    parser.add_argument('--show-mismatches', '-m', type=int, metavar='N', default=0,
                       help='Show top N worst mismatched responses (default: 0 = don\'t show)')
    parser.add_argument('--with-verbatims', '-v', action='store_true',
                       help='Include verbatim text when showing mismatches (slower)')
    
    args = parser.parse_args()
    
    # Validate files exist
    for path in [args.benchmark, args.model_output]:
        if not Path(path).exists():
            print(f"❌ ERROR: File not found: {path}")
            return
    
    # Run evaluation
    evaluator = CodebookEvaluator(args.benchmark, args.model_output)
    evaluator.run(
        save_report=not args.no_save,
        show_mismatches=args.show_mismatches,
        with_verbatims=args.with_verbatims
    )


if __name__ == '__main__':
    main()


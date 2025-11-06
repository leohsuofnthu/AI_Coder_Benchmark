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
        self.codebook = {}  # CBCKey -> code details (from benchmark)
        self.benchmark_responses = {}  # (respondent_id, question_id) -> set of CBCKeys
        self.model_responses = {}  # (respondent_id, question_id) -> set of CBCKeys
        self.model_to_benchmark_code_map = {}  # model_code_key -> benchmark_code_key (by description)
        
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
                    description = self._get_text(code, 'CBCDescription')
                    
                    # Normalization: Filter out "Uncoded Segment" codes from codebook
                    # These are placeholder codes that should be treated as Uncoded
                    desc_norm = (description or '').strip().lower()
                    if "uncoded" in desc_norm and "segment" in desc_norm:
                        continue  # Skip this code - it's a placeholder
                    
                    codebook[key] = {
                        'key': key,
                        'description': description,
                        'depth': int(self._get_text(code, 'CBCDepth', '0')),
                        'is_net': self._get_text(code, 'CBCIsNet') == 'True',
                        'output_id': self._get_text(code, 'CBCOutputID'),
                        'code_id': self._get_text(code, 'CBCCodeID'),
                    }
        
        # Parse Questions and Responses
        # Only include open-ended questions (QuestionType = 0)
        # QuestionType 0 = Open-ended text questions
        # QuestionType 1+ = Closed-ended questions (multiple choice, etc.)
        open_ended_count = 0
        skipped_questions = []
        skipped_count = 0
        
        for question in root.findall('.//Question'):
            question_type = self._get_text(question, 'QuestionType', '0')
            question_id = self._get_text(question, 'QuestionID')
            
            # Filter: Only process open-ended questions (QuestionType = 0)
            if question_type != '0':
                response_count = len(question.findall('.//Response'))
                skipped_count += response_count
                if question_id not in [q['id'] for q in skipped_questions]:
                    skipped_questions.append({
                        'id': question_id,
                        'type': question_type,
                        'count': response_count
                    })
                continue
            
            open_ended_count += 1
            
            for response in question.findall('.//Response'):
                respondent_id = self._get_text(response, 'DRORespondent')
                verbatim = self._get_text(response, 'DROVerbatim')
                
                # Extract assigned codes
                codes = set()
                for resp_code in response.findall('.//ResponseCode'):
                    cbc_key = self._get_text(resp_code, 'DCCBCKey')
                    if cbc_key:
                        codes.add(cbc_key)

                # Normalization: map any "Uncoded Segment"-style placeholder codes to Uncoded
                # We implement this by removing such placeholder codes from the set.
                # If a response only contains these placeholders, it will end up with an empty set
                # which downstream is treated as the Uncoded bucket.
                if codes:
                    filtered_codes = set()
                    for key in codes:
                        # Skip if code was filtered from codebook (e.g., "Uncoded Segment")
                        if key not in codebook:
                            continue
                        desc = codebook[key].get('description', '')
                        norm = (desc or '').strip().lower()
                        # match common variants: "uncoded segment", "uncoded segments", "uncoded segment category"
                        if not ("uncoded" in norm and "segment" in norm):
                            filtered_codes.add(key)
                    codes = filtered_codes
                
                # Store with unique key
                key = (respondent_id, question_id)
                responses[key] = codes
        
        if skipped_count > 0:
            print(f"  ⚠️  Filtered: {skipped_count} responses from {len(skipped_questions)} non-open-ended question(s) (QuestionType != 0)")
            if len(skipped_questions) <= 5:
                for q in skipped_questions:
                    print(f"     - Question '{q['id']}': Type {q['type']}, {q['count']} responses skipped")
        if open_ended_count > 0:
            print(f"  ✓ Processing {open_ended_count} open-ended question(s) (QuestionType = 0)")
        else:
            print(f"  ⚠️  Warning: No open-ended questions (QuestionType = 0) found in XML!")
        
        print(f"  Found {len(codebook)} codes and {len(responses)} responses")
        return codebook, responses
    
    def _get_text(self, element, tag, default=''):
        """Safely extract text from XML element."""
        child = element.find(tag)
        return child.text if child is not None and child.text else default
    
    def _validate_codebook_structure(self, benchmark_codebook: Dict, model_codebook: Dict):
        """
        Validate that codebooks have the same structure (number of codes and hierarchy).
        Assumes "Uncoded Segment" codes have already been filtered out.
        
        Args:
            benchmark_codebook: Codebook from benchmark XML
            model_codebook: Codebook from model output XML
            
        Raises:
            ValueError: If codebook structures don't match
        """
        print("\n🔍 Validating codebook structure...")
        
        # Check if both have codebooks
        if not benchmark_codebook:
            raise ValueError("Benchmark XML has no codebook!")
        
        if not model_codebook:
            raise ValueError("Model output XML has no codebook!")
        
        # Check code count
        bench_count = len(benchmark_codebook)
        model_count = len(model_codebook)
        
        if bench_count != model_count:
            raise ValueError(
                f"❌ Codebook structure mismatch: Different number of codes!\n"
                f"   Benchmark: {bench_count} codes\n"
                f"   Model output: {model_count} codes"
            )
        
        # Check hierarchy structure (depth distribution)
        bench_depths = {}
        model_depths = {}
        
        for code in benchmark_codebook.values():
            depth = code.get('depth', 1)
            bench_depths[depth] = bench_depths.get(depth, 0) + 1
        
        for code in model_codebook.values():
            depth = code.get('depth', 1)
            model_depths[depth] = model_depths.get(depth, 0) + 1
        
        if bench_depths != model_depths:
            raise ValueError(
                f"❌ Codebook hierarchy mismatch: Different depth distribution!\n"
                f"   Benchmark: {bench_depths}\n"
                f"   Model output: {model_depths}"
            )
        
        print(f"✓ Codebook structure validated - {bench_count} codes with matching hierarchy")
    
    def _build_code_mapping(self, benchmark_codebook: Dict, model_codebook: Dict) -> Dict[str, str]:
        """
        Build one-to-one mapping from model code IDs to benchmark code IDs.
        First tries ID-based mapping, then falls back to description+depth matching.
        
        Args:
            benchmark_codebook: Codebook from benchmark XML
            model_codebook: Codebook from model output XML
            
        Returns:
            Dictionary mapping model_code_key -> benchmark_code_key
        """
        mapping = {}
        
        # Step 1: Try direct ID matching (for same file or when IDs match)
        for model_key in model_codebook.keys():
            if model_key in benchmark_codebook:
                mapping[model_key] = model_key
        
        # Step 2: For unmapped codes, match by (description, depth)
        # Create lookup: (description, depth) -> list of benchmark_code_keys
        # Use list to handle potential duplicates (though structure validation should prevent this)
        desc_depth_to_benchmark = {}
        for bench_key, bench_code in benchmark_codebook.items():
            if bench_key not in mapping.values():  # Only for unmapped codes
                desc = (bench_code.get('description', '') or '').strip().lower()
                depth = bench_code.get('depth', 1)
                if desc:
                    key = (desc, depth)
                    if key not in desc_depth_to_benchmark:
                        desc_depth_to_benchmark[key] = []
                    desc_depth_to_benchmark[key].append(bench_key)
        
        # Map remaining model codes by description+depth
        for model_key in model_codebook.keys():
            if model_key not in mapping:  # Not yet mapped
                model_code = model_codebook[model_key]
                desc = (model_code.get('description', '') or '').strip().lower()
                depth = model_code.get('depth', 1)
                if desc:
                    key = (desc, depth)
                    if key in desc_depth_to_benchmark:
                        # Get list of candidate benchmark codes
                        candidates = desc_depth_to_benchmark[key]
                        # Find first unmapped candidate
                        bench_key = None
                        for candidate in candidates:
                            if candidate not in mapping.values():
                                bench_key = candidate
                                break
                        
                        if bench_key:
                            mapping[model_key] = bench_key
                            # Remove this candidate from list
                            candidates.remove(bench_key)
                            # If list is empty, remove the key
                            if not candidates:
                                del desc_depth_to_benchmark[key]
        
        return mapping
    
    def _validate_codebook(self, benchmark_codebook: Dict, model_codebook: Dict):
        """
        Validate that model output has the same codebook structure as benchmark.
        Validates structure (count and hierarchy) and builds mapping (ID-based, then description+depth).
        
        Args:
            benchmark_codebook: Codebook from benchmark XML
            model_codebook: Codebook from model output XML
            
        Raises:
            ValueError: If codebooks don't match
        """
        # Validate structure (count and hierarchy)
        self._validate_codebook_structure(benchmark_codebook, model_codebook)
        
        # Build mapping (ID-based first, then description+depth)
        mapping = self._build_code_mapping(benchmark_codebook, model_codebook)
        
        # Check that all codes are mapped
        unmapped_benchmark = set(benchmark_codebook.keys()) - set(mapping.values())
        unmapped_model = set(model_codebook.keys()) - set(mapping.keys())
        
        if unmapped_benchmark:
            # Show examples with descriptions for better debugging
            examples = []
            for code_id in list(unmapped_benchmark)[:5]:
                code = benchmark_codebook.get(code_id, {})
                desc = code.get('description', 'Unknown')
                examples.append(f"{code_id} ('{desc}')")
            
            raise ValueError(
                f"❌ Codebook mapping error: {len(unmapped_benchmark)} benchmark code(s) not found in model output!\n"
                f"   Examples: {', '.join(examples)}\n"
                f"   This usually means the codebook structure differs or codes have different descriptions/depths."
            )
        
        if unmapped_model:
            # Show examples with descriptions for better debugging
            examples = []
            for code_id in list(unmapped_model)[:5]:
                code = model_codebook.get(code_id, {})
                desc = code.get('description', 'Unknown')
                examples.append(f"{code_id} ('{desc}')")
            
            raise ValueError(
                f"❌ Codebook mapping error: {len(unmapped_model)} model code(s) not found in benchmark!\n"
                f"   Examples: {', '.join(examples)}\n"
                f"   This usually means the codebook structure differs or codes have different descriptions/depths."
            )
        
        # Count how many were mapped by ID vs description+depth
        id_mapped = sum(1 for m, b in mapping.items() if m == b)
        desc_mapped = len(mapping) - id_mapped
        
        if desc_mapped > 0:
            print(f"✓ Codebook mapping validated - {len(mapping)} codes mapped ({id_mapped} by ID, {desc_mapped} by description+depth)")
        else:
            print(f"✓ Codebook mapping validated - {len(mapping)} codes mapped by ID")
    
    def load_data(self):
        """Load both benchmark and model output data."""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load benchmark (ground truth)
        self.codebook, self.benchmark_responses = self.parse_xml(self.benchmark_path)
        
        # Load model output
        model_codebook, self.model_responses = self.parse_xml(self.model_output_path)
        
        # Check if comparing same file - use identity mapping for efficiency
        from pathlib import Path
        same_file = Path(self.benchmark_path).resolve() == Path(self.model_output_path).resolve()
        
        # Validate codebook structure first (before mapping)
        self._validate_codebook(self.codebook, model_codebook)
        
        # Build simple ID-based mapping (one-to-one by code ID)
        self.model_to_benchmark_code_map = self._build_code_mapping(self.codebook, model_codebook)
        
        # For same file, it's already identity mapping
        if same_file:
            print(f"\n✓ Same file detected - using identity mapping for {len(self.model_to_benchmark_code_map)} codes")
        else:
            print(f"\n✓ Code mapping built - {len(self.model_to_benchmark_code_map)} codes mapped by ID")
        
        print(f"\nCodebook contains {len(self.codebook)} unique codes")
        print(f"Benchmark contains {len(self.benchmark_responses)} responses")
        print(f"Model output contains {len(self.model_responses)} responses")
        print(f"Code mapping: {len(self.model_to_benchmark_code_map)} model codes mapped to benchmark codes")
        
        # Debug: Show mapping coverage for same file
        if same_file:
            all_model_codes_in_responses = set()
            for codes in self.model_responses.values():
                all_model_codes_in_responses.update(codes)
            unmapped_in_mapping = all_model_codes_in_responses - set(self.model_to_benchmark_code_map.keys())
            if unmapped_in_mapping:
                print(f"   ⚠️  WARNING: {len(unmapped_in_mapping)} codes in model responses are not in mapping!")
                print(f"   Unmapped codes: {list(unmapped_in_mapping)[:10]}")
        
    def align_responses(self) -> List[Tuple[str, Set[str], Set[str]]]:
        """
        Align benchmark and model responses by (respondent_id, question_id).
        Maps model codes to benchmark codes using ID-based mapping.
        
        Returns:
            List of (key, ground_truth_codes, predicted_codes)
            where predicted_codes are mapped to benchmark code keys
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
        
        # Align common responses and map model codes to benchmark codes
        same_file = Path(self.benchmark_path).resolve() == Path(self.model_output_path).resolve()
        unmapped_codes = set()
        mismatches = []  # Track mismatches for debugging
        
        for key in common_keys:
            ground_truth = self.benchmark_responses[key]
            model_codes = self.model_responses[key]
            
            # Map model codes to benchmark codes (works for both same file and different files)
            predicted = set()
            for model_code_key in model_codes:
                if model_code_key in self.model_to_benchmark_code_map:
                    mapped_code = self.model_to_benchmark_code_map[model_code_key]
                    predicted.add(mapped_code)
                else:
                    # Track unmapped codes (shouldn't happen if validation passed, but handle gracefully)
                    unmapped_codes.add(model_code_key)
                    # For same file, this is a critical error - codes should always be mapped
                    if same_file:
                        print(f"   ⚠️  CRITICAL: Code {model_code_key} in response {key} is not in identity mapping!")
            
            # Debug: Track mismatches for same file (shouldn't happen)
            if same_file and ground_truth != predicted:
                mismatches.append((key, ground_truth, predicted, model_codes))
            
            aligned.append((key, ground_truth, predicted))
        
        if unmapped_codes:
            print(f"\n⚠️  WARNING: {len(unmapped_codes)} model code(s) could not be mapped to benchmark codes")
            print(f"   Unmapped codes: {list(unmapped_codes)[:10]}")  # Show first 10 for debugging
            print(f"   This may indicate codes in responses that weren't in the codebook or mapping failed")
        
        if same_file and mismatches:
            print(f"\n⚠️  DEBUG: Found {len(mismatches)} mismatches when comparing same file (should be 0)")
            print(f"   First 3 mismatches:")
            for i, (key, gt, pred, mc) in enumerate(mismatches[:3]):
                print(f"   {i+1}. Key: {key}")
                print(f"      Benchmark: {gt}")
                print(f"      Model (before mapping): {mc}")
                print(f"      Model (after mapping): {pred}")
                print(f"      Missing: {gt - pred}, Extra: {pred - gt}")
        
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
        
        # Track uncoded responses
        uncoded_benchmark = 0  # Responses with no codes in benchmark
        uncoded_model = 0  # Responses with no codes in model output
        uncoded_both = 0  # Responses with no codes in both
        
        # Track uncoded responses by question
        question_uncoded_stats = defaultdict(lambda: {
            'benchmark_uncoded': 0,
            'model_uncoded': 0,
            'both_uncoded': 0,
            'total_responses': 0,
            'uncoded_responses': []  # List of (respondent_id, verbatim, status)
        })
        
        # Separate counters for coded responses only (ONLY when benchmark has codes)
        # This is the main evaluation metric - how well does model code responses that should have codes?
        coded_tp = 0
        coded_fp = 0
        coded_fn = 0
        coded_exact_matches = 0
        coded_jaccard_scores = []
        n_coded_responses = 0  # Responses where benchmark has codes (main evaluation set)
        
        # Track uncoded classification errors
        # When benchmark is uncoded but model assigned codes (False Positive for uncoded classification)
        incorrectly_coded_count = 0  # Benchmark uncoded but model coded (model should not have coded)
        incorrectly_coded_responses = []  # List of (respondent_id, question_id, assigned_codes)
        
        # Per-response metrics
        response_metrics = []
        
        # Mismatched responses (for detailed analysis)
        mismatches = []
        
        # Per-code metrics
        code_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0})
        
        for key, ground_truth, predicted in aligned_data:
            respondent_id, question_id = key
            
            # Track uncoded responses
            is_benchmark_uncoded = len(ground_truth) == 0
            is_model_uncoded = len(predicted) == 0
            
            # Update question-level statistics
            question_uncoded_stats[question_id]['total_responses'] += 1
            
            if is_benchmark_uncoded:
                uncoded_benchmark += 1
                question_uncoded_stats[question_id]['benchmark_uncoded'] += 1
            if is_model_uncoded:
                uncoded_model += 1
                question_uncoded_stats[question_id]['model_uncoded'] += 1
            if is_benchmark_uncoded and is_model_uncoded:
                uncoded_both += 1
                question_uncoded_stats[question_id]['both_uncoded'] += 1
                
                # Track this uncoded response
                question_uncoded_stats[question_id]['uncoded_responses'].append({
                    'respondent_id': respondent_id,
                    'status': 'both_uncoded'  # Correctly identified as uncoded
                })
            elif is_benchmark_uncoded and not is_model_uncoded:
                # Benchmark uncoded but model assigned codes (error)
                question_uncoded_stats[question_id]['uncoded_responses'].append({
                    'respondent_id': respondent_id,
                    'status': 'incorrectly_coded',  # Model incorrectly assigned codes
                    'assigned_codes': list(predicted)
                })
            elif not is_benchmark_uncoded and is_model_uncoded:
                # Model uncoded but benchmark has codes (missed codes)
                question_uncoded_stats[question_id]['uncoded_responses'].append({
                    'respondent_id': respondent_id,
                    'status': 'missed_codes',  # Model missed codes that should be assigned
                    'missed_codes': list(ground_truth)
                })
            
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
            
            # Separate metrics for coded responses only (ONLY when benchmark has codes)
            # This is the main evaluation - how well does model code responses that should have codes?
            if not is_benchmark_uncoded:  # Benchmark has codes - this is what we evaluate
                n_coded_responses += 1
                coded_tp += tp
                coded_fp += fp
                coded_fn += fn
                
                if ground_truth == predicted:
                    coded_exact_matches += 1
                
                # Jaccard for coded responses
                coded_jaccard = tp / union if union > 0 else 0.0
                coded_jaccard_scores.append(coded_jaccard)
            
            # Track uncoded classification errors
            # When benchmark is uncoded but model assigned codes (this is an error)
            if is_benchmark_uncoded and not is_model_uncoded:
                incorrectly_coded_count += 1
                incorrectly_coded_responses.append({
                    'respondent_id': respondent_id,
                    'question_id': question_id,
                    'assigned_codes': list(predicted),
                    'assigned_code_count': len(predicted)
                })
            
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
        
        # Support-weighted macro averaging (weight by code frequency)
        total_support = sum(stats['support'] for stats in code_stats.values())
        weighted_precision_macro = 0
        weighted_recall_macro = 0
        weighted_f1_macro = 0
        
        if total_support > 0:
            for code, stats in code_stats.items():
                tp = stats['tp']
                fp = stats['fp']
                fn = stats['fn']
                support = stats['support']
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                
                weight = support / total_support
                weighted_precision_macro += prec * weight
                weighted_recall_macro += rec * weight
                weighted_f1_macro += f1 * weight
        
        # Hamming Loss: average fraction of labels that are incorrectly predicted
        # Formula: (FP + FN) / (n_responses * n_codes)
        n_codes = len(code_stats)
        hamming_loss = (total_fp + total_fn) / (n_responses * n_codes) if (n_responses > 0 and n_codes > 0) else 0
        
        # Separate metrics for coded responses only
        coded_precision_micro = coded_tp / (coded_tp + coded_fp) if (coded_tp + coded_fp) > 0 else 0
        coded_recall_micro = coded_tp / (coded_tp + coded_fn) if (coded_tp + coded_fn) > 0 else 0
        coded_f1_micro = 2 * (coded_precision_micro * coded_recall_micro) / (coded_precision_micro + coded_recall_micro) \
                        if (coded_precision_micro + coded_recall_micro) > 0 else 0
        coded_exact_match_ratio = coded_exact_matches / n_coded_responses if n_coded_responses > 0 else 0
        coded_jaccard_mean = sum(coded_jaccard_scores) / len(coded_jaccard_scores) if coded_jaccard_scores else 0
        
        # Uncoded classification metrics (how well does model identify when nothing should be coded?)
        # This is separate from coding quality - it's about identifying when to NOT code
        uncoded_classification_tp = uncoded_both  # Both correctly uncoded (correctly identified as uncoded)
        uncoded_classification_fp = incorrectly_coded_count  # Benchmark uncoded but model coded (should NOT have coded)
        uncoded_classification_fn = uncoded_model - uncoded_both  # Model uncoded but benchmark has codes (missed codes - already counted in coded metrics)
        uncoded_classification_total = uncoded_benchmark  # Total cases where benchmark is uncoded
        
        # Accuracy: When benchmark is uncoded, did model also leave it uncoded?
        uncoded_classification_accuracy = (uncoded_classification_tp / uncoded_classification_total) if uncoded_classification_total > 0 else 0
        
        # Overlap: How many responses are uncoded in both (agreement)
        uncoded_overlap_count = uncoded_both
        uncoded_overlap_percentage = (uncoded_both / uncoded_benchmark * 100) if uncoded_benchmark > 0 else 0
        
        # Calculate question-level uncoded statistics
        question_uncoded_summary = []
        for question_id, stats in question_uncoded_stats.items():
            total = stats['total_responses']
            if total > 0:
                # Calculate overlap percentage
                overlap_pct = (stats['both_uncoded'] / total * 100) if total > 0 else 0
                benchmark_uncoded_pct = (stats['benchmark_uncoded'] / total * 100) if total > 0 else 0
                model_uncoded_pct = (stats['model_uncoded'] / total * 100) if total > 0 else 0
                
                # Classification accuracy for this question (when benchmark is uncoded, did model also leave uncoded?)
                question_uncoded_tp = stats['both_uncoded']
                question_uncoded_total = stats['benchmark_uncoded']
                question_uncoded_acc = (question_uncoded_tp / question_uncoded_total) if question_uncoded_total > 0 else 0
                
                question_uncoded_summary.append({
                    'question_id': question_id,
                    'total_responses': total,
                    'benchmark_uncoded': stats['benchmark_uncoded'],
                    'model_uncoded': stats['model_uncoded'],
                    'both_uncoded': stats['both_uncoded'],
                    'overlap_count': stats['both_uncoded'],
                    'overlap_percentage': overlap_pct,
                    'benchmark_uncoded_percentage': benchmark_uncoded_pct,
                    'model_uncoded_percentage': model_uncoded_pct,
                    'classification_accuracy': question_uncoded_acc,
                    'uncoded_responses': stats['uncoded_responses']
                })
        
        # Sort by total uncoded responses (descending)
        question_uncoded_summary.sort(key=lambda x: x['benchmark_uncoded'] + x['model_uncoded'], reverse=True)
        
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
                'precision_macro_weighted': weighted_precision_macro,
                'recall_macro_weighted': weighted_recall_macro,
                'f1_macro_weighted': weighted_f1_macro,
                'hamming_loss': hamming_loss,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'uncoded_benchmark': uncoded_benchmark,
                'uncoded_model': uncoded_model,
                'uncoded_both': uncoded_both,
                # Separate metrics for coded responses only
                'coded_only': {
                    'n_responses': n_coded_responses,
                    'exact_match_ratio': coded_exact_match_ratio,
                    'jaccard_mean': coded_jaccard_mean,
                    'precision_micro': coded_precision_micro,
                    'recall_micro': coded_recall_micro,
                    'f1_micro': coded_f1_micro,
                    'total_tp': coded_tp,
                    'total_fp': coded_fp,
                    'total_fn': coded_fn,
                },
                # Uncoded classification metrics (separate from coding quality)
                'uncoded_classification': {
                    'accuracy': uncoded_classification_accuracy,
                    'true_positives': uncoded_classification_tp,  # Both correctly uncoded
                    'false_positives': uncoded_classification_fp,  # Model coded when shouldn't (incorrectly_coded_count)
                    'false_negatives': uncoded_classification_fn,  # Model uncoded when should have codes (already in coded metrics)
                    'total_cases': uncoded_classification_total,  # Total benchmark uncoded
                    'overlap_count': uncoded_overlap_count,
                    'overlap_percentage': uncoded_overlap_percentage,
                    'incorrectly_coded_responses': incorrectly_coded_responses[:20]  # Top 20 for display
                },
                # Question-level uncoded breakdown
                'question_uncoded_breakdown': question_uncoded_summary,
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
        print(f"Precision (Macro-Weighted): {overall.get('precision_macro_weighted', 0):.4f}")
        print(f"Recall (Macro-Weighted):    {overall.get('recall_macro_weighted', 0):.4f}")
        print(f"F1-Score (Macro-Weighted):  {overall.get('f1_macro_weighted', 0):.4f}")
        print()
        print(f"Hamming Loss:           {overall.get('hamming_loss', 0):.4f}")
        print()
        print(f"True Positives:         {overall['total_tp']}")
        print(f"False Positives:        {overall['total_fp']}")
        print(f"False Negatives:        {overall['total_fn']}")
        
        # Main evaluation metrics (coded responses only - when benchmark has codes)
        if 'coded_only' in overall and overall['coded_only']['n_responses'] > 0:
            coded = overall['coded_only']
            print(f"\n📊 MAIN EVALUATION - CODED RESPONSES (n={coded['n_responses']})")
            print("   (Evaluating model performance when benchmark has codes)")
            print("-" * 70)
            print(f"Exact Match Ratio:      {coded['exact_match_ratio']:.2%}")
            print(f"Jaccard Similarity:     {coded['jaccard_mean']:.4f}")
            print(f"Precision (Micro):      {coded['precision_micro']:.4f}")
            print(f"Recall (Micro):         {coded['recall_micro']:.4f}")
            print(f"F1-Score (Micro):       {coded['f1_micro']:.4f}")
        
        # Uncoded responses statistics
        if 'uncoded_benchmark' in overall or 'uncoded_model' in overall:
            print(f"\n📋 UNCODED RESPONSES")
            print("-" * 70)
            uncoded_bench = overall.get('uncoded_benchmark', 0)
            uncoded_model = overall.get('uncoded_model', 0)
            uncoded_both = overall.get('uncoded_both', 0)
            
            if uncoded_bench > 0:
                print(f"Benchmark uncoded:      {uncoded_bench} ({uncoded_bench/overall['n_responses']*100:.1f}%)")
            if uncoded_model > 0:
                print(f"Model output uncoded:   {uncoded_model} ({uncoded_model/overall['n_responses']*100:.1f}%)")
            if uncoded_both > 0:
                print(f"Both uncoded (perfect): {uncoded_both} ({uncoded_both/overall['n_responses']*100:.1f}%)")
            
            # Uncoded classification metrics (separate evaluation)
            if 'uncoded_classification' in overall and overall['uncoded_classification']['total_cases'] > 0:
                uncoded_clf = overall['uncoded_classification']
                print(f"\n📋 UNCODED CLASSIFICATION PERFORMANCE")
                print("   (Separate evaluation: How well model identifies uncoded responses)")
                print("-" * 70)
                print(f"Total Benchmark Uncoded:    {uncoded_clf['total_cases']}")
                print(f"Both Uncoded (Correct):     {uncoded_clf['true_positives']} ({uncoded_clf['overlap_percentage']:.1f}% overlap)")
                print(f"Incorrectly Coded (Error):  {uncoded_clf['false_positives']} (Model coded when shouldn't)")
                print(f"Classification Accuracy:    {uncoded_clf['accuracy']:.2%}")
                print(f"  (When benchmark is uncoded, did model also leave it uncoded?)")
        
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
    
    def save_mismatches_report(self, output_path: str = "mismatches_report.csv"):
        """
        Save detailed mismatch report with verbatim text to CSV.
        
        Args:
            output_path: Path to save CSV file
        """
        if 'mismatches' not in self.metrics or not self.metrics['mismatches']:
            print("\n✓ No mismatches to export - all predictions are perfect!")
            return
        
        import csv
        import xml.etree.ElementTree as ET
        
        print(f"\n📊 Generating mismatches report...")
        
        # Load verbatims from benchmark
        verbatims = {}
        tree = ET.parse(self.benchmark_path)
        root = tree.getroot()
        
        for response in root.findall('.//Response'):
            respondent = response.find('DRORespondent')
            verbatim = response.find('DROVerbatim')
            if respondent is not None and verbatim is not None:
                verbatims[respondent.text] = verbatim.text or ""
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Rank',
                'Respondent ID',
                'Question ID',
                'Verbatim',
                'Error Count',
                'Jaccard Score',
                'Missed Codes (False Negatives)',
                'Missed Descriptions',
                'Extra Codes (False Positives)',
                'Extra Descriptions',
                'Correct Codes',
                'Ground Truth (All)',
                'Predicted (All)'
            ])
            
            # Data rows
            for rank, mismatch in enumerate(self.metrics['mismatches'], 1):
                respondent_id = mismatch['respondent_id']
                verbatim = verbatims.get(respondent_id, 'N/A')
                
                # Get code descriptions
                missed_descs = [
                    self.codebook.get(code, {}).get('description', 'Unknown')
                    for code in mismatch['missed_codes']
                ]
                extra_descs = [
                    self.codebook.get(code, {}).get('description', 'Unknown')
                    for code in mismatch['extra_codes']
                ]
                
                correct_codes = set(mismatch['predicted']) & set(mismatch['ground_truth'])
                
                writer.writerow([
                    rank,
                    respondent_id,
                    mismatch['question_id'],
                    verbatim,
                    mismatch['error_count'],
                    f"{mismatch['jaccard']:.3f}",
                    ', '.join(mismatch['missed_codes']),
                    ' | '.join(missed_descs),
                    ', '.join(mismatch['extra_codes']),
                    ' | '.join(extra_descs),
                    len(correct_codes),
                    ', '.join(mismatch['ground_truth']),
                    ', '.join(mismatch['predicted'])
                ])
        
        print(f"💾 Mismatches report saved to: {output_path}")
        print(f"   Total mismatches: {len(self.metrics['mismatches'])}")
    
    def run(self, save_report: bool = True, show_mismatches: int = 0, with_verbatims: bool = False, 
            export_mismatches: str = None):
        """Run complete evaluation pipeline.
        
        Args:
            save_report: Whether to save JSON report
            show_mismatches: Number of worst mismatches to display (0 = don't show)
            with_verbatims: Include verbatim text in mismatch display
            export_mismatches: Path to save mismatches CSV (None = don't export)
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
            
            # Export mismatches to CSV if requested
            if export_mismatches:
                self.save_mismatches_report(export_mismatches)
            
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
    parser.add_argument('--export-mismatches', '-e', metavar='FILE', default=None,
                       help='Export ALL mismatches with verbatims to CSV file (e.g., mismatches.csv)')
    
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
        with_verbatims=args.with_verbatims,
        export_mismatches=args.export_mismatches
    )


if __name__ == '__main__':
    main()


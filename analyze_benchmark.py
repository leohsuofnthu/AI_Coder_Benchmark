"""
Utility script to analyze benchmark XML files.
Provides statistics and insights about the data structure.

Usage:
    python analyze_benchmark.py Benchmark/Kids_Meal.xml
"""

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
import sys


def analyze_benchmark(xml_path: str):
    """Analyze benchmark XML and print statistics."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {Path(xml_path).name}")
    print(f"{'='*70}\n")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Study metadata
    study = root.find('.//Study')
    study_name = study.find('StudyName')
    study_desc = study.find('StudyDescription')
    
    print(f"📋 Study Information")
    print(f"-" * 70)
    if study_name is not None:
        print(f"Name: {study_name.text}")
    if study_desc is not None:
        print(f"Description: {study_desc.text}")
    
    # Analyze CodeBook
    print(f"\n📚 CodeBook Structure")
    print(f"-" * 70)
    
    codes_by_depth = defaultdict(int)
    net_codes = 0
    leaf_codes = 0
    codes_with_regex = 0
    
    all_codes = []
    for code in root.findall('.//CodeBookCode'):
        depth_elem = code.find('CBCDepth')
        is_net_elem = code.find('CBCIsNet')
        regex_elem = code.find('CBCRegExp')
        desc_elem = code.find('CBCDescription')
        key_elem = code.find('CBCKey')
        
        if depth_elem is not None:
            depth = int(depth_elem.text)
            codes_by_depth[depth] += 1
        
        if is_net_elem is not None and is_net_elem.text == 'True':
            net_codes += 1
        else:
            leaf_codes += 1
        
        if regex_elem is not None and regex_elem.text:
            codes_with_regex += 1
        
        all_codes.append({
            'key': key_elem.text if key_elem is not None else None,
            'description': desc_elem.text if desc_elem is not None else 'N/A',
            'depth': depth if depth_elem is not None else 0,
        })
    
    total_codes = len(all_codes)
    print(f"Total codes: {total_codes}")
    print(f"  - Net codes (summary): {net_codes}")
    print(f"  - Leaf codes (specific): {leaf_codes}")
    print(f"  - Codes with RegExp: {codes_with_regex}")
    print(f"\nCodes by hierarchy depth:")
    for depth in sorted(codes_by_depth.keys()):
        print(f"  Depth {depth}: {codes_by_depth[depth]} codes")
    
    # Analyze Questions
    print(f"\n❓ Questions")
    print(f"-" * 70)
    
    questions = root.findall('.//Question')
    print(f"Total questions: {len(questions)}")
    
    for i, question in enumerate(questions, 1):
        question_id_elem = question.find('QuestionID')
        question_label_elem = question.find('QuestionLabel')
        question_id = question_id_elem.text if question_id_elem is not None else f"Q{i}"
        question_label = question_label_elem.text if question_label_elem is not None else "N/A"
        
        responses = question.findall('.//Response')
        print(f"  {question_id}: {len(responses)} responses - \"{question_label}\"")
    
    # Analyze Responses
    print(f"\n💬 Response Analysis")
    print(f"-" * 70)
    
    all_responses = root.findall('.//Response')
    total_responses = len(all_responses)
    
    codes_per_response = []
    verbatim_lengths = []
    code_frequency = Counter()
    
    for response in all_responses:
        verbatim_elem = response.find('DROVerbatim')
        if verbatim_elem is not None and verbatim_elem.text:
            verbatim_lengths.append(len(verbatim_elem.text))
        
        response_codes = response.findall('.//ResponseCode')
        num_codes = len(response_codes)
        codes_per_response.append(num_codes)
        
        for resp_code in response_codes:
            cbc_key_elem = resp_code.find('DCCBCKey')
            if cbc_key_elem is not None:
                code_frequency[cbc_key_elem.text] += 1
    
    print(f"Total responses: {total_responses}")
    
    if codes_per_response:
        print(f"\nCodes per response:")
        print(f"  Mean: {sum(codes_per_response) / len(codes_per_response):.2f}")
        print(f"  Min: {min(codes_per_response)}")
        print(f"  Max: {max(codes_per_response)}")
        print(f"  Median: {sorted(codes_per_response)[len(codes_per_response) // 2]}")
    
    if verbatim_lengths:
        print(f"\nVerbatim text length (characters):")
        print(f"  Mean: {sum(verbatim_lengths) / len(verbatim_lengths):.1f}")
        print(f"  Min: {min(verbatim_lengths)}")
        print(f"  Max: {max(verbatim_lengths)}")
        print(f"  Median: {sorted(verbatim_lengths)[len(verbatim_lengths) // 2]}")
    
    # Most common codes
    print(f"\n🏆 Top 15 Most Frequently Assigned Codes")
    print(f"-" * 70)
    
    code_lookup = {code['key']: code['description'] for code in all_codes}
    
    for rank, (code_key, count) in enumerate(code_frequency.most_common(15), 1):
        desc = code_lookup.get(code_key, 'Unknown')
        pct = (count / total_responses) * 100
        print(f"{rank:2d}. [{code_key}] {desc[:45]:45s} | {count:4d} ({pct:5.1f}%)")
    
    # Least common codes
    print(f"\n📉 15 Least Frequently Assigned Codes (excluding 0)")
    print(f"-" * 70)
    
    least_common = sorted(code_frequency.items(), key=lambda x: x[1])[:15]
    for rank, (code_key, count) in enumerate(least_common, 1):
        desc = code_lookup.get(code_key, 'Unknown')
        pct = (count / total_responses) * 100
        print(f"{rank:2d}. [{code_key}] {desc[:45]:45s} | {count:4d} ({pct:5.1f}%)")
    
    # Unused codes
    all_code_keys = set(code['key'] for code in all_codes if code['key'])
    used_code_keys = set(code_frequency.keys())
    unused_codes = all_code_keys - used_code_keys
    
    if unused_codes:
        print(f"\n⚠️  {len(unused_codes)} codes defined but never assigned in responses")
    
    # Sample responses
    print(f"\n📝 Sample Responses (first 5)")
    print(f"-" * 70)
    
    for i, response in enumerate(all_responses[:5], 1):
        verbatim_elem = response.find('DROVerbatim')
        respondent_elem = response.find('DRORespondent')
        
        verbatim = verbatim_elem.text if verbatim_elem is not None else "N/A"
        respondent = respondent_elem.text if respondent_elem is not None else "N/A"
        
        response_codes = []
        for resp_code in response.findall('.//ResponseCode'):
            cbc_key_elem = resp_code.find('DCCBCKey')
            if cbc_key_elem is not None:
                code_key = cbc_key_elem.text
                desc = code_lookup.get(code_key, 'Unknown')
                response_codes.append(f"{code_key} ({desc})")
        
        print(f"\n{i}. Respondent: {respondent}")
        print(f"   Verbatim: \"{verbatim[:100]}{'...' if len(verbatim) > 100 else ''}\"")
        print(f"   Codes: {len(response_codes)}")
        for code in response_codes:
            print(f"     - {code}")
    
    print(f"\n{'='*70}")
    print(f"✓ ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_benchmark.py <path_to_benchmark_xml>")
        print("\nExamples:")
        print("  python analyze_benchmark.py Benchmark/Kids_Meal.xml")
        print("  python analyze_benchmark.py Benchmark/Fitness.xml")
        sys.exit(1)
    
    xml_path = sys.argv[1]
    
    if not Path(xml_path).exists():
        print(f"❌ ERROR: File not found: {xml_path}")
        sys.exit(1)
    
    try:
        analyze_benchmark(xml_path)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


"""
Helper script to create a model output template from benchmark XML.

This script:
1. Loads a benchmark XML file
2. Creates a copy with empty ResponseCodes
3. Saves it as a template for you to fill with model predictions

Usage:
    python create_model_template.py Benchmark/Kids_Meal.xml ModelOutput/Kids_Meal_template.xml
    
Then you can:
    1. Load the template
    2. Predict codes for each response's verbatim
    3. Fill in the ResponseCodes
    4. Run evaluation
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import sys


def create_template(benchmark_path: str, output_path: str, clear_codes: bool = True):
    """
    Create model output template from benchmark.
    
    Args:
        benchmark_path: Path to benchmark XML
        output_path: Where to save the template
        clear_codes: If True, removes all response codes; if False, keeps them as reference
    """
    print(f"\n{'='*70}")
    print(f"Creating Model Output Template")
    print(f"{'='*70}\n")
    
    # Load benchmark
    print(f"📂 Loading benchmark: {benchmark_path}")
    tree = ET.parse(benchmark_path)
    root = tree.getroot()
    
    # Statistics
    n_responses = len(root.findall('.//Response'))
    n_codes = len(root.findall('.//CodeBookCode'))
    n_questions = len(root.findall('.//Question'))
    
    print(f"   Found: {n_questions} questions, {n_responses} responses, {n_codes} codes")
    
    # Clear or mark response codes
    if clear_codes:
        print(f"\n🔧 Clearing response codes (you'll fill these with model predictions)...")
        cleared = 0
        
        for response in root.findall('.//Response'):
            response_codes = response.find('ResponseCodes')
            
            if response_codes is not None:
                # Clear all codes
                response_codes.clear()
                cleared += 1
        
        print(f"   Cleared codes from {cleared} responses")
    else:
        print(f"\n🔧 Keeping existing codes as reference (mark them for replacement)...")
        
        for response in root.findall('.//Response'):
            response_codes = response.find('ResponseCodes')
            
            if response_codes is not None:
                # Add comment marker
                for resp_code in response_codes.findall('ResponseCode'):
                    # Add a marker field
                    marker = ET.SubElement(resp_code, 'TEMPLATE_NOTE')
                    marker.text = 'REPLACE_WITH_MODEL_PREDICTION'
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    if not output_dir.exists():
        print(f"\n📁 Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save template
    print(f"\n💾 Saving template to: {output_path}")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"\n{'='*70}")
    print(f"✓ Template created successfully!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Load this template in your model script")
    print(f"  2. For each <Response>, read the <DROVerbatim>")
    print(f"  3. Use your AI model to predict codes")
    print(f"  4. Fill <ResponseCodes> with <ResponseCode> elements")
    print(f"  5. Save the modified XML")
    print(f"  6. Run: python evaluate_model.py {benchmark_path} {output_path}")
    print()


def create_example_with_predictions(benchmark_path: str, output_path: str):
    """
    Create an example with dummy predictions to show the format.
    This is NOT for evaluation - just to show structure.
    """
    print(f"\n{'='*70}")
    print(f"Creating Example with Dummy Predictions")
    print(f"{'='*70}\n")
    print("⚠️  NOTE: This creates RANDOM predictions for demonstration only!")
    print("    Do NOT use this for actual evaluation.\n")
    
    import random
    
    # Load benchmark
    tree = ET.parse(benchmark_path)
    root = tree.getroot()
    
    # Collect all code keys
    code_keys = []
    for code in root.findall('.//CodeBookCode'):
        key_elem = code.find('CBCKey')
        if key_elem is not None:
            code_keys.append(key_elem.text)
    
    print(f"📚 Found {len(code_keys)} codes in codebook")
    
    # Fill with random predictions
    modified = 0
    for response in root.findall('.//Response'):
        response_codes = response.find('ResponseCodes')
        
        if response_codes is None:
            response_codes = ET.SubElement(response, 'ResponseCodes')
        else:
            response_codes.clear()
        
        # Generate 1-3 random codes
        n_codes = random.randint(1, 3)
        selected_codes = random.sample(code_keys, min(n_codes, len(code_keys)))
        
        for code_key in selected_codes:
            resp_code = ET.SubElement(response_codes, 'ResponseCode')
            
            autocoded = ET.SubElement(resp_code, 'DCAutoCoded')
            autocoded.text = '16'
            
            cbc_key = ET.SubElement(resp_code, 'DCCBCKey')
            cbc_key.text = code_key
            
            session = ET.SubElement(resp_code, 'DCSessionKey')
            session.text = '99999'
        
        modified += 1
    
    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"✓ Created example with random predictions for {modified} responses")
    print(f"✓ Saved to: {output_path}")
    print(f"\n⚠️  REMEMBER: These are RANDOM predictions, not real model output!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create model output template from benchmark XML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create empty template (recommended)
  python create_model_template.py Benchmark/Kids_Meal.xml ModelOutput/Kids_Meal_template.xml
  
  # Create template keeping benchmark codes as reference
  python create_model_template.py Benchmark/Kids_Meal.xml ModelOutput/Kids_Meal_template.xml --keep-codes
  
  # Create example with dummy random predictions (for testing only!)
  python create_model_template.py Benchmark/Kids_Meal.xml ModelOutput/Kids_Meal_example.xml --example
        """
    )
    
    parser.add_argument('benchmark', help='Path to benchmark XML file')
    parser.add_argument('output', help='Path for output template XML file')
    parser.add_argument('--keep-codes', action='store_true',
                       help='Keep existing codes as reference (default: clear them)')
    parser.add_argument('--example', action='store_true',
                       help='Create example with random predictions (for demo only!)')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.benchmark).exists():
        print(f"❌ ERROR: Benchmark file not found: {args.benchmark}")
        sys.exit(1)
    
    try:
        if args.example:
            create_example_with_predictions(args.benchmark, args.output)
        else:
            create_template(args.benchmark, args.output, clear_codes=not args.keep_codes)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


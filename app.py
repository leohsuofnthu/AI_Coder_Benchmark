"""
AI Coder Evaluation - Web Application
Single-page application for evaluating model predictions against benchmarks.

Usage:
    python app.py
    
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import json
from evaluate_model import CodebookEvaluator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
Path('uploads').mkdir(exist_ok=True)

# Available benchmarks
BENCHMARKS = {
    'kids_meal': {
        'name': 'Kids Meal',
        'path': 'Benchmark/Kids_Meal.xml',
        'description': 'CFA Kid\'s Meal Pre-test (798 responses)',
        'size': '465KB'
    },
    'fitness': {
        'name': 'Fitness',
        'path': 'Benchmark/Fitness.xml',
        'description': 'Fitness survey responses',
        'size': '4.3MB'
    },
    'static': {
        'name': 'Static',
        'path': 'Benchmark/Static.xml',
        'description': 'Static coding benchmark',
        'size': '1.6MB'
    },
    'ferrara': {
        'name': 'Ferrara Sugar Candy',
        'path': 'Benchmark/Ferrara_Sugar_Candy_Consumer_Journey.xml',
        'description': 'Consumer journey survey',
        'size': '6.5MB'
    }
}


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html', benchmarks=BENCHMARKS)


@app.route('/api/benchmarks')
def get_benchmarks():
    """Get list of available benchmarks."""
    return jsonify({
        'benchmarks': BENCHMARKS
    })


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate uploaded model output against selected benchmark.
    
    Expects:
        - file: Model output XML
        - benchmark: Benchmark ID (e.g., 'kids_meal')
    
    Returns:
        JSON with evaluation metrics
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get selected benchmark
        benchmark_id = request.form.get('benchmark')
        if not benchmark_id or benchmark_id not in BENCHMARKS:
            return jsonify({'error': 'Invalid benchmark selected'}), 400
        
        benchmark_path = BENCHMARKS[benchmark_id]['path']
        
        # Check if benchmark exists
        if not Path(benchmark_path).exists():
            return jsonify({'error': f'Benchmark file not found: {benchmark_path}'}), 404
        
        # Save uploaded file temporarily
        upload_path = Path('uploads') / file.filename
        file.save(str(upload_path))
        
        # Run evaluation
        try:
            evaluator = CodebookEvaluator(
                benchmark_path=benchmark_path,
                model_output_path=str(upload_path)
            )
            
            # Load and process data (includes codebook validation)
            evaluator.load_data()
            aligned_data = evaluator.align_responses()
            
        except ValueError as ve:
            # Clean up uploaded file
            upload_path.unlink(missing_ok=True)
            # Return validation error with helpful message
            error_msg = str(ve)
            if "Codebook" in error_msg or "codebook" in error_msg:
                return jsonify({
                    'error': 'Codebook Architecture Mismatch',
                    'message': error_msg,
                    'suggestion': 'Make sure you copied the entire <CodeBooks> section from the benchmark XML to your model output XML. The codebook structure must be identical.'
                }), 400
            else:
                return jsonify({'error': error_msg}), 400
        
        if not aligned_data:
            return jsonify({'error': 'No aligned responses found. Check RespondentIDs and QuestionIDs match.'}), 400
        
        # Calculate metrics
        evaluator.calculate_metrics(aligned_data)
        
        # Prepare response
        overall = evaluator.metrics['overall']
        
        # Get top/bottom performing codes
        code_performance = []
        for code_key, stats in evaluator.metrics['per_code'].items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            support = stats['support']
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            code_info = evaluator.codebook.get(code_key, {})
            code_performance.append({
                'code': code_key,
                'description': code_info.get('description', 'Unknown'),
                'depth': code_info.get('depth', 0),
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'support': support,
                'tp': tp,
                'fp': fp,
                'fn': fn
            })
        
        # Sort by F1 score
        code_performance.sort(key=lambda x: x['f1'], reverse=True)
        
        # Code distribution analysis
        depth_stats = {}
        for code in code_performance:
            depth = code['depth']
            if depth not in depth_stats:
                depth_stats[depth] = {'count': 0, 'avg_f1': 0}
            depth_stats[depth]['count'] += 1
            depth_stats[depth]['avg_f1'] += code['f1']
        
        for depth in depth_stats:
            depth_stats[depth]['avg_f1'] /= depth_stats[depth]['count']
        
        # Get worst mismatches (top 20 with most errors)
        mismatches = evaluator.metrics.get('mismatches', [])[:20]
        
        # Enrich mismatches with code descriptions
        for mismatch in mismatches:
            mismatch['missed_descriptions'] = [
                evaluator.codebook.get(code, {}).get('description', 'Unknown')
                for code in mismatch['missed_codes']
            ]
            mismatch['extra_descriptions'] = [
                evaluator.codebook.get(code, {}).get('description', 'Unknown')
                for code in mismatch['extra_codes']
            ]
        
        response_data = {
            'success': True,
            'benchmark': {
                'id': benchmark_id,
                'name': BENCHMARKS[benchmark_id]['name'],
                'description': BENCHMARKS[benchmark_id]['description']
            },
            'overall': {
                'n_responses': overall['n_responses'],
                'n_mismatches': overall.get('n_mismatches', 0),
                'exact_match_ratio': round(overall['exact_match_ratio'], 4),
                'jaccard_mean': round(overall['jaccard_mean'], 4),
                'jaccard_median': round(overall['jaccard_median'], 4),
                'f1_micro': round(overall['f1_micro'], 4),
                'f1_macro': round(overall['f1_macro'], 4),
                'precision_micro': round(overall['precision_micro'], 4),
                'precision_macro': round(overall['precision_macro'], 4),
                'recall_micro': round(overall['recall_micro'], 4),
                'recall_macro': round(overall['recall_macro'], 4),
                'total_tp': overall['total_tp'],
                'total_fp': overall['total_fp'],
                'total_fn': overall['total_fn']
            },
            'top_codes': code_performance[:10],
            'bottom_codes': sorted(code_performance, key=lambda x: x['f1'])[:10],
            'depth_stats': depth_stats,
            'all_codes': code_performance,
            'worst_mismatches': mismatches
        }
        
        # Clean up uploaded file
        upload_path.unlink(missing_ok=True)
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_msg = f"Evaluation error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg, 'details': traceback.format_exc()}), 500


@app.route('/api/analyze/<benchmark_id>')
def analyze_benchmark(benchmark_id):
    """Analyze a benchmark and return statistics."""
    try:
        if benchmark_id not in BENCHMARKS:
            return jsonify({'error': 'Invalid benchmark ID'}), 400
        
        benchmark_path = BENCHMARKS[benchmark_id]['path']
        
        if not Path(benchmark_path).exists():
            return jsonify({'error': f'Benchmark file not found: {benchmark_path}'}), 404
        
        # Quick analysis
        import xml.etree.ElementTree as ET
        from collections import Counter
        
        tree = ET.parse(benchmark_path)
        root = tree.getroot()
        
        # Count codes
        n_codes = len(root.findall('.//CodeBookCode'))
        
        # Count responses
        responses = root.findall('.//Response')
        n_responses = len(responses)
        
        # Codes per response
        codes_per_response = []
        for response in responses:
            n = len(response.findall('.//ResponseCode'))
            codes_per_response.append(n)
        
        avg_codes = sum(codes_per_response) / len(codes_per_response) if codes_per_response else 0
        
        return jsonify({
            'success': True,
            'benchmark': BENCHMARKS[benchmark_id],
            'stats': {
                'n_codes': n_codes,
                'n_responses': n_responses,
                'avg_codes_per_response': round(avg_codes, 2),
                'min_codes': min(codes_per_response) if codes_per_response else 0,
                'max_codes': max(codes_per_response) if codes_per_response else 0
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 AI Coder Evaluation Web App")
    print("="*70)
    print("\n📡 Server starting...")
    print("🌐 Open in browser: http://localhost:5000")
    print("💡 Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)


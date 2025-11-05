"""
AI Coder Evaluation - Web Application
Single-page application for evaluating model predictions against benchmarks.

Usage:
    python app.py
    
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_file, session, make_response
from pathlib import Path
import json
import csv
import io
import os
from evaluate_model import CodebookEvaluator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size (reduced for Render free tier)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Session configuration for Render
# Use signed cookies (default) - works on Render free tier
# Note: Sessions are ephemeral on Render free tier (lost on restart)
# But CSV download happens immediately after evaluation, so this is fine
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Create uploads directory if possible (may fail on read-only filesystem)
try:
    Path('uploads').mkdir(exist_ok=True)
except (PermissionError, OSError):
    # On Render, use /tmp which is writable
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    Path('/tmp/uploads').mkdir(exist_ok=True)

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
        
        # Log for debugging
        print(f"[UPLOAD] Receiving file: {file.filename} for benchmark: {benchmark_id}")
        print(f"[UPLOAD] Upload folder: {app.config['UPLOAD_FOLDER']}")
        
        # Save uploaded file temporarily
        upload_folder = Path(app.config['UPLOAD_FOLDER'])
        upload_path = upload_folder / file.filename
        file.save(str(upload_path))
        print(f"[UPLOAD] File saved to: {upload_path}")
        
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
        
        # Get all mismatches (for CSV download only, not for display)
        all_mismatches = evaluator.metrics.get('mismatches', [])
        
        # Load verbatims from benchmark - keyed by (respondent_id, question_id)
        import xml.etree.ElementTree as ET
        verbatims = {}
        benchmark_tree = ET.parse(benchmark_path)
        benchmark_root = benchmark_tree.getroot()
        
        for question in benchmark_root.findall('.//Question'):
            question_id_elem = question.find('QuestionID')
            if question_id_elem is None:
                continue
            question_id = question_id_elem.text
            
            for response in question.findall('.//Response'):
                respondent = response.find('DRORespondent')
                verbatim = response.find('DROVerbatim')
                if respondent is not None and verbatim is not None:
                    key = (respondent.text, question_id)
                    verbatims[key] = verbatim.text or ""
        
        # Return minimal data first (core metrics only)
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
            # Only include summary - no detailed mismatches
            'has_mismatches': len(all_mismatches) > 0,
            'mismatch_count': len(all_mismatches)
        }
        
        # Clean up uploaded file first (before session storage)
        upload_path.unlink(missing_ok=True)
        
        # Store mismatches in session for CSV download only
        # Enrich mismatches with verbatims before storing
        mismatch_keys = {(m['respondent_id'], m['question_id']) for m in all_mismatches}
        for mismatch in all_mismatches:
            key = (mismatch['respondent_id'], mismatch['question_id'])
            mismatch['verbatim'] = verbatims.get(key, 'N/A')
            mismatch['missed_descriptions'] = [
                evaluator.codebook.get(code, {}).get('description', 'Unknown')
                for code in mismatch['missed_codes']
            ]
            mismatch['extra_descriptions'] = [
                evaluator.codebook.get(code, {}).get('description', 'Unknown')
                for code in mismatch['extra_codes']
            ]
        
        session['all_mismatches'] = all_mismatches
        session['benchmark_name'] = BENCHMARKS[benchmark_id]['name']
        session['codebook'] = {k: {'description': v.get('description', 'Unknown')} 
                               for k, v in evaluator.codebook.items()}
        
        # Calculate response size before creating JSON
        response_json = json.dumps(response_data)
        response_size = len(response_json.encode('utf-8'))
        
        print(f"[RESPONSE] Preparing response...")
        print(f"[RESPONSE] Response size: {response_size} bytes ({response_size/1024:.1f} KB)")
        print(f"[RESPONSE] Mismatches stored in session: {len(all_mismatches)}")
        print(f"[RESPONSE] Session keys: {list(session.keys())}")
        
        # Create response with explicit headers for Render compatibility
        response = make_response(response_json)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.headers['Content-Length'] = str(response_size)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['X-Content-Size'] = str(response_size)  # Debug header
        response.status_code = 200
        
        print(f"[RESPONSE] Response created, sending...")
        
        # Force flush for Render logging
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        return response
        
    except Exception as e:
        import traceback
        error_msg = f"Evaluation error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg, 'details': traceback.format_exc()}), 500


@app.route('/api/download-mismatches')
def download_mismatches():
    """Download mismatches as CSV file."""
    try:
        # Get data from session
        if 'all_mismatches' not in session:
            print("[CSV] No session data found")
            return jsonify({'error': 'No evaluation data found. Please run evaluation first.'}), 400
        
        mismatches = session.get('all_mismatches', [])
        benchmark_name = session.get('benchmark_name', 'Benchmark')
        codebook = session.get('codebook', {})
        
        if not mismatches:
            print("[CSV] Mismatches list is empty")
            return jsonify({'error': 'No mismatches found in evaluation data.'}), 400
        
        print(f"[CSV] Generating CSV for {len(mismatches)} mismatches")
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
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
        for rank, mismatch in enumerate(mismatches, 1):
            respondent_id = mismatch['respondent_id']
            question_id = mismatch['question_id']
            verbatim = mismatch.get('verbatim', 'N/A')
            
            # Get code descriptions (use pre-stored or fallback)
            missed_descs = mismatch.get('missed_descriptions', [
                codebook.get(code, {}).get('description', 'Unknown')
                for code in mismatch['missed_codes']
            ])
            extra_descs = mismatch.get('extra_descriptions', [
                codebook.get(code, {}).get('description', 'Unknown')
                for code in mismatch['extra_codes']
            ])
            
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
        
        # Prepare file for download
        output.seek(0)
        csv_content = output.getvalue()
        csv_bytes = csv_content.encode('utf-8-sig')  # UTF-8 with BOM for Excel compatibility
        filename = f"{benchmark_name.replace(' ', '_')}_mismatches.csv"
        
        print(f"[CSV] CSV generated: {len(csv_bytes)} bytes, {len(mismatches)} rows")
        
        # Create response with proper headers for Render
        response = make_response(csv_bytes)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        response.headers['Content-Length'] = str(len(csv_bytes))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        
        print(f"[CSV] Response created, sending file...")
        
        return response
        
    except Exception as e:
        import traceback
        print(f"Download error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/test')
def test_endpoint():
    """Test endpoint to verify server can send responses."""
    return jsonify({
        'status': 'ok',
        'message': 'Server is responding',
        'timestamp': __import__('datetime').datetime.now().isoformat()
    })


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
    # Get port from environment variable (for Render/Heroku) or use 5000
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print("\n" + "="*70)
    print("🚀 AI Coder Evaluation Web App")
    print("="*70)
    print("\n📡 Server starting...")
    print(f"🌐 Open in browser: http://localhost:{port}")
    print("💡 Press Ctrl+C to stop\n")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)


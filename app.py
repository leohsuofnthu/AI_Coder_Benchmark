"""
AI Coder Evaluation - Web Application
Single-page application for evaluating model predictions against benchmarks.

Usage:
    python app.py
    
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_file, session, make_response
from flask_session import Session
from pathlib import Path
import json
import csv
import io
import os
from evaluate_model import CodebookEvaluator
from hierarchy_evaluator import HierarchyEvaluator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size (reduced for Render free tier)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Robust server-side session configuration using Flask-Session
# This solves the 4KB cookie size limit by storing sessions on the server filesystem
# Works for both local development and production (Render)
app.config['SESSION_TYPE'] = 'filesystem'  # Server-side session storage
app.config['SESSION_PERMANENT'] = False  # Sessions expire when browser closes
app.config['SESSION_USE_SIGNER'] = True  # Sign session cookies for security
app.config['SESSION_KEY_PREFIX'] = 'ai_coder:'  # Prefix for session keys
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Determine session directory (works on both local and Render)
# Render sets RENDER environment variable, also check for production mode
# Use /tmp on Render (ephemeral but writable) or local directory for development
if os.environ.get('RENDER') or os.environ.get('FLASK_ENV') == 'production':
    # On Render, use /tmp which is writable (ephemeral but fine for short-lived sessions)
    session_dir = '/tmp/flask_session'
    print(f"[SESSION] Using Render filesystem backend: {session_dir}")
else:
    # Local development
    session_dir = 'flask_session'
    print(f"[SESSION] Using local filesystem backend: {session_dir}")

app.config['SESSION_FILE_DIR'] = session_dir
try:
    Path(session_dir).mkdir(exist_ok=True, parents=True)
    print(f"[SESSION] Session directory created/verified: {session_dir}")
except Exception as e:
    print(f"[SESSION] Warning: Could not create session directory {session_dir}: {e}")
    # Fallback to /tmp if available
    try:
        session_dir = '/tmp/flask_session'
        Path(session_dir).mkdir(exist_ok=True, parents=True)
        app.config['SESSION_FILE_DIR'] = session_dir
        print(f"[SESSION] Using fallback directory: {session_dir}")
    except Exception as fallback_error:
        print(f"[SESSION] Error: Could not create fallback directory: {fallback_error}")

# Initialize Flask-Session
Session(app)

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
        
        upload_path_str = str(upload_path)
        benchmark_path_str = benchmark_path
        
        # Run evaluation synchronously
        try:
            evaluator = CodebookEvaluator(
                benchmark_path=benchmark_path_str,
                model_output_path=upload_path_str,
                progress_callback=None  # No progress tracking
            )
            
            # Load and process data (includes codebook validation)
            evaluator.load_data()
            aligned_data = evaluator.align_responses()
            
            if not aligned_data:
                Path(upload_path_str).unlink(missing_ok=True)
                return jsonify({
                    'error': 'No aligned responses found. Check RespondentIDs and QuestionIDs match.'
                }), 400
            
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
            benchmark_tree = ET.parse(benchmark_path_str)
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
            
            # Prepare response data
            response_data = {
                'success': True,
                'benchmark': {
                    'id': benchmark_id,
                    'name': BENCHMARKS[benchmark_id]['name'],
                    'description': BENCHMARKS[benchmark_id]['description']
                },
                'overall': {
                    'coded_only': overall.get('coded_only', {}),
                    'n_responses': overall['n_responses'],
                    'n_mismatches': overall.get('n_mismatches', 0),
                    'total_tp': overall['total_tp'],
                    'total_fp': overall['total_fp'],
                    'total_fn': overall['total_fn'],
                    'uncoded_classification': overall.get('uncoded_classification', {}),
                    'uncoded_benchmark': overall.get('uncoded_benchmark', 0),
                    'uncoded_model': overall.get('uncoded_model', 0),
                    'uncoded_both': overall.get('uncoded_both', 0),
                    'question_uncoded_breakdown': overall.get('question_uncoded_breakdown', [])
                },
                'top_codes': code_performance[:10],
                'bottom_codes': sorted(code_performance, key=lambda x: x['f1'])[:10],
                'depth_stats': depth_stats,
                'has_mismatches': len(all_mismatches) > 0,
                'mismatch_count': len(all_mismatches)
            }
            
            # Enrich mismatches with verbatims
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
            
            # Store in session for CSV download
            try:
                session['all_mismatches'] = all_mismatches
                session['benchmark_name'] = BENCHMARKS[benchmark_id]['name']
                session['codebook'] = {k: {'description': v.get('description', 'Unknown')} 
                                       for k, v in evaluator.codebook.items()}
            except Exception as session_error:
                print(f"[SESSION] Error storing session: {session_error}")
            
            # Clean up files
            Path(upload_path_str).unlink(missing_ok=True)
            
            return jsonify(response_data)
            
        except ValueError as ve:
            # Clean up files
            Path(upload_path_str).unlink(missing_ok=True)
            return jsonify({'error': str(ve)}), 400
        except Exception as eval_error:
            # Clean up files
            Path(upload_path_str).unlink(missing_ok=True)
            raise eval_error
        
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
        print("[CSV] ===== CSV Download Request Received =====")
        print(f"[CSV] Request method: {request.method}")
        print(f"[CSV] Session keys: {list(session.keys())}")
        print(f"[CSV] Using server-side session storage (Flask-Session filesystem backend)")
        
        # Get data from session
        if 'all_mismatches' not in session:
            print("[CSV] ❌ No 'all_mismatches' key in session")
            print("[CSV] Available session keys:", list(session.keys()))
            return jsonify({
                'error': 'No evaluation data found',
                'message': 'The evaluation session has expired or no evaluation was run. Please run an evaluation first, then download the CSV immediately after.',
                'suggestion': 'Run a new evaluation and download the CSV right away. Sessions may expire after inactivity.'
            }), 400
        
        mismatches = session.get('all_mismatches', [])
        benchmark_name = session.get('benchmark_name', 'Benchmark')
        codebook = session.get('codebook', {})
        
        print(f"[CSV] ✅ Found {len(mismatches)} mismatches in session")
        print(f"[CSV] Benchmark name: {benchmark_name}")
        print(f"[CSV] Codebook size: {len(codebook)} codes")
        
        if not mismatches:
            print("[CSV] ❌ Mismatches list is empty")
            return jsonify({'error': 'No mismatches found in evaluation data.'}), 400
        
        print(f"[CSV] 📝 Generating CSV for {len(mismatches)} mismatches")
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header - Concise format
        writer.writerow([
            'Rank',
            'Respondent ID',
            'Question ID',
            'Error Count',
            'Jaccard',
            'Missed Codes',
            'Extra Codes',
            'Verbatim'
        ])
        
        # Data rows
        print(f"[CSV] Writing {len(mismatches)} rows to CSV...")
        rows_written = 0
        for rank, mismatch in enumerate(mismatches, 1):
            try:
                respondent_id = mismatch.get('respondent_id', 'N/A')
                question_id = mismatch.get('question_id', 'N/A')
                verbatim = mismatch.get('verbatim', 'N/A')
                
                # Get code descriptions (use pre-stored or fallback)
                missed_descs = mismatch.get('missed_descriptions', [
                    codebook.get(code, {}).get('description', 'Unknown')
                    for code in mismatch.get('missed_codes', [])
                ])
                extra_descs = mismatch.get('extra_descriptions', [
                    codebook.get(code, {}).get('description', 'Unknown')
                    for code in mismatch.get('extra_codes', [])
                ])
                
                # Format codes with full descriptions - "CODE: Description"
                missed_formatted = []
                missed_codes_list = mismatch.get('missed_codes', [])
                for i, code in enumerate(missed_codes_list):
                    desc = missed_descs[i] if i < len(missed_descs) else 'Unknown'
                    missed_formatted.append(f"{code}: {desc}")
                
                extra_formatted = []
                extra_codes_list = mismatch.get('extra_codes', [])
                for i, code in enumerate(extra_codes_list):
                    desc = extra_descs[i] if i < len(extra_descs) else 'Unknown'
                    extra_formatted.append(f"{code}: {desc}")
                
                writer.writerow([
                    rank,
                    respondent_id,
                    question_id,
                    mismatch.get('error_count', 0),
                    f"{mismatch.get('jaccard', 0):.2f}",
                    ' | '.join(missed_formatted) if missed_formatted else 'None',
                    ' | '.join(extra_formatted) if extra_formatted else 'None',
                    verbatim
                ])
                rows_written += 1
            except Exception as row_error:
                print(f"[CSV] ❌ Error writing row {rank}: {row_error}")
                import traceback
                print(traceback.format_exc())
                continue
        
        print(f"[CSV] ✅ Wrote {rows_written} rows successfully")
        
        # Prepare file for download
        output.seek(0)
        csv_content = output.getvalue()
        csv_bytes = csv_content.encode('utf-8-sig')  # UTF-8 with BOM for Excel compatibility
        filename = f"{benchmark_name.replace(' ', '_')}_mismatches.csv"
        
        print(f"[CSV] CSV generated: {len(csv_bytes)} bytes, {len(mismatches)} rows")
        print(f"[CSV] Filename: {filename}")
        print(f"[CSV] CSV preview (first 200 chars): {csv_content[:200]}")
        
        # Create response with proper headers for Render
        response = make_response(csv_bytes)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        response.headers['Content-Length'] = str(len(csv_bytes))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['X-Filename'] = filename  # Debug header
        
        print(f"[CSV] ✅ Response created with headers:")
        print(f"[CSV]   Content-Type: {response.headers.get('Content-Type')}")
        print(f"[CSV]   Content-Length: {response.headers.get('Content-Length')}")
        print(f"[CSV]   Content-Disposition: {response.headers.get('Content-Disposition')}")
        print(f"[CSV] 🚀 Sending file to client...")
        
        # Force flush
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        return response
        
    except Exception as e:
        import traceback
        print(f"Download error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate-single-hierarchy', methods=['POST'])
def evaluate_single_hierarchy():
    """Evaluate metrics for a single hierarchy XML file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'XML file required'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'File must be selected'}), 400
        
        # Save file temporarily
        upload_folder = Path(app.config['UPLOAD_FOLDER'])
        file_path = upload_folder / file.filename
        file.save(str(file_path))
        file_path_str = str(file_path)
        
        try:
            from hierarchy_evaluator import HierarchyEvaluator, HierarchyMetricsEvaluator
            
            # Build tree
            evaluator = HierarchyEvaluator(file_path_str, file_path_str)  # Use same file for both
            root_nodes = evaluator.extract_hierarchy(file_path_str)
            
            if not root_nodes:
                Path(file_path_str).unlink(missing_ok=True)
                return jsonify({'error': 'No hierarchy tree could be built from the XML file'}), 400
            
            # Create virtual root if multiple root nodes
            if len(root_nodes) > 1:
                from hierarchy_evaluator import HierarchyNode
                virtual_root = HierarchyNode('root', 'Root', 0, False, 0)
                virtual_root.children = root_nodes
                for node in root_nodes:
                    node.parent = virtual_root
                root = virtual_root
            else:
                root = root_nodes[0]
            
            # Compute metrics (no progress callback)
            metrics_evaluator = HierarchyMetricsEvaluator(progress_callback=None)
            metrics = metrics_evaluator.compute_metrics(root, file_path_str)
            
            if metrics is None:
                Path(file_path_str).unlink(missing_ok=True)
                return jsonify({'error': 'Failed to compute metrics. Check if hierarchy has valid structure.'}), 400
            
            # Verify metrics structure
            if not isinstance(metrics, dict):
                Path(file_path_str).unlink(missing_ok=True)
                return jsonify({'error': f'Invalid metrics type: {type(metrics)}, expected dict'}), 500
            
            # Prepare response
            response_data = {
                'success': True,
                'metrics': metrics,
                'tree': root.to_dict() if hasattr(root, 'to_dict') else None,
                'file_name': file.filename
            }
            
            print(f"[METRICS] Results prepared: success={response_data['success']}, metrics_keys={list(metrics.keys()) if metrics else 'None'}")
            
            # Clean up file
            Path(file_path_str).unlink(missing_ok=True)
            
            return jsonify(response_data)
            
        except Exception as eval_error:
            # Clean up file on error
            Path(file_path_str).unlink(missing_ok=True)
            raise eval_error
        
    except Exception as e:
        import traceback
        error_msg = f"Single hierarchy evaluation error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg, 'details': traceback.format_exc()}), 500


@app.route('/api/compare-hierarchy', methods=['POST'])
def compare_hierarchy():
    """Extract and compare hierarchy structures side-by-side (no metrics calculation)."""
    try:
        if 'benchmark_file' not in request.files or 'model_file' not in request.files:
            return jsonify({'error': 'Both benchmark and model files required'}), 400
        
        benchmark_file = request.files['benchmark_file']
        model_file = request.files['model_file']
        
        if benchmark_file.filename == '' or model_file.filename == '':
            return jsonify({'error': 'Both files must be selected'}), 400
        
        # Save files temporarily
        upload_folder = Path(app.config['UPLOAD_FOLDER'])
        benchmark_path = upload_folder / benchmark_file.filename
        model_path = upload_folder / f"model_{model_file.filename}"
        
        benchmark_file.save(str(benchmark_path))
        model_file.save(str(model_path))
        
        try:
            # Extract hierarchies (simple, no metrics)
            evaluator = HierarchyEvaluator(str(benchmark_path), str(model_path))
            
            # Extract hierarchy structures only
            benchmark_roots = evaluator.extract_hierarchy(str(benchmark_path))
            model_roots = evaluator.extract_hierarchy(str(model_path))
            
            # Convert to single root if multiple roots
            from hierarchy_evaluator import HierarchyNode
            if len(benchmark_roots) == 1:
                benchmark_tree = benchmark_roots[0]
            else:
                virtual_root = HierarchyNode('root', 'Root', 0, False, 0)
                virtual_root.children = benchmark_roots
                benchmark_tree = virtual_root
            
            if len(model_roots) == 1:
                model_tree = model_roots[0]
            else:
                virtual_root = HierarchyNode('root', 'Root', 0, False, 0)
                virtual_root.children = model_roots
                model_tree = virtual_root
            
            # Prepare response
            response_data = {
                'success': True,
                'benchmark': {
                    'tree': benchmark_tree.to_dict() if hasattr(benchmark_tree, 'to_dict') else None,
                    'file_name': benchmark_file.filename
                },
                'model': {
                    'tree': model_tree.to_dict() if hasattr(model_tree, 'to_dict') else None,
                    'file_name': model_file.filename
                }
            }
            
            # Clean up files
            benchmark_path.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)
            
            return jsonify(response_data)
            
        except Exception as eval_error:
            # Clean up files on error
            benchmark_path.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)
            raise eval_error
        
    except Exception as e:
        import traceback
        error_msg = f"Hierarchy comparison error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg, 'details': traceback.format_exc()}), 500


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


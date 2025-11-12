# 🎯 AI Coder Evaluation Platform

Web application for evaluating AI model performance in mapping survey responses to predefined codes using multi-label classification metrics.

## 🌟 Features

- 🌐 **Beautiful Web Interface** - Professional dark theme with interactive charts
- 📤 **Drag & Drop Upload** - Easy XML file upload
- 📊 **Comprehensive Metrics** - F1, Precision, Recall, Jaccard, Exact Match
- 🔍 **Mismatch Analysis** - See exactly where your model fails
- 📝 **Verbatim Display** - View actual response text for errors
- 💾 **CSV Export** - Download full error analysis for Excel
- ⚡ **Fast Evaluation** - Results in 2-15 seconds
- 🔒 **Codebook Validation** - Ensures model uses correct codes

## 📁 Project Structure

```
AI_Coder_Metrics/
├── app.py                    # Web application (Flask)
├── evaluate_model.py         # Core evaluation engine
├── templates/index.html      # Web UI
├── Benchmark/               # 4 benchmark XML files
│   ├── Kids_Meal.xml
│   ├── Fitness.xml
│   ├── Static.xml
│   └── Ferrara_Sugar_Candy_Consumer_Journey.xml
├── ModelOutput/
│   └── Kids_Meal_example.xml # Example model output
├── requirements.txt         # Dependencies (Flask, Gunicorn)
├── README.md               # This file
├── DEPLOYMENT.md           # Render deployment guide
└── LICENSE                 # MIT License
```

## 🚀 Quick Start

### Web Application (Recommended)

#### Install CPU-Only Version (Recommended for CPU-only systems)

```bash
# Step 1: Install CPU-only PyTorch (no CUDA dependencies, smaller size)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install other dependencies
pip install -r requirements.txt

# Step 3: Run locally
python app.py

# Open browser to: http://localhost:5000
```

#### Alternative: Install with GPU Support (if you have CUDA)

```bash
# Install dependencies (includes GPU support)
pip install -r requirements.txt

# Run locally
python app.py

# Open browser to: http://localhost:5000
```

**Note**: The CPU-only version is recommended because:
- ✅ **Smaller installation** (~150MB vs ~2GB for GPU version)
- ✅ **No CUDA dependencies** (works on any system)
- ✅ **Faster installation** (no GPU drivers needed)
- ✅ **Sufficient performance** (sentence-transformers is optimized for CPU)

### Usage
1. **Upload** your model's XML output (drag & drop or click)
2. **Select** which benchmark to compare against
3. **Evaluate** and view comprehensive metrics
4. **Analyze** mismatches with verbatim text
5. **Download** CSV report for detailed error analysis

### Example Files
- Use `ModelOutput/Kids_Meal_example.xml` to test the app
- Compare against `Benchmark/Kids_Meal.xml`

## 📊 Evaluation Metrics

### Overall Metrics

| Metric | Description | Range | Goal |
|--------|-------------|-------|------|
| **Exact Match Ratio** | % responses with all codes correct | 0-1 | Higher ✓ |
| **F1-Score (Micro)** | Harmonic mean of precision/recall | 0-1 | Higher ✓ |
| **F1-Score (Macro)** | Average F1 across all codes | 0-1 | Higher ✓ |
| **Jaccard Similarity** | Intersection over Union | 0-1 | Higher ✓ |
| **Precision** | Correct predictions / total predictions | 0-1 | Higher ✓ |
| **Recall** | Correct predictions / total ground truth | 0-1 | Higher ✓ |

### Per-Code Analysis
- Support (frequency)
- True Positives, False Positives, False Negatives
- F1, Precision, Recall per code
- Best/worst performing codes

### Mismatch Analysis
- Ranked by error count
- Shows missed codes (false negatives)
- Shows extra codes (false positives)
- Displays verbatim text
- Exportable to CSV with full details

## 🔧 XML Format

### Required Structure

Your model output XML must match the benchmark structure:

```xml
<?xml version="1.0" encoding="utf-8"?>
<Study>
  <CodeBook>
    <CodeBookCode>
      <CBCKey>100</CBCKey>
      <CBCDescription>Response description</CBCDescription>
      <CBCDepth>1</CBCDepth>
      <CBCIsNet>0</CBCIsNet>
    </CodeBookCode>
    <!-- More codes... -->
  </CodeBook>
  
  <Questions>
    <Question>
      <QuestionKey>Q1</QuestionKey>
      <Responses>
        <Response>
          <DRRespondentID>12345</DRRespondentID>
          <DROVerbatim>The actual response text</DROVerbatim>
          <ResponseCodes>
            <ResponseCode>
              <DCCBCKey>100</DCCBCKey>
              <!-- Your model's predicted code -->
            </ResponseCode>
          </ResponseCodes>
        </Response>
      </Responses>
    </Question>
  </Questions>
</Study>
```

### Key Requirements
- ✅ **Same codebook** as benchmark (validated automatically)
- ✅ **Same respondent IDs** and verbatim text
- ✅ **Same questions** structure
- ⚠️ **Different ResponseCodes** (your predictions)

## 💻 Command-Line Usage

### Basic Evaluation
```bash
python evaluate_model.py Benchmark/Kids_Meal.xml ModelOutput/Kids_Meal_example.xml
```

### With Mismatch Analysis
```bash
# Show top 20 worst mismatches in console
python evaluate_model.py benchmark.xml model.xml --show-mismatches 20 --with-verbatims

# Export all mismatches to CSV
python evaluate_model.py benchmark.xml model.xml --export-mismatches report.csv
```

### Output
- Console summary with key metrics
- `evaluation_report.json` - Full metrics in JSON
- Optional CSV with all mismatches and verbatims

## 🔒 Codebook Validation

The evaluator automatically checks:
- ✅ Same number of codes
- ✅ Matching code keys (CBCKey)
- ✅ Identical descriptions
- ✅ Same depth levels
- ✅ Matching net/non-net flags

**Error Example:**
```
ValueError: Codebook mismatch - Different number of codes
Benchmark: 45 codes
Model Output: 43 codes
Suggestion: Ensure your model uses the exact same codebook...
```

## 📈 Best Practices

### Model Development
1. **Start with example**: Use `Kids_Meal_example.xml` as template
2. **Validate early**: Test with small samples first
3. **Check codebook**: Ensure exact match with benchmark
4. **Iterate**: Use mismatch analysis to improve

### Evaluation
1. **Review overall metrics**: Check F1-Micro and Exact Match
2. **Analyze per-code**: Identify which codes are problematic
3. **Study mismatches**: Use verbatim text to understand failures
4. **Export to CSV**: Deep dive in Excel/Google Sheets
5. **Compare benchmarks**: Test on multiple datasets

## 🌐 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed Render deployment instructions.

**Quick Deploy to Render:**
1. Push to GitHub
2. Connect to Render
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn app:app`
5. Set `SECRET_KEY` environment variable

## 📊 Available Benchmarks

| Benchmark | Codes | Responses | Avg Codes/Response |
|-----------|-------|-----------|-------------------|
| Kids Meal | 45 | 100 | 2.3 |
| Fitness | 52 | 150 | 3.1 |
| Static | 38 | 80 | 1.8 |
| Ferrara Sugar Candy | 67 | 200 | 4.2 |

## 🛠️ Technical Details

- **Python**: 3.7+
- **Framework**: Flask 2.3+
- **Server**: Gunicorn (production)
- **ML Backend**: sentence-transformers (CPU-optimized, all-MiniLM-L6-v2 model)
- **Dependencies**: Flask, sentence-transformers, PyTorch (CPU-only), scikit-learn, numpy
- **Storage**: File-based (no database needed)
- **Session**: Flask sessions for CSV download
- **Upload Limit**: 50MB per file

### Dependencies Installation

**CPU-Only Installation (Recommended):**
```bash
# Install CPU-only PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies
pip install -r requirements.txt
```

**Benefits of CPU-Only Version:**
- Smaller package size (~150MB vs ~2GB)
- No CUDA/GPU drivers required
- Works on any system (Windows, Mac, Linux)
- Fast enough for hierarchy metrics (CPU-optimized model)

## 📝 License

MIT License - See [LICENSE](LICENSE) file

## 🤝 Support

- Check logs in Render dashboard if deployment fails
- Use browser console to debug frontend issues
- Validate XML structure matches benchmark format
- Test with provided example files first

---

**Built for evaluating AI coding models with professional metrics and analysis tools.**

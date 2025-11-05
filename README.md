# AI Coder Evaluation Metrics

Evaluation framework for measuring AI model accuracy in mapping survey verbatim responses to predefined codes using multi-label classification metrics.

## 📁 Project Structure

```
AI_Coder_Metrics/
├── Benchmark/              # Ground truth XML files (human-coded)
├── ModelOutput/            # Model predictions (you generate these)
├── evaluate_model.py       # Main evaluation script
├── analyze_benchmark.py    # Benchmark analysis tool
├── create_model_template.py # Template generator
├── README.md              # This file
└── requirements.txt       # Dependencies (none required!)
```

## 🚀 Quick Start

### Option A: Web Interface (Recommended) 🌐
```bash
conda activate coder
pip install flask  # Only needed for web app
python app.py
```
Open browser to `http://localhost:5000` for a beautiful interface!

**Features:**
- 🎨 Beautiful, modern UI with gradient design
- 📤 Drag & drop XML file upload
- 📊 Interactive benchmark selection (4 options)
- 📈 Real-time evaluation with loading indicator
- 🎯 Color-coded metric cards (green=excellent, yellow=fair, red=poor)
- 📉 Interactive charts (radar chart + confusion matrix)
- 🏆 Top 10 best/worst performing codes
- 📱 Responsive design (works on mobile/tablet/desktop)

### Option B: Command Line

### 1. Setup
```bash
conda activate coder  # No pip installs needed for command-line!
```

### 2. Analyze Benchmark
```bash
python analyze_benchmark.py Benchmark/Kids_Meal.xml
```

Shows: codes structure, response statistics, most/least common codes, samples.

### 3. Create Template
```bash
python create_model_template.py Benchmark/Kids_Meal.xml ModelOutput/Kids_Meal_template.xml
```

### 4. Generate Model Predictions

Your model must produce XML with:
- ✅ Same codebook as benchmark
- ✅ Same respondents and verbatims  
- ⚠️ Different ResponseCodes (your predictions)

**Minimal example:**
```python
import xml.etree.ElementTree as ET

# Load template
tree = ET.parse('ModelOutput/Kids_Meal_template.xml')
root = tree.getroot()

# Get codebook
codebook = {}
for code in root.findall('.//CodeBookCode'):
    codebook[code.find('CBCKey').text] = code.find('CBCDescription').text

# Predict for each response
for response in root.findall('.//Response'):
    verbatim = response.find('DROVerbatim').text
    
    # YOUR MODEL HERE
    predicted_codes = your_model.predict(verbatim, codebook)
    
    # Fill ResponseCodes
    response_codes = response.find('ResponseCodes')
    if not response_codes:
        response_codes = ET.SubElement(response, 'ResponseCodes')
    else:
        response_codes.clear()
    
    for code_key in predicted_codes:
        resp_code = ET.SubElement(response_codes, 'ResponseCode')
        ET.SubElement(resp_code, 'DCAutoCoded').text = '16'
        ET.SubElement(resp_code, 'DCCBCKey').text = code_key
        ET.SubElement(resp_code, 'DCSessionKey').text = '99999'

tree.write('ModelOutput/Kids_Meal_predicted.xml', encoding='utf-8', xml_declaration=True)
```

### 5. Evaluate
```bash
python evaluate_model.py Benchmark/Kids_Meal.xml ModelOutput/Kids_Meal_predicted.xml
```

Output: Overall metrics, best/worst codes, detailed JSON report.

## 📊 Metrics Explained

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **F1-Score (Micro)** | Overall balanced accuracy (aggregated) | > 0.70 |
| **F1-Score (Macro)** | Average per code (unbiased) | > 0.60 |
| **Exact Match Ratio** | % responses with ALL codes correct | > 0.50 |
| **Jaccard Similarity** | Average overlap (∩/∪) | > 0.70 |
| **Precision** | % predicted codes that are correct | > 0.75 |
| **Recall** | % actual codes that were predicted | > 0.70 |

### Interpreting Results

**High Precision, Low Recall**: Model too conservative → lower threshold  
**Low Precision, High Recall**: Model too aggressive → raise threshold  
**Good Micro, Poor Macro**: Works on frequent codes, fails on rare ones → balance data

## 🎯 Expected Performance

| Quality | F1 (Micro) | Exact Match | Notes |
|---------|------------|-------------|-------|
| Random | 0.01-0.02 | < 1% | Baseline |
| Poor | 0.20-0.40 | 5-15% | Needs work |
| Fair | 0.40-0.60 | 15-30% | Acceptable |
| Good | 0.60-0.75 | 30-50% | Production-ready |
| Excellent | 0.75-0.85 | 50-70% | High quality |
| Human-level | 0.85-0.95 | 70-85% | State-of-art |

## 💻 Command-Line Usage

### Evaluate Model
```bash
# Basic evaluation
python evaluate_model.py <benchmark.xml> <model_output.xml>

# Show top 20 worst mismatches
python evaluate_model.py <benchmark.xml> <model_output.xml> -m 20

# Show mismatches with verbatim text (slower)
python evaluate_model.py <benchmark.xml> <model_output.xml> -m 20 -v

# Custom output path
python evaluate_model.py <benchmark.xml> <model_output.xml> --output report.json

# Don't save JSON report
python evaluate_model.py <benchmark.xml> <model_output.xml> --no-save
```

**Options:**
- `-m N, --show-mismatches N` - Display top N worst mismatched responses
- `-v, --with-verbatims` - Include verbatim text in mismatch display
- `-o, --output FILE` - Output path for JSON report
- `--no-save` - Don't save JSON report

### Analyze Benchmark
```bash
python analyze_benchmark.py <benchmark.xml>
```

### Create Template
```bash
# Empty template (recommended)
python create_model_template.py <benchmark.xml> <output.xml>

# Keep codes as reference
python create_model_template.py <benchmark.xml> <output.xml> --keep-codes

# Random predictions for testing
python create_model_template.py <benchmark.xml> <output.xml> --example
```

## 🔧 Programmatic Usage

```python
from evaluate_model import CodebookEvaluator

evaluator = CodebookEvaluator('Benchmark/Kids_Meal.xml', 'ModelOutput/predicted.xml')
metrics = evaluator.run(save_report=True)

# Access metrics
print(f"F1-Score: {metrics['overall']['f1_micro']:.4f}")
print(f"Exact Match: {metrics['overall']['exact_match_ratio']:.2%}")

# Analyze specific code
code_stats = metrics['per_code']['346997']
print(f"TP={code_stats['tp']}, FP={code_stats['fp']}, FN={code_stats['fn']}")

# Find problematic codes
for code, stats in metrics['per_code'].items():
    recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
    if recall < 0.3 and stats['support'] >= 10:
        desc = evaluator.codebook[code]['description']
        print(f"Low recall: {desc}")
```

## 📋 XML Structure Required

Both benchmark and model output must follow this structure:

```xml
<Studies>
  <Study>
    <CodeBooks>
      <CodeBook>
        <CodeBookCodes>
          <CodeBookCode>
            <CBCKey>346997</CBCKey>                    <!-- Code identifier -->
            <CBCDescription>Activities/puzzles</CBCDescription>
            <CBCDepth>3</CBCDepth>                    <!-- Hierarchy level -->
          </CodeBookCode>
        </CodeBookCodes>
      </CodeBook>
    </CodeBooks>
    
    <Questions>
      <Question>
        <QuestionID>Q10C</QuestionID>
        <Responses>
          <Response>
            <DROVerbatim>My son loves it!</DROVerbatim>
            <DRORespondent>134CK3513UC</DRORespondent>  <!-- Match key -->
            <ResponseCodes>
              <ResponseCode>
                <DCCBCKey>346997</DCCBCKey>            <!-- Assigned code -->
              </ResponseCode>
            </ResponseCodes>
          </Response>
        </Responses>
      </Question>
    </Questions>
  </Study>
</Studies>
```

**Critical**: `DRORespondent` + `QuestionID` must match between benchmark and model output for alignment.

## 📊 Output Report

`evaluation_report.json` contains:
- **overall**: All aggregate metrics (including `n_mismatches` count)
- **per_code**: TP/FP/FN, precision, recall, F1 for each code
- **per_response**: Detailed breakdown per response
- **mismatches**: List of all mismatched responses with details
- **codebook**: Code definitions

### Mismatch Analysis

Each mismatch includes:
- `respondent_id` / `question_id` - Response identifier
- `ground_truth` - Codes that should have been assigned
- `predicted` - Codes the model actually assigned
- `missed_codes` - Codes the model failed to predict (False Negatives)
- `extra_codes` - Codes the model incorrectly added (False Positives)
- `jaccard` - Overlap score for this response
- `error_count` - Total errors (FP + FN)

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Codebook size mismatch" | Model XML must have same number of codes as benchmark |
| "Codebook architecture mismatch" | Copy entire `<CodeBooks>` section from benchmark - must be identical |
| "Missing codes from benchmark" | Model XML is using different code keys - use exact same codebook |
| "No aligned responses found" | Check `DRORespondent` and `QuestionID` match exactly |
| Very low scores (< 0.1) | Verify model is running (not copying benchmark) |
| XML parsing error | Validate XML syntax, escape special chars (`&lt;` `&gt;` `&amp;`) |

### ⚠️ Codebook Validation

The system **validates that your model output has the exact same codebook** as the benchmark:
- ✅ Same number of codes
- ✅ Same code keys (CBCKey values)
- ✅ Same code descriptions
- ✅ Same hierarchy depth levels
- ✅ Same net code flags

**Why?** Evaluation only makes sense if you're using the same code structure. Copy the entire `<CodeBooks>` section from the benchmark XML to your model output XML.

## 🎓 Key Insights

### About the Data (Kids_Meal example)
- 121 codes in hierarchy (4 levels deep)
- 798 responses across 2 questions
- Average 1.36 codes per response
- Most common: "None/Nothing/NA" (33.5%)
- Median verbatim length: 30 characters

### Best Practices
1. **Start small**: Test on Kids_Meal.xml first (smallest)
2. **Establish baseline**: Run random predictions to ensure model beats it
3. **Analyze failures**: Use per-code metrics to find systematic errors
4. **Track progress**: Save reports to monitor improvements
5. **Sanity check**: Manually review predictions for face validity

## 📦 Dependencies

**None required!** Uses Python 3.7+ standard library only:
- `xml.etree.ElementTree` - XML parsing
- `collections` - Data structures
- `json` - Report output
- `pathlib` - File handling

## 🔬 Testing Example

```bash
# Create random baseline
python create_model_template.py Benchmark/Kids_Meal.xml ModelOutput/random.xml --example

# Evaluate (expect F1 ≈ 0.01)
python evaluate_model.py Benchmark/Kids_Meal.xml ModelOutput/random.xml
```

Expected: F1-Score (Micro) ≈ 0.0127, Exact Match ≈ 0.6%

## 📈 Workflow

```bash
# 1. Understand data
python analyze_benchmark.py Benchmark/Kids_Meal.xml

# 2. Create template
python create_model_template.py Benchmark/Kids_Meal.xml ModelOutput/template.xml

# 3. Run your model (generate predictions)
python your_model.py

# 4. Evaluate
python evaluate_model.py Benchmark/Kids_Meal.xml ModelOutput/predicted.xml

# 5. Analyze & iterate
# Review evaluation_report.json, improve model, repeat
```

## 📝 License

MIT License - Free to use and modify.

## 🤝 Support

Questions? Issues?
1. Check this README
2. Run `python evaluate_model.py --help`
3. Examine `evaluation_report.json` structure
4. Review the Python scripts (well-commented)

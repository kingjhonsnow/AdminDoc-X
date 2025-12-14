# AdminDoc-X: Information Extraction & NER on Administrative Documents

A Flask-based web application for automated extraction and classification of key information from administrative PDF documents using OCR, document classification, and Named Entity Recognition (NER).

## 🎯 Objectives

- Extract key administrative fields (Name, CIN, Dates, Organisation) from noisy document scans
- Classify document types (attestation, relevé, convention, other)
- Perform Named Entity Recognition (NER) to identify persons, organisations, and dates
- Validate extracted fields against business rules and patterns
- Export structured JSON results with validation status

## 📋 Requirements

### System Dependencies
- **Tesseract OCR**: For text extraction from PDFs
  - Install from: https://github.com/UB-Mannheim/tesseract/wiki
  - Default path (Windows): `C:\Program Files\Tesseract-OCR\tesseract.exe`
  
- **Poppler**: For PDF image conversion
  - Download from: https://github.com/oschwartz10612/poppler-windows/releases/
  - Extract and configure path in `scripts/config.py`

### Python Dependencies
```
Flask
joblib
pandas
pdf2image
pytesseract
scikit-learn
sklearn-crfsuite
spacy
Pillow
PyMuPDF
```

## 🚀 Setup

### 1. Clone/Download Project
```bash
cd "c:\Users\R I B\Desktop\projet ner"
```

### 2. Create Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
cd scripts
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 4. Configure Paths
Edit `scripts/config.py` and update paths:
```python
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\path\to\poppler-XX.XX.X\Library\bin'
```

Verify paths exist:
```powershell
Test-Path 'C:\Program Files\Tesseract-OCR\tesseract.exe'
Test-Path 'C:\path\to\poppler\Library\bin\pdftoppm.exe'
```

### 5. Train Models (One-time)
```powershell
# Train document classifier
python train_classifier.py

# Train NER model
python train_ner.py
```

This generates:
- `../models/doc_classifier.joblib`
- `../models/tfidf_vectorizer.joblib`
- `../models/ner_crf.joblib`

## 🏃 Running the Application

```powershell
cd scripts
python app.py
```

Server starts at: **http://127.0.0.1:5000**

### Quick Verification
```powershell
# Check health (dependencies)
Invoke-WebRequest http://127.0.0.1:5000/health

# Debug run (test on a sample PDF from data/)
Invoke-WebRequest 'http://127.0.0.1:5000/debug_run?file=Attestation de stage ingenieur.pdf'
```

## 📁 Project Structure

```
projet ner/
├── data/
│   ├── *.pdf                   # Input PDF documents
│   ├── labels.csv              # Document labels for training
│   └── ner_annotations.tsv     # NER annotations (BIO format)
├── models/
│   ├── doc_classifier.joblib   # TF-IDF + LogisticRegression classifier
│   ├── tfidf_vectorizer.joblib # TF-IDF vectorizer
│   └── ner_crf.joblib          # CRF NER model
├── scripts/
│   ├── app.py                  # Flask web app
│   ├── config.py               # Configuration (paths)
│   ├── ocr.py                  # OCR extraction
│   ├── predict.py              # Prediction pipeline
│   ├── train_classifier.py     # Training script (classifier)
│   ├── train_ner.py            # Training script (NER)
│   ├── requirements.txt        # Python dependencies
│   └── __pycache__/
├── templates/
│   ├── index.html              # Upload/analysis page
│   └── results.html            # Results display page
├── uploads/                    # Temporary upload storage
├── logs/
│   └── predict_errors.log      # Prediction error logs
├── output/                     # Results export (optional)
└── README.md
```

## 🔄 Pipeline Flow

```
1. PDF Upload
   ↓
2. OCR Text Extraction (pdf2image + pytesseract)
   ↓
3. Document Classification (TF-IDF + LogisticRegression)
   ↓
4. Named Entity Recognition (spaCy tokenization + CRF)
   ↓
5. Field Extraction (NAME, CIN, DATE, ORGANISATION)
   ↓
6. Validation (regex patterns, field presence checks)
   ↓
7. JSON Export (structured results with metadata)
```

## 📊 API Endpoints

### `GET /`
Home page with upload form.

### `POST /predict`
Upload a PDF and run full pipeline.

**Response (Success):**
```json
{
  "status": "success",
  "document_type": "attestation",
  "extracted_entities": {
    "PER": ["Ahmed Ben Hassen"],
    "ORG": ["QuetraTech", "ISIMA"],
    "DATE": ["2023", "2022"]
  },
  "extracted_fields": {
    "name": "Ahmed Ben Hassen",
    "cin": "12345678",
    "date": "2023",
    "organisation": "QuetraTech"
  },
  "validation": {
    "is_valid": true,
    "errors": [],
    "warnings": ["No date found"]
  },
  "extracted_text": "...",
  "metadata": {
    "filename": "document.pdf",
    "processed_at": "2025-12-12T10:30:45.123456"
  }
}
```

### `GET /health`
Check system dependencies and model files.

```json
{
  "tesseract_cmd": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
  "tesseract_exists": true,
  "poppler_path": "C:\\path\\to\\poppler\\Library\\bin",
  "poppler_exists": true,
  "models_dir": "C:\\...\\models",
  "classifier_exists": true,
  "vectorizer_exists": true,
  "ner_exists": true
}
```

### `GET /debug_run`
Run pipeline on a sample PDF from `data/` (debugging).

**Query Params:**
- `file` (optional): Filename in `data/` (default: first PDF found)

## 🔑 Extracted Fields

| Field | Pattern | Source |
|-------|---------|--------|
| **Name** | Person entities | NER (B-PER, I-PER tags) |
| **CIN** | 8-digit identifier | Regex `\d{8}` |
| **Date** | Temporal references | NER (DATE tags) + regex patterns |
| **Organisation** | Company/institution names | NER (B-ORG, I-ORG tags) |

## ✅ Validation Rules

- **CIN**: Must be exactly 8 digits
- **Date**: Should match French date patterns (DD/MM/YYYY, DD-MM-YYYY, or YYYY)
- **Name**: Should be extracted from document (warning if missing)
- **Organisation**: Should be extracted from document (warning if missing)

## 📝 Training Data Format

### `data/labels.csv`
```csv
filename,label
ahmed.pdf,other
Attestation de stage ingenieur.pdf,attestation
Relevé de notes.pdf,releve
```

### `data/ner_annotations.tsv` (BIO Format)
```
word1 B-PER
word2 I-PER
word3 O
word4 B-ORG
word5 I-ORG

word6 O
word7 B-DATE
...
```

## 🐛 Troubleshooting

### ModuleNotFoundError: No module named 'flask'
```powershell
python -m pip install -r requirements.txt
```

### TemplateNotFound: index.html
Ensure `templates/` folder exists at project root with `index.html` and `results.html`.

### OCR Error: Tesseract not found
Check path in `scripts/config.py`:
```powershell
Test-Path 'C:\Program Files\Tesseract-OCR\tesseract.exe'
& 'C:\Program Files\Tesseract-OCR\tesseract.exe' --version
```

### OCR Error: Poppler not found
Verify Poppler path and test:
```powershell
Test-Path 'C:\path\to\poppler\Library\bin\pdftoppm.exe'
```

### Content Security Policy Error
CSP blocks eval by design. App is set to `debug=False` which removes unsafe scripts.

### 500 Error on /predict
1. Check logs: `logs/predict_errors.log`
2. Run health check: `GET /health`
3. Test debug endpoint: `GET /debug_run`

## 📊 Metrics & Evaluation

- **F1 Score**: Micro/macro on extracted entities
- **Field Accuracy**: % of correctly extracted key fields
- **CER (Character Error Rate)**: OCR quality metric
- **Latency**: End-to-end processing time

## 🔒 Privacy & Security

- Temporary uploaded files are deleted after processing
- Error logs contain minimal PII; can be manually purged
- No external API calls; all processing is local
- Consider encryption for production deployment

## 📌 Notes

- Models are loaded once at app startup (pre-caching)
- Maximum upload size: 16 MB
- Supported format: PDF only
- Language support: French (via spaCy `fr_core_news_sm` fallback to blank model)
- CRF NER handles BIO tag sequences correctly

## 📚 References

- [spaCy NLP Library](https://spacy.io)
- [scikit-learn CRF](https://sklearn-crfsuite.readthedocs.io)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [Poppler PDF Tools](https://poppler.freedesktop.org)
- [Flask Documentation](https://flask.palletsprojects.com)

---

**Created:** December 2025  
**Version:** 1.0 (Bronze - CRF/TF-IDF pipeline)

import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from predict import predict_pipeline, load_models
import traceback
from config import TESSERACT_CMD, POPPLER_PATH
import logging
from datetime import datetime

# Get the absolute path of the directory where this script is located
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
# Define the project root as the parent directory of 'scripts'
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}

# --- App Initialization ---
# Assume templates are in a 'templates' folder at the project root
app = Flask(__name__, template_folder=os.path.join(PROJECT_ROOT, 'templates'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create logs folder and configure logging
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
log_file = os.path.join(LOGS_DIR, 'predict_errors.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Renders the main page with the upload form."""
    return render_template('index.html')


@app.route('/results')
def results():
    """Renders the results page."""
    return render_template('results.html')


@app.route('/health')
def health():
    """Simple health endpoint to check external dependencies and model files."""
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    status = {
        'tesseract_cmd': TESSERACT_CMD,
        'tesseract_exists': os.path.exists(TESSERACT_CMD),
        'poppler_path': POPPLER_PATH,
        'poppler_exists': os.path.exists(POPPLER_PATH),
        'models_dir': models_dir,
        'classifier_exists': os.path.exists(os.path.join(models_dir, 'doc_classifier.joblib')),
        'vectorizer_exists': os.path.exists(os.path.join(models_dir, 'tfidf_vectorizer.joblib')),
        'ner_exists': os.path.exists(os.path.join(models_dir, 'ner_crf.joblib')),
    }
    return jsonify(status)


@app.route('/debug_run')
def debug_run():
    """Run prediction on a local sample PDF from the `data/` directory for debugging.

    Optional query parameter `file` can be provided to select a specific file.
    """
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    filename = request.args.get('file')

    if filename:
        target = os.path.join(data_dir, filename)
        if not os.path.exists(target):
            return jsonify({"error": f"File not found: {filename}"}), 404
    else:
        # pick the first PDF in data/
        pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        if not pdfs:
            return jsonify({"error": "No PDF files found in data/ to debug."}), 404
        target = os.path.join(data_dir, pdfs[0])

    try:
        results = predict_pipeline(target)
        return jsonify({"debug_file": os.path.basename(target), "results": results})
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Debug run failed for %s: %s", target, tb)
        return jsonify({"error": "Debug run failed.", "details": str(e), "traceback": tb}), 500


@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """Handles the file upload and runs the prediction pipeline."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Run the prediction pipeline from predict.py
            print(f"DEBUG: Starting prediction on {filepath}")
            results = predict_pipeline(filepath)
            print(f"DEBUG: Prediction succeeded")
            return jsonify(results), 200
        except Exception as e:
            # Log the full error for debugging
            error_msg = f"An error occurred during prediction: {e}"
            print(error_msg)
            tb = traceback.format_exc()
            print(tb)
            print(f"DEBUG: Full traceback\n{tb}")
            logging.error(f"Prediction failed for {filename}: {tb}")
            # Return error details in development to aid debugging
            return jsonify({
                "status": "error",
                "error": "An internal error occurred during prediction.",
                "details": str(e),
                "traceback": tb
            }), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({"error": "File type not allowed. Please upload a PDF."}), 400


if __name__ == '__main__':
    # Pre-load models on startup for better performance.
    # This avoids the delay on the first request.
    print("Loading models, please wait...")
    load_models()
    print("--- NER/Document Classifier Web App ---")
    print("Navigate to http://127.0.0.1:5000 in your web browser.")
    # Run without the interactive debugger to avoid client-side Content Security
    # Policy errors caused by the Werkzeug debug toolbar (uses eval in the browser).
    # For development with debugger enabled, set the environment variable
    # FLASK_DEBUG=1 and run with the flask CLI instead.
    app.run(debug=False)
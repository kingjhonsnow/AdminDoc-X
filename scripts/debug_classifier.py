from predict import load_models, extract_text_from_pdf
import joblib
import os

# Manually load models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_dir = os.path.join(PROJECT_ROOT, "models")

classifier = joblib.load(os.path.join(models_dir, "doc_classifier.joblib"))
vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))

pdf_path = "../data/ahmed.pdf"
text = extract_text_from_pdf(pdf_path)

# Check classifier prediction
text_vectorized = vectorizer.transform([text])
pred = classifier.predict(text_vectorized)[0]
probs = classifier.predict_proba(text_vectorized)[0]
classes = list(classifier.classes_)
confidence = float(probs.max())

print(f"Classifier predicted: {pred}")
print(f"All classes: {classes}")
print(f"Probabilities: {dict(zip(classes, probs))}")
print(f"Confidence: {confidence}")
print(f"\nHeuristic check:")
print(f"CV heuristic would trigger if: doc_type == 'attestation' (current: {pred == 'attestation'})")

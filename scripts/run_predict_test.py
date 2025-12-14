from predict import predict_pipeline
import json

pdfs = ["../data/ahmed.pdf", "../data/Attestation de stage ingenieur.pdf", "../data/Relevé de notes.pdf"]
for p in pdfs:
    try:
        r = predict_pipeline(p)
        print("---", p)
        print(json.dumps({
            'document_type': r['document_type'],
            'classification_confidence': round(r.get('classification_confidence'), 3) if r.get('classification_confidence') else None,
            'classification_override_reason': r.get('classification_override_reason')
        }, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error processing {p}: {e}")

import os
import joblib
import spacy
import re
import csv
from datetime import datetime
from ocr import extract_text_from_pdf
from train_ner import sent2features

# Define the project root relative to this script's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Global variables to hold loaded models
classifier = None
vectorizer = None
ner_model = None
nlp = None
allowed_labels = None
allowed_labels_map = None

def load_models():
    """Loads all the trained models and the vectorizer."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    
    # Load Document Classifier
    classifier_path = os.path.join(models_dir, "doc_classifier.joblib")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    if not os.path.exists(classifier_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Classifier model or vectorizer not found. Please run train_classifier.py first.")
    
    global classifier, vectorizer, ner_model, nlp
    classifier = joblib.load(classifier_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Load NER Model
    ner_path = os.path.join(models_dir, "ner_crf.joblib")
    if not os.path.exists(ner_path):
        raise FileNotFoundError("NER model not found. Please run train_ner.py first.")
    ner_model = joblib.load(ner_path)

    # Load spaCy model for tokenization
    try:
        nlp = spacy.blank("fr")
    except OSError:
        print("\n--- spaCy French model not found ---")
        print("Please run: python -m spacy download fr_core_news_sm")
        print("Then restart the application.")
        # Using a generic blank model as a fallback for basic tokenization
        nlp = spacy.blank("xx")
    
    # Add sentencizer component if not already present
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

def ensure_models_loaded():
    """Checks if models are loaded and loads them if not."""
    if classifier is None or vectorizer is None or ner_model is None or nlp is None:
        print("Models not loaded yet. Loading now...")
        load_models()
    global allowed_labels, allowed_labels_map
    if allowed_labels is None or allowed_labels_map is None:
        # Load allowed labels from data/labels.csv (unique labels).
        # Build a mapping for case-insensitive matching: lower() -> original label.
        try:
            labels_path = os.path.join(PROJECT_ROOT, 'data', 'labels.csv')
            allowed_labels = set()
            allowed_labels_map = {}
            if os.path.exists(labels_path):
                with open(labels_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        lab = row.get('label') or row.get('Label')
                        if lab:
                            lab_str = lab.strip()
                            allowed_labels.add(lab_str)
                            allowed_labels_map[lab_str.lower()] = lab_str
            else:
                allowed_labels = set()
                allowed_labels_map = {}
        except Exception:
            allowed_labels = set()
            allowed_labels_map = {}


def extract_key_fields(entities_dict, extracted_text):
    """
    Extract key administrative fields from NER results.
    Returns dict with extracted fields and validation status.
    
    Fields:
      - NAME: B-PER, I-PER tags
      - CIN: Moroccan ID (8 digits pattern)
      - DATE: B-DATE, I-DATE tags
      - ORGANISATION: B-ORG, I-ORG tags
    """
    fields = {
        'name': None,
        'cin': None,
        'date': None,
        'organisation': None,
        'all_entities': entities_dict
    }
    
    # Extract NAME from PER entities
    if 'PER' in entities_dict:
        names = entities_dict['PER']
        if names:
            fields['name'] = names[0]  # Take first person mentioned
    
    # Extract ORGANISATION
    if 'ORG' in entities_dict:
        orgs = entities_dict['ORG']
        if orgs:
            fields['organisation'] = orgs[0]
    
    # Extract DATE
    if 'DATE' in entities_dict:
        dates = entities_dict['DATE']
        if dates:
            fields['date'] = dates[0]
    
    # Try to find CIN via regex in raw text (8 digits pattern)
    cin_pattern = r'\b\d{8}\b'
    cin_matches = re.findall(cin_pattern, extracted_text)
    if cin_matches:
        fields['cin'] = cin_matches[0]
    
    return fields


def validate_fields(fields):
    """
    Validate extracted fields according to business rules.
    Returns validation report.
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # CIN validation: should be 8 digits
    if fields.get('cin'):
        if not re.match(r'^\d{8}$', fields['cin']):
            validation['errors'].append(f"Invalid CIN format: {fields['cin']} (expected 8 digits)")
            validation['is_valid'] = False
    else:
        validation['warnings'].append("No CIN found")
    
    # DATE validation: should be parseable
    if fields.get('date'):
        date_str = fields['date']
        # Try common French date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'\d{4}',
        ]
        if not any(re.search(pattern, date_str) for pattern in date_patterns):
            validation['warnings'].append(f"Date format unclear: {date_str}")
    else:
        validation['warnings'].append("No date found")
    
    # NAME validation: should exist
    if not fields.get('name'):
        validation['warnings'].append("No name found")
    
    # ORGANISATION validation: should exist
    if not fields.get('organisation'):
        validation['warnings'].append("No organisation found")
    
    return validation

def predict_pipeline(pdf_path):
    """
    Runs the full prediction pipeline on a single PDF document.
    
    Steps:
    1. Extracts text using OCR.
    2. Classifies the document type.
    3. Extracts named entities.
    4. Extracts key administrative fields.
    5. Validates extracted fields.
    6. Returns structured JSON result.
    """
    # Ensure models are loaded before proceeding
    ensure_models_loaded()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found at {pdf_path}")

    print(f"--- Processing: {os.path.basename(pdf_path)} ---")

    # Step 1: OCR
    print("Step 1: Extracting text with OCR...")
    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        raise RuntimeError(f"OCR step failed: {e}")

    # Check for empty text after OCR
    if not text or text.isspace():
        raise RuntimeError("OCR produced no text. The document may be empty or unreadable.")

    # Step 2: Classify Document
    print("Step 2: Classifying document type...")
    try:
        text_vectorized = vectorizer.transform([text])

        # If classifier supports probability estimates, use them for confidence
        confidence = None
        override_reason = None
        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(text_vectorized)[0]
            classes = list(classifier.classes_)
            max_idx = int(probs.argmax())
            doc_type = classes[max_idx]
            confidence = float(probs[max_idx])
        else:
            doc_type = classifier.predict(text_vectorized)[0]

        # Heuristic: detect if the document looks like a CV/resume (stronger)
        def looks_like_cv(t: str, filename: str = "") -> bool:
            # keywords and section headings commonly present in CVs/resumes
            keywords = [
                r"\bcv\b", r"curriculum vitae", r"\brésum[eé]\b", r"\bresume\b",
                r"\bexperience\b", r"\beducation\b", r"\bwork experience\b",
                r"\blinkedin\b", r"\bgithub\b", r"\bprojects?\b", r"\bskills?\b",
                r"\bobjective\b", r"\babout me\b", r"\bprofile\b", r"\bcertificat|certificate\b",
                r"\bcontact\b", r"\btelephone\b", r"\btel\b", r"\bemail\b", r"\baddress\b"
            ]
            text_low = t.lower()
            hits = 0
            for k in keywords:
                try:
                    if re.search(k, text_low):
                        hits += 1
                except re.error:
                    if k in text_low:
                        hits += 1

            # filename hints (e.g., CV_ or resume)
            fname = filename.lower() if filename else ""
            if "cv" in fname or "resume" in fname or "résumé" in fname:
                hits += 2

            # require >=3 distinct hits to decide it's a CV (reduces false positives)
            return hits >= 3

        # Heuristic: detect if the document is an attestation (opposite of CV)
        def looks_like_attestation(t: str, filename: str = "") -> bool:
            # Keywords and phrases commonly in attestations/certificates
            keywords = [
                r"\batt[eé]st[eé]\b", r"\battestations?\b", r"\bcertifi[eé]\b", r"\bcertificat\b",
                r"\bfait \u00e0\b", r"\bsignature\b", r"\bsous-sign[eé]s?\b",
                r"\battestent que\b", r"\battestons que\b", r"\bon atteste\b",
                r"\b[àa] qui de droit\b", r"\bvis-\u00e0-vis\b", r"\bstage\b",
                r"\bdur[eé]e du stage\b", r"\bduration\b"
            ]
            text_low = t.lower()
            hits = 0
            for k in keywords:
                try:
                    if re.search(k, text_low):
                        hits += 1
                except re.error:
                    if k in text_low:
                        hits += 1
            
            # filename hints
            fname = filename.lower() if filename else ""
            if "attestation" in fname or "attestat" in fname:
                hits += 2

            # require >=2 hits to boost attestation
            return hits >= 2

        # Heuristic: detect if the document is a "relevé de notes" (academic transcript)
        def looks_like_releve(t: str, filename: str = "") -> bool:
            keywords = [
                r"\brelev[eé]\b", r"\brelev[eé] de notes\b", r"\brelev[eé]s?\b",
                r"\bnotes?\b", r"\bbulletin\b", r"\bmoyenne\b", r"\bsemestre\b",
                r"\bdipl[oô]me\b", r"\bmention\b", r"\bgrade\b", r"\bgrades?\b",
                r"\buniversit[eé]\b", r"\bfacult[eé]\b", r"\betudiant\b", r"\bstudent\b"
            ]
            text_low = t.lower()
            hits = 0
            for k in keywords:
                try:
                    if re.search(k, text_low):
                        hits += 1
                except re.error:
                    if k in text_low:
                        hits += 1

            # filename hints
            fname = filename.lower() if filename else ""
            if "relev" in fname or "notes" in fname or "bulletin" in fname:
                hits += 2

            # require >=2 hits to consider it a releve
            return hits >= 2

        # Heuristic: detect if the document looks like a presentation/slides
        def looks_like_presentation(t: str, filename: str = "") -> bool:
            # Keywords commonly present in presentation slides or PDFs exported from PowerPoint
            keywords = [
                r"\bpresentation\b", r"\bslide(s)?\b", r"\bdiapositive(s)?\b",
                r"\bpowerpoint\b", r"\bpptx?\b", r"\bprésentation\b", r"\bsommaire\b",
                r"\bplan\b", r"\bobjectifs\b", r"\bconclusion\b", r"\bintroducti(on)?\b",
                r"\bbullet\b", r"\bpoints?\b", r"\bagenda\b"
            ]
            text_low = t.lower()
            hits = 0
            for k in keywords:
                try:
                    if re.search(k, text_low):
                        hits += 1
                except re.error:
                    if k in text_low:
                        hits += 1

            # filename hints
            fname = filename.lower() if filename else ""
            if fname.endswith('.ppt') or fname.endswith('.pptx') or 'presentation' in fname or 'slide' in fname:
                hits += 2

            # require >=2 hits to consider it a presentation (relatively permissive)
            return hits >= 2

        # Save the raw predicted label and probability distribution before heuristics
        try:
            raw_predicted_label = doc_type
        except Exception:
            raw_predicted_label = None
        try:
            predicted_probabilities = {classes[i]: float(probs[i]) for i in range(len(classes))} if 'probs' in locals() and 'classes' in locals() else None
        except Exception:
            predicted_probabilities = None

        # Apply heuristics: CV/attestation/presentation/releve checks
        cv_score = looks_like_cv(text, os.path.basename(pdf_path))
        attestation_score = looks_like_attestation(text, os.path.basename(pdf_path))
        presentation_score = looks_like_presentation(text, os.path.basename(pdf_path))
        releve_score = looks_like_releve(text, os.path.basename(pdf_path))

        # Debug: print heuristic scores to help trace classification issues
        try:
            print(f"Heuristics -> cv:{cv_score}, attestation:{attestation_score}, presentation:{presentation_score}, releve:{releve_score}")
        except Exception:
            pass

        # If CV heuristic matches classifier prediction, boost confidence
        if cv_score and doc_type == "CV":
            override_reason = "cv_heuristic_confirmed"
            # Boost confidence so low_confidence override doesn't trigger
            if confidence is not None:
                confidence = max(confidence, 0.75)
        # If attestation heuristic matches classifier prediction, boost confidence
        elif attestation_score and doc_type == "attestation":
            override_reason = "attestation_heuristic_confirmed"
            # Boost confidence
            if confidence is not None:
                confidence = max(confidence, 0.75)
        # If presentation heuristic matches classifier prediction, boost confidence
        elif presentation_score and doc_type == "presentation":
            override_reason = "presentation_heuristic_confirmed"
            if confidence is not None:
                confidence = max(confidence, 0.75)
        # If releve heuristic matches classifier prediction, boost confidence
        elif releve_score and doc_type == "releve":
            override_reason = "releve_heuristic_confirmed"
            if confidence is not None:
                confidence = max(confidence, 0.75)
        # If CV heuristic strongly matches but classifier predicted attestation, override to CV
        elif cv_score and doc_type == "attestation":
            override_reason = "cv_heuristic"
            doc_type = "CV"
        # If attestation heuristic strongly matches but classifier predicted CV, override to attestation
        elif attestation_score and doc_type == "CV":
            override_reason = "attestation_heuristic"
            doc_type = "attestation"
        # If presentation heuristic strongly matches but classifier predicted something else, override to presentation
        elif presentation_score and doc_type != "presentation":
            override_reason = "presentation_heuristic"
            doc_type = "presentation"
        # If releve heuristic strongly matches but classifier predicted something else, override to releve
        elif releve_score and doc_type != "releve":
            override_reason = "releve_heuristic"
            doc_type = "releve"

        # If confidence is available and is low, only fallback to 'other' if NO heuristic match
        # (heuristics take precedence over low confidence)
        if confidence is not None and confidence < 0.55 and not override_reason:
            # Quick text-based fallback for known keywords (handle OCR variations)
            text_low = text.lower()
            if 'relev' in text_low or 'relevé' in text_low or 'releve de notes' in text_low or releve_score:
                override_reason = 'releve_heuristic_low_confidence'
                doc_type = 'releve'
            else:
                override_reason = "low_confidence"
                doc_type = "other"
        elif confidence is not None and confidence < 0.55 and override_reason and "confirmed" not in override_reason:
            # Low confidence but heuristic matched, just append low_confidence note
            override_reason = override_reason + " low_confidence"
        # Ensure final document type is one of the allowed labels (from data/labels.csv).
        # Strict policy: if the final `doc_type` is not in `allowed_labels` then
        # - allow a heuristic-driven mapping to an allowed label (if a heuristic clearly matched),
        # - otherwise set the type to 'other'. Do NOT silently choose the highest-probability
        #   allowed class — that previously caused unexpected mappings.
        try:
            global allowed_labels
            if allowed_labels:
                if doc_type not in allowed_labels:
                    mapped = None
                    mapped_reason = None
                    # Only map via a heuristic if the mapped label is present in allowed_labels
                    if 'CV' in allowed_labels and cv_score:
                        mapped = 'CV'
                        mapped_reason = 'cv_heuristic_restricted'
                    elif 'attestation' in allowed_labels and attestation_score:
                        mapped = 'attestation'
                        mapped_reason = 'attestation_heuristic_restricted'
                    elif 'releve' in allowed_labels and releve_score:
                        mapped = 'releve'
                        mapped_reason = 'releve_heuristic_restricted'

                    if mapped:
                        # Use original-cased label from allowed_labels_map if available
                        try:
                            mapped_final = allowed_labels_map.get(mapped.lower(), mapped)
                        except Exception:
                            mapped_final = mapped
                        doc_type = mapped_final
                        override_reason = (override_reason + ' ' + mapped_reason) if override_reason else mapped_reason
                    else:
                        # No allowed label could be confidently selected — mark as 'other'
                        override_reason = (override_reason + ' not_in_allowed_labels') if override_reason else 'not_in_allowed_labels'
                        doc_type = 'other'
                        # Clear confidence when we force 'other' to avoid misleading scores
                        confidence = None
        except Exception:
            pass

    except Exception as e:
        raise RuntimeError(f"Classification step failed: {e}")

    # Step 3: Extract Entities (NER)
    print("Step 3: Extracting named entities...")
    try:
        # Tokenize text into sentences and words using spaCy
        doc = nlp(text)

        # Build list of sentences with (word, 'O') placeholder tags
        sentences = [[(token.text, 'O') for token in sent] for sent in doc.sents]

        # Extract features and predict
        X_test = [sent2features(s) for s in sentences]
        y_pred = ner_model.predict(X_test)

        # Combine words with predicted tags into entity groups
        entities = {}
        for i, sent in enumerate(sentences):
            current_entity_words = []
            current_entity_type = None
            for j, (word, _) in enumerate(sent):
                tag = y_pred[i][j]
                if tag.startswith('B-'):  # Begin new entity
                    if current_entity_words:
                        entities.setdefault(current_entity_type, []).append(" ".join(current_entity_words))
                    current_entity_words = [word]
                    current_entity_type = tag.split('-')[1]
                elif tag.startswith('I-') and current_entity_type == tag.split('-')[1]:  # Continue entity
                    current_entity_words.append(word)
                else:  # 'O' tag or tag mismatch - end current entity
                    if current_entity_words:
                        entities.setdefault(current_entity_type, []).append(" ".join(current_entity_words))
                    current_entity_words = []
                    current_entity_type = None
            
            # Handle lingering entity at end of sentence
            if current_entity_words:
                entities.setdefault(current_entity_type, []).append(" ".join(current_entity_words))
    except Exception as e:
        raise RuntimeError(f"NER step failed: {e}")

    # Step 4: Extract Key Fields
    print("Step 4: Extracting key administrative fields...")
    try:
        fields = extract_key_fields(entities, text)
    except Exception as e:
        raise RuntimeError(f"Field extraction step failed: {e}")

    # Step 5: Validate Fields
    print("Step 5: Validating extracted fields...")
    try:
        validation = validate_fields(fields)
    except Exception as e:
        raise RuntimeError(f"Validation step failed: {e}")

    # Step 6: Return structured result
    result = {
        "status": "success",
        "document_type": doc_type,
        "raw_predicted_label": raw_predicted_label if 'raw_predicted_label' in locals() else None,
        "classification_confidence": confidence if 'confidence' in locals() else None,
        "classification_override_reason": override_reason if 'override_reason' in locals() else None,
        "predicted_probabilities": predicted_probabilities,
        "heuristics": {
            "cv": bool(cv_score) if 'cv_score' in locals() else None,
            "attestation": bool(attestation_score) if 'attestation_score' in locals() else None,
            "presentation": bool(presentation_score) if 'presentation_score' in locals() else None,
            "releve": bool(releve_score) if 'releve_score' in locals() else None
        },
        "allowed_labels": sorted(list(allowed_labels)) if allowed_labels else [],
        "extracted_entities": entities,
        "extracted_fields": {
            "name": fields['name'],
            "cin": fields['cin'],
            "date": fields['date'],
            "organisation": fields['organisation']
        },
        "validation": validation,
        "extracted_text": text,
        "metadata": {
            "filename": os.path.basename(pdf_path),
            "processed_at": datetime.now().isoformat()
        }
    }

    return result

if __name__ == '__main__':
    # The main block is now for direct script testing only.
    
    # --- Test with a document ---
    # Make sure this file exists and is a valid PDF
    # Place the PDF you want to test in the 'data' directory and update the filename below.
    test_pdf_filename = "Relevé de notes.pdf"
    test_pdf = os.path.join("..", "data", test_pdf_filename)
    results = predict_pipeline(test_pdf)
    print("\n--- Prediction Results ---")
    import json
    print(json.dumps(results, indent=2, ensure_ascii=False))
import pandas as pd
import os
import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Define the project root relative to this script's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def read_ner_data(file_path):
    """
    Reads the NER data from a TSV file.
    Each line is expected to have a word and its tag, separated by a space.
    Sentences are separated by empty lines.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    sentence = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 2:
                sentence.append((parts[0], parts[1]))
            else:
                # Handle cases where a line might be malformed
                sentence.append((line, 'O'))
        else:
            if sentence:
                sentences.append(sentence)
                sentence = []
    if sentence: # Add the last sentence if the file doesn't end with a newline
        sentences.append(sentence)
    return sentences

def word2features(sent, i):
    """
    Extracts features for a word in a sentence.
    """
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
                
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def train_ner_model():
    """
    Trains a Named Entity Recognition (NER) model.
    """
    # 1. Load data
    ner_data_path = os.path.join(PROJECT_ROOT, "data", "ner_annotations.tsv")
    if not os.path.exists(ner_data_path):
        print(f"Error: ner_annotations.tsv not found at {ner_data_path}")
        return

    print("Loading NER data...")
    sentences = read_ner_data(ner_data_path)
    
    if not sentences:
        print("Warning: No data found in the annotation file.")
        return

    # 2. Prepare data for training
    print("Preparing data for training...")
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    
    # Note: With more data, we would do a train-test split
    # For now, we train on the whole dataset
    X_train, y_train = X, y

    # 3. Train the CRF model
    print("\nTraining NER model...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    print("Model training complete.")

    # -- Evaluation ---
    # With a test set, we would evaluate properly.
    # For now, let's see the performance on the training data.
    labels = list(crf.classes_)
    if 'O' in labels:
        labels.remove('O') # We don't care about 'O' in the report
    
    y_pred = crf.predict(X_train)
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print("\nClassification Report (on training data):")
    if not sorted_labels:
        print("No labels to report on other than 'O'.")
    else:
        print(metrics.flat_classification_report(
            y_train, y_pred, labels=sorted_labels, digits=3
        ))

    # 4. Save the model
    print("\nSaving model...")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    joblib.dump(crf, os.path.join(models_dir, "ner_crf.joblib"))
    print(f"Model saved successfully in the '{models_dir}' directory.")


if __name__ == '__main__':
    train_ner_model()

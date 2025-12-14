import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Import the OCR function from our other script
from ocr import extract_text_from_pdf

# Define the project root relative to this script's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_text_from_file(file_path):
    """
    Extracts text from a file, handling PDF and TXT formats.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}. Skipping.")
        return ""
        
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"Warning: Unsupported file type: {file_path}. Skipping.")
        return ""

def train_classification_model():
    """
    Trains a document classification model.
    """
    # 1. Load labels
    labels_path = os.path.join(PROJECT_ROOT, "data", "labels.csv")
    if not os.path.exists(labels_path):
        print(f"Error: labels.csv not found at {labels_path}")
        return
    
    df = pd.read_csv(labels_path)
    print("Loaded labels:")
    print(df)

    # --- Note on data size ---
    if len(df) < 2:
        print("\nWarning: Not enough data to train a meaningful model. Need at least 2 documents.")
        print("Please add more documents to the 'data' directory and update 'data/labels.csv'.")
        return

    # 2. Extract text for each document
    import pandas as pd
    import os
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    # Import the OCR function from our other script
    from ocr import extract_text_from_pdf

    # Define the project root relative to this script's location
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


    def get_text_from_file(file_path):
        """
        Extracts text from a file, handling PDF and TXT formats.
        """
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}. Skipping.")
            return ""

        if file_path.lower().endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Unsupported file type: {file_path}. Skipping.")
            return ""


    def train_classification_model():
        """
        Trains a document classification model.
        """
        # 1. Load labels
        labels_path = os.path.join(PROJECT_ROOT, "data", "labels.csv")
        if not os.path.exists(labels_path):
            print(f"Error: labels.csv not found at {labels_path}")
            return

        df = pd.read_csv(labels_path)
        print("Loaded labels:")
        print(df)

        # --- Note on data size ---
        if len(df) < 2:
            print("\nWarning: Not enough data to train a meaningful model. Need at least 2 documents.")
            print("Please add more documents to the 'data' directory and update 'data/labels.csv'.")
            return

        # 2. Extract text for each document
        print("\nExtracting text from documents...")
        df['text'] = df['filename'].apply(lambda f: get_text_from_file(os.path.join(PROJECT_ROOT, "data", f)))

        # Filter out any rows where text extraction failed
        df = df[df['text'] != ""]

        if len(df) < 2:
            print("\nWarning: Not enough data to train a meaningful model after processing files.")
            return

        X = df['text']
        y = df['label']

        # 3. Create TF-IDF features with word n-grams
        print("\nVectorizing text...")
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            analyzer='word',
            lowercase=True,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        X_tfidf = vectorizer.fit_transform(X)

        # 4. Train the model - use ALL data since we only have a few samples
        # This ensures all classes are represented in the training set
        print("\nTraining classification model...")
        print(f"Using all {len(y)} samples for training")
        print(f"Classes: {sorted(y.unique())}")

        model = LogisticRegression(
            C=0.1,
            max_iter=500,
            solver='lbfgs',
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_tfidf, y)
        print("Model training complete.")

        # --- Evaluation (on training data) ---
        y_pred = model.predict(X_tfidf)
        print("\nClassification Report (on training data):")
        print(classification_report(y, y_pred, zero_division=0))
        print(f"\nModel classes learned: {sorted(model.classes_)}")

        # 5. Save the model and vectorizer
        print("\nSaving model and vectorizer...")
        models_dir = os.path.join(PROJECT_ROOT, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        joblib.dump(model, os.path.join(models_dir, "doc_classifier.joblib"))
        joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
        print(f"Model and vectorizer saved successfully in the '{models_dir}' directory.")


    if __name__ == '__main__':
        train_classification_model()
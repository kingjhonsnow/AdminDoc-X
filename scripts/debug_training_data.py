import pandas as pd
import os
from ocr import extract_text_from_pdf

# Load labels
labels_path = os.path.join('..', 'data', 'labels.csv')
df = pd.read_csv(labels_path)
print('Labels CSV:')
print(df)
print('\nClass distribution:')
print(df['label'].value_counts())

# Try extracting text to see what works
print('\nExtracting text from each file:')
for idx, row in df.iterrows():
    file_path = os.path.join('..', 'data', row['filename'])
    try:
        text = extract_text_from_pdf(file_path)
        status = 'OK' if text and text.strip() else 'EMPTY'
        text_len = len(text) if text else 0
        print(f"{row['filename']}: {row['label']} -> {status} ({text_len} chars)")
    except Exception as e:
        print(f"{row['filename']}: ERROR - {e}")

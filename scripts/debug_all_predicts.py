import os
import json
from predict import predict_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def main():
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    results = {}
    for p in pdfs:
        path = os.path.join(DATA_DIR, p)
        try:
            res = predict_pipeline(path)
        except Exception as e:
            res = {'status': 'error', 'error': str(e)}
        results[p] = res
        print(json.dumps({p: res}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

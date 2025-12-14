#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end test script for the AdminDoc-X pipeline.
Tests OCR -> Classification -> NER -> Field Extraction -> Validation.
"""

import os
import sys
import json
from pathlib import Path

# Add scripts to path
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, SCRIPT_DIR)

from predict import predict_pipeline, load_models

def test_pipeline(pdf_path):
    """Test the full pipeline on a single PDF."""
    print("\n" + "="*70)
    print(f"TESTING PIPELINE ON: {os.path.basename(pdf_path)}")
    print("="*70)
    
    try:
        # Run the pipeline
        result = predict_pipeline(pdf_path)
        
        # Pretty print the result
        print("\n✓ PIPELINE SUCCESS\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return result
        
    except Exception as e:
        print(f"\n✗ PIPELINE FAILED\n")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Change to scripts directory
    os.chdir(SCRIPT_DIR)
    
    # Preload models
    print("Loading models...")
    load_models()
    print("✓ Models loaded\n")
    
    # Test on available PDFs in data/
    data_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
    if not os.path.exists(data_dir):
        print(f"Error: data directory not found at {data_dir}")
        sys.exit(1)
    
    pdfs = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')])
    
    if not pdfs:
        print(f"No PDFs found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdfs)} PDF(s) to test:\n")
    for i, pdf in enumerate(pdfs, 1):
        print(f"  {i}. {pdf}")
    
    # Test each PDF
    results = {}
    for pdf_name in pdfs:
        pdf_path = os.path.join(data_dir, pdf_name)
        result = test_pipeline(pdf_path)
        if result:
            results[pdf_name] = result
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Successfully processed: {len(results)}/{len(pdfs)} PDFs\n")
    
    for pdf_name, result in results.items():
        print(f"✓ {pdf_name}")
        print(f"  Document Type: {result.get('document_type', 'N/A')}")
        print(f"  Fields Extracted:")
        fields = result.get('extracted_fields', {})
        print(f"    - Name: {fields.get('name', 'N/A')}")
        print(f"    - CIN: {fields.get('cin', 'N/A')}")
        print(f"    - Date: {fields.get('date', 'N/A')}")
        print(f"    - Organisation: {fields.get('organisation', 'N/A')}")
        validation = result.get('validation', {})
        print(f"  Validation: {'✓ Valid' if validation.get('is_valid') else '⚠ Has Issues'}")
        if validation.get('errors'):
            for err in validation['errors']:
                print(f"    ✗ {err}")
        if validation.get('warnings'):
            for warn in validation['warnings']:
                print(f"    ⚠ {warn}")
        print()
    
    print("="*70)
    print("✓ Test complete!")
    print("="*70)

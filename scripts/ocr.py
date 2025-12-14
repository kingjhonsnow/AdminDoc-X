from pdf2image import convert_from_path
import pytesseract
import os
from config import TESSERACT_CMD, POPPLER_PATH

# Set the Tesseract command path from the config file
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using OCR.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    try:
        # Use the POPPLER_PATH from the config file
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        full_text = ""
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            text = pytesseract.image_to_string(image)
            full_text += text + "\n"
        return full_text
    except Exception as e: # Catching a broad exception to wrap it
        # If pdf2image failed (corrupt or non-standard PDF), try Python-based text extraction
        original_err = e
        # Try PyPDF2 text extraction as a fallback (if available)
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text_parts = []
                for p in reader.pages:
                    try:
                        t = p.extract_text() or ''
                    except Exception:
                        t = ''
                    text_parts.append(t)
                extracted = "\n".join(text_parts).strip()
                if extracted:
                    return extracted
        except Exception:
            # PyPDF2 not available or failed; continue to other fallbacks
            pass

        # Try PyMuPDF (fitz) if installed
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            extracted = "\n".join(text_parts).strip()
            if extracted:
                return extracted
        except Exception:
            pass

        # If we reach here, no fallback succeeded. Raise a helpful error.
        help_msg = (
            f"Failed to process PDF with OCR. Original error: {original_err}\n"
            "Possible causes: the file is corrupted, not a PDF, or uses an unsupported PDF feature.\n"
            "Suggested actions: 1) Recreate or re-export the PDF as a standard PDF (PDF/A recommended).\n"
            "2) If it's a scanned image, ensure it's a true PDF containing images; try opening and re-saving in a PDF editor.\n"
            "3) Install optional dependencies: `pip install PyPDF2 pymupdf` to enable text-extraction fallbacks.\n"
            "4) If you prefer, upload the original image files (PNG/JPEG) instead of a PDF.\n"
        )
        raise RuntimeError(help_msg)

if __name__ == '__main__':
    # --- Instructions for use ---
    # 1. Place your PDF files in the 'data' directory.
    # 2. Make sure the filename is correct below.

    pdf_file = os.path.join("data", "Attestation de stage ingenieur.pdf")

    extracted_text = extract_text_from_pdf(pdf_file)

    print("\n--- Extracted Text ---")
    print(extracted_text)

    # --- To save the output to a file ---
    # output_dir = "output"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # output_filepath = os.path.join(output_dir, os.path.basename(pdf_file).replace('.pdf', '.txt'))
    # with open(output_filepath, 'w', encoding='utf-8') as f:
    #     f.write(extracted_text)
    # print(f"\nExtracted text saved to: {output_filepath}")
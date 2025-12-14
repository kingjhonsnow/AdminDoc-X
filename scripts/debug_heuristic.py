from predict import load_models, extract_text_from_pdf
import re
import os

# Load text from the CV
pdf_path = "../data/ahmed.pdf"
text = extract_text_from_pdf(pdf_path)

# Run the CV heuristic
keywords = [
    r"\bcv\b", r"curriculum vitae", r"\brésum[eé]\b", r"\bresume\b",
    r"\bexperience\b", r"\beducation\b", r"\bwork experience\b",
    r"\blinkedin\b", r"\bgithub\b", r"\bprojects?\b", r"\bskills?\b",
    r"\bobjective\b", r"\babout me\b", r"\bprofile\b", r"\bcertificat|certificate\b",
    r"\bcontact\b", r"\btelephone\b", r"\btel\b", r"\bemail\b", r"\baddress\b"
]

text_low = text.lower()
hits = 0
matched_keywords = []
for k in keywords:
    try:
        if re.search(k, text_low):
            hits += 1
            matched_keywords.append(k)
    except re.error:
        if k in text_low:
            hits += 1
            matched_keywords.append(k)

# filename hints
fname = os.path.basename(pdf_path).lower()
if "cv" in fname or "resume" in fname or "résumé" in fname:
    hits += 2
    print(f"Filename '{fname}' contains CV hint: +2")

print(f"Total CV keyword hits: {hits}")
print(f"Matched keywords: {matched_keywords}")
print(f"Is CV? {hits >= 3}")

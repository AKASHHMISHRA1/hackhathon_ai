from pypdf import PdfReader

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

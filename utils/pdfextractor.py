from pypdf import PdfReader

def text_extractor(pdf_path):
    try:
        pdf_reader = PdfReader(pdf_path)
        pdf_text = []
        for page in pdf_reader.pages:
            text_only = page.extract_text()
            if text_only:
                pdf_text.append(text_only)

        return "\n".join(pdf_text)

    except Exception as e:
        print(e)
        return None
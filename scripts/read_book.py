import PyPDF2
import spacy

nlp = spacy.load("pt_core_news_sm")
nlp.max_length = 2000000


class BookReader:
    def __init__(self, path):
        self.path = path
        
    def read_book(self):
        with open(self.path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
                
            doc = nlp(text)

            marvin_falas = []
            for ent in doc.ents:
                if ent.label_ == "PER" and "marvin" in ent.text.lower():
                    marvin_falas.append(ent.sent.text)
                    
            return "\n".join(marvin_falas)

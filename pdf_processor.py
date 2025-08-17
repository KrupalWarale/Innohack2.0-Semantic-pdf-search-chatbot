import fitz  # PyMuPDF
import io
import re
from typing import List, Dict

class PDFProcessor:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_data) -> str:
        """Extract text from PDF file or bytes using PyMuPDF"""
        try:
            # Handle both file objects and byte data
            if hasattr(pdf_data, 'read'):
                pdf_bytes = pdf_data.read()
            else:
                # pdf_data is already bytes
                pdf_bytes = pdf_data
            
            # Open PDF with PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            doc.close()
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better processing"""
        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences using regex
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def split_into_chunks(self, text: str, chunk_size: int = 2000) -> List[str]:
        """Split text into manageable chunks for processing"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def get_text_with_positions(self, text: str) -> Dict:
        """Get text with character positions for highlighting"""
        sentences = self.split_into_sentences(text)
        sentence_positions = []
        
        current_pos = 0
        for sentence in sentences:
            start_pos = text.find(sentence, current_pos)
            if start_pos != -1:
                end_pos = start_pos + len(sentence)
                sentence_positions.append({
                    'text': sentence,
                    'start': start_pos,
                    'end': end_pos
                })
                current_pos = end_pos
        
        return {
            'full_text': text,
            'sentences': sentence_positions
        }

import fitz  # PyMuPDF
from typing import List
import re

class PDFHighlighter:
    def clean_text(self, text: str) -> str:
        """Clean text for better matching"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def highlight_text_in_pdf(self, pdf_bytes: bytes, sentences_to_highlight: List[str]) -> bytes:
        """
        Opens a PDF from bytes, adds highlights to specified sentences, and returns the new PDF as bytes.
        """
        if not sentences_to_highlight:
            return pdf_bytes

        try:
            # Open the PDF from bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            highlights_added = 0

            # Clean and sort sentences by length (longest first) to avoid partial matches
            cleaned_sentences = [self.clean_text(s) for s in sentences_to_highlight if s.strip()]
            sorted_sentences = sorted(cleaned_sentences, key=len, reverse=True)

            # Iterate through each page and highlight the sentences
            for page_num, page in enumerate(doc):
                for sentence in sorted_sentences:
                    if sentence:
                        # Try exact search first
                        text_instances = page.search_for(sentence, quads=True)
                        
                        # If exact search fails, try with variations
                        if not text_instances:
                            # Try without page markers
                            cleaned_sentence = re.sub(r'--- Page \d+ ---', '', sentence).strip()
                            if cleaned_sentence:
                                text_instances = page.search_for(cleaned_sentence, quads=True)
                        
                        # If still no match, try searching for parts of the sentence
                        if not text_instances and len(sentence.split()) > 5:
                            # Split long sentences and try to find substantial parts
                            words = sentence.split()
                            for i in range(0, len(words), 5):
                                chunk = ' '.join(words[i:i+8])  # 8-word chunks with overlap
                                if len(chunk.strip()) > 20:  # Only search meaningful chunks
                                    chunk_instances = page.search_for(chunk, quads=True)
                                    text_instances.extend(chunk_instances)
                        
                        # Highlight all found instances
                        for inst in text_instances:
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=(1, 1, 0))  # Yellow color
                            highlight.update()
                            highlights_added += 1

            print(f"Added {highlights_added} highlights across {len(doc)} pages")
            
            # Save the modified PDF to a new byte string
            output_pdf_bytes = doc.tobytes()
            doc.close()
            
            return output_pdf_bytes

        except Exception as e:
            print(f"Error during PDF highlighting: {e}")
            # If highlighting fails, return the original PDF to avoid crashing
            return pdf_bytes

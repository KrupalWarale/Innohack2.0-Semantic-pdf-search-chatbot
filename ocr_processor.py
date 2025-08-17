import os
import json
import fitz  # PyMuPDF
import tempfile
import io
import google.generativeai as genai
from PIL import Image
import base64

class OCRProcessor:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.use_paddle = False  # Force use of Gemini API for better results
        
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        else:
            print("No API key provided for OCR processing")

    def process_pdf_bytes(self, pdf_bytes):
        """Process PDF from bytes data"""
        try:
            # Handle bytes data
            if isinstance(pdf_bytes, bytes):
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            else:
                doc = fitz.open(pdf_bytes)
            
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                img_data = pix.tobytes("png")
                
                # Process with OCR
                if self.use_paddle:
                    ocr_result = self._process_with_paddle(img_data)
                else:
                    ocr_result = self._process_with_gemini(img_data)
                
                pages_data.append({
                    "page_number": page_num + 1,
                    "ocr_result": ocr_result,
                    "extracted_text": self._extract_text_from_ocr(ocr_result)
                })
            
            doc.close()
            return pages_data
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def _process_with_paddle(self, img_data):
        """Process image with PaddleOCR"""
        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(img_data)
                tmp_path = tmp_file.name
            
            # Run OCR
            result = self.ocr.ocr(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            return result
        except Exception as e:
            print(f"PaddleOCR failed: {e}")
            return []

    def _process_with_gemini(self, img_data):
        """Process image with Gemini API"""
        try:
            # Convert image data to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Use Gemini to extract text
            prompt = "Extract all text from this image. Return only the text content, maintaining the original structure and formatting as much as possible."
            response = self.gemini_model.generate_content([prompt, image])
            
            # Format as PaddleOCR-like structure
            text = response.text.strip()
            return [[[[0, 0], [100, 0], [100, 20], [0, 20]], (text, 0.9)]] if text else []
            
        except Exception as e:
            print(f"Gemini OCR failed: {e}")
            return []

    def _extract_text_from_ocr(self, ocr_result):
        """Extract plain text from OCR results"""
        text_parts = []
        
        if ocr_result:
            for line in ocr_result:
                if isinstance(line, list) and len(line) >= 2:
                    if isinstance(line[1], tuple) and len(line[1]) >= 1:
                        text_parts.append(line[1][0])
                    elif isinstance(line[1], str):
                        text_parts.append(line[1])
        
        return "\n".join(text_parts)

    def save_ocr_results(self, pages_data, pdf_output_path, json_output_path):
        """Save OCR results to JSON and create searchable PDF"""
        try:
            # Prepare JSON data
            json_data = {
                "total_pages": len(pages_data),
                "extraction_method": "PaddleOCR" if self.use_paddle else "Gemini",
                "pages": [],
                "full_text": ""
            }
            
            full_text_parts = []
            
            for page_data in pages_data:
                page_info = {
                    "page_number": page_data["page_number"],
                    "text": page_data["extracted_text"],
                    "ocr_details": page_data["ocr_result"]
                }
                json_data["pages"].append(page_info)
                full_text_parts.append(f"--- Page {page_data['page_number']} ---\n{page_data['extracted_text']}")
            
            json_data["full_text"] = "\n\n".join(full_text_parts)
            
            # Save JSON
            os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            # Create searchable PDF
            self._create_searchable_pdf(pages_data, pdf_output_path)
            
            return json_data
            
        except Exception as e:
            raise Exception(f"Error saving OCR results: {str(e)}")

    def _create_searchable_pdf(self, pages_data, output_path):
        """Create a searchable PDF from OCR results"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            doc = fitz.open()
            
            for page_data in pages_data:
                page = doc.new_page(width=595, height=842)  # A4 size
                
                # Insert text
                text = page_data["extracted_text"]
                if text.strip():
                    rect = fitz.Rect(50, 50, 545, 792)  # Margins
                    page.insert_textbox(rect, text, fontsize=11, 
                                      fontname="helv", color=(0, 0, 0))
            
            doc.save(output_path)
            doc.close()
            
        except Exception as e:
            print(f"Error creating searchable PDF: {e}")
            # Create a simple text-based PDF as fallback
            self._create_simple_pdf(pages_data, output_path)

    def _create_simple_pdf(self, pages_data, output_path):
        """Create a simple PDF with just text"""
        try:
            doc = fitz.open()
            page = doc.new_page()
            
            all_text = "\n\n".join([page_data["extracted_text"] for page_data in pages_data])
            rect = fitz.Rect(50, 50, 545, 792)
            page.insert_textbox(rect, all_text, fontsize=10)
            
            doc.save(output_path)
            doc.close()
            
        except Exception as e:
            print(f"Error creating simple PDF: {e}")


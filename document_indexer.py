import json
import os
import glob
from pdf_processor import PDFProcessor
from semantic_searcher import SemanticSearch
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from functools import partial

class DocumentIndexer:
    def __init__(self, api_key=None):
        self.pdf_processor = PDFProcessor()
        self.index_file = "document_index.json"
        self.documents_dir = os.path.join(os.path.dirname(__file__), "documents")
        # Removed downloads directory - only using documents folder
        self.content_cache_dir = os.path.join(os.path.dirname(__file__), "content_cache")
        
        # Initialize Gemini for AI-powered summaries
        if api_key:
            self.semantic_searcher = SemanticSearch(api_key)
            self.use_ai_summaries = True
        else:
            # Try to get API key from environment
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("API_KEY")
            if api_key:
                self.semantic_searcher = SemanticSearch(api_key)
                self.use_ai_summaries = True
            else:
                self.use_ai_summaries = False
                print("⚠️ No API key found - using rule-based summaries instead of AI")
        
        # Create content cache directory if it doesn't exist
        if not os.path.exists(self.content_cache_dir):
            os.makedirs(self.content_cache_dir)
    
    def get_file_hash(self, file_path):
        """Get MD5 hash of file to detect changes"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def create_intelligent_summary(self, text, max_length=300):
        """Create an intelligent summary focusing on important sentences and keywords"""
        if len(text) <= max_length:
            return text
        
        # Split into sentences
        sentences = [s.strip() for s in text.replace('.', '.\n').replace('!', '!\n').replace('?', '?\n').split('\n') if s.strip()]
        
        # If no proper sentences, use the beginning
        if not sentences or len(sentences) == 1:
            return text[:max_length] + "..."
        
        # Score sentences based on important indicators
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            
            # Higher score for sentences with:
            # - Numbers, dates, percentages
            if any(char.isdigit() for char in sentence):
                score += 2
            
            # - Important keywords
            important_words = ['important', 'key', 'main', 'primary', 'significant', 'conclusion', 'result', 'summary', 'objective', 'goal', 'purpose', 'method', 'approach', 'finding', 'recommendation']
            for word in important_words:
                if word in sentence_lower:
                    score += 3
            
            # - First and last sentences (often important)
            if i == 0 or i == len(sentences) - 1:
                score += 1
            
            # - Sentences with proper nouns (capitalized words)
            words = sentence.split()
            capitalized_count = sum(1 for word in words if word and word[0].isupper() and len(word) > 1)
            score += capitalized_count * 0.5
            
            # - Longer sentences (often more informative)
            if len(sentence.split()) > 10:
                score += 1
            
            scored_sentences.append((sentence, score, len(sentence)))
        
        # Sort by score (descending) and then by position (ascending)
        scored_sentences.sort(key=lambda x: (-x[1], sentences.index(x[0])))
        
        # Build summary by selecting highest-scoring sentences
        summary_parts = []
        current_length = 0
        
        for sentence, score, length in scored_sentences:
            if current_length + length + 3 <= max_length:  # +3 for space and potential punctuation
                summary_parts.append(sentence)
                current_length += length + 1
            elif current_length == 0:  # If first sentence is too long, truncate it
                summary_parts.append(sentence[:max_length-3] + "...")
                break
        
        if not summary_parts:
            return text[:max_length] + "..."
        
        summary = " ".join(summary_parts)
        if current_length < max_length and len(text) > current_length:
            summary += "..."
        
        return summary
    
    def create_ai_summary(self, text, max_length=300):
        """Create an AI-powered summary using Gemini"""
        if not self.use_ai_summaries:
            return self.create_intelligent_summary(text, max_length)
        
        # If text is already short enough, return as is
        if len(text) <= max_length:
            return text
        
        try:
            # Create a prompt for Gemini to generate a concise summary
            prompt = f"""Please create a concise summary of the following text in approximately {max_length} characters or less. Focus on the most important information, key findings, main points, and essential details. Preserve important numbers, dates, names, and technical terms.

Text to summarize:
{text[:2000]}"""  # Limit input text to avoid token limits
            
            # Use the semantic searcher's client to generate summary
            response = self.semantic_searcher.client.generate_content(prompt)
            
            if response and response.text:
                summary = response.text.strip()
                # Ensure the summary doesn't exceed the max length
                if len(summary) > max_length + 50:  # Allow some flexibility
                    summary = summary[:max_length] + "..."
                return summary
            else:
                # Fallback to rule-based summary if AI fails
                return self.create_intelligent_summary(text, max_length)
                
        except Exception as e:
            print(f"AI summary failed, using rule-based summary: {str(e)}")
            return self.create_intelligent_summary(text, max_length)
    
    def process_single_page(self, page_data):
        """Process a single page (for parallel processing)"""
        page_num, page_text, filename = page_data
        
        if not page_text.strip():
            return None
            
        try:
            # Create AI-powered summary if available, otherwise use rule-based
            if self.use_ai_summaries:
                summary = self.create_ai_summary(page_text.strip(), max_length=400)
            else:
                summary = self.create_intelligent_summary(page_text.strip(), max_length=400)
            
            return {
                "page_number": page_num + 1,
                "content": page_text.strip(),
                "summary": summary,
                "word_count": len(page_text.split())
            }
        except Exception as e:
            print(f"Error processing page {page_num + 1} of {filename}: {str(e)}")
            return None
    
    def extract_keywords(self, text):
        """Extract important keywords from text using improved algorithm"""
        import re
        from collections import Counter
        
        # Clean text and remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'a', 'an', 'as', 'if', 'then', 'than', 'so', 'very', 'much', 'more', 'most', 'such', 'no', 'not', 'only', 'own', 'same', 'other', 'some', 'any', 'all', 'each', 'every', 'many', 'few', 'several', 'page'}
        
        # Extract words, numbers, and compound terms
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9]*\b|\b\d+(?:\.\d+)?%?\b', text.lower())
        
        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequency and get top keywords
        word_freq = Counter(filtered_words)
        
        # Extract compound terms (2-3 words)
        sentences = re.split(r'[.!?]+', text)
        compound_terms = []
        for sentence in sentences:
            sentence_words = re.findall(r'\b[A-Za-z][A-Za-z0-9]*\b', sentence.lower())
            for i in range(len(sentence_words) - 1):
                if sentence_words[i] not in stop_words and sentence_words[i + 1] not in stop_words:
                    compound = f"{sentence_words[i]} {sentence_words[i + 1]}"
                    if len(compound) > 8:
                        compound_terms.append(compound)
        
        # Combine single words and compound terms
        keywords = [word for word, count in word_freq.most_common(15)]  # Top 15 single words
        keywords.extend(list(set(compound_terms))[:10])  # Top 10 unique compound terms
        
        return keywords[:20]  # Return top 20 keywords

    def extract_relations(self, text):
        """Extract semantic relations and key phrases from the text"""
        import re
        
        relations = []
        
        # Extract numerical relationships
        numerical_patterns = [
            r'\b\d+(?:\.\d+)?\s*(?:percent|%|times|fold|increase|decrease|ratio|rate)\b',
            r'\b(?:increased|decreased|reduced|improved|enhanced)\s+by\s+\d+(?:\.\d+)?\s*(?:percent|%)?\b',
            r'\b(?:from|between)\s+\d+(?:\.\d+)?\s+(?:to|and)\s+\d+(?:\.\d+)?\b'
        ]
        
        for pattern in numerical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            relations.extend([match.strip() for match in matches])
        
        # Extract causal relationships
        causal_patterns = [
            r'\b\w+\s+(?:causes?|leads?\s+to|results?\s+in|due\s+to|because\s+of)\s+\w+\b',
            r'\b(?:if|when|while|since)\s+\w+.*?\s+then\s+\w+\b',
            r'\b\w+\s+(?:affects?|influences?|impacts?)\s+\w+\b'
        ]
        
        for pattern in causal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            relations.extend([match.strip() for match in matches])
        
        # Extract comparative relationships
        comparative_patterns = [
            r'\b\w+\s+(?:is|are|was|were)\s+(?:higher|lower|greater|less|better|worse)\s+than\s+\w+\b',
            r'\b(?:compared\s+to|versus|vs\.?)\s+\w+\b',
            r'\b(?:more|less)\s+\w+\s+than\s+\w+\b'
        ]
        
        for pattern in comparative_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            relations.extend([match.strip() for match in matches])
        
        # Extract temporal relationships
        temporal_patterns = [
            r'\b(?:before|after|during|while|when|since|until)\s+\w+.*?\w+\b',
            r'\b(?:in|at|on)\s+\d{4}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b'
        ]
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            relations.extend([match.strip() for match in matches])
        
        # Clean and deduplicate relations
        relations = list(set([rel for rel in relations if len(rel) > 10 and len(rel) < 150]))
        
        return relations[:15]  # Return top 15 relations

    def extract_page_content_parallel(self, pdf_path, max_workers=4):
        """Extract content from each page of PDF using parallel processing"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            filename = os.path.basename(pdf_path)
            
            # Prepare page data for parallel processing
            page_data_list = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                page_data_list.append((page_num, page_text, filename))
            
            doc.close()
            
            # Process pages in parallel using ThreadPoolExecutor (better for I/O bound tasks like AI API calls)
            pages = []
            print(f"  📄 Processing {len(page_data_list)} pages in parallel...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all page processing tasks
                future_to_page = {executor.submit(self.process_single_page, page_data): page_data[0] 
                                 for page_data in page_data_list}
                
                # Collect results as they complete
                for future in as_completed(future_to_page):
                    page_result = future.result()
                    if page_result:
                        pages.append(page_result)
                        print(f"    ✅ Completed page {page_result['page_number']}")
            
            # Sort pages by page number to maintain order
            pages.sort(key=lambda x: x['page_number'])
            print(f"  🎯 Completed processing all {len(pages)} pages for {filename}")
            
            return pages
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def extract_page_content(self, pdf_path):
        """Extract content from each page of PDF (original sequential method)"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text.strip():  # Only add pages with content
                    print(f"  📄 Processing page {page_num + 1}...")
                    
                    # Create AI-powered summary if available, otherwise use rule-based
                    if self.use_ai_summaries:
                        summary = self.create_ai_summary(page_text.strip(), max_length=400)
                        print(f"    🤖 AI summary generated for page {page_num + 1}")
                    else:
                        summary = self.create_intelligent_summary(page_text.strip(), max_length=400)
                        print(f"    📝 Rule-based summary generated for page {page_num + 1}")
                    
                    pages.append({
                        "page_number": page_num + 1,
                        "content": page_text.strip(),
                        "summary": summary,
                        "word_count": len(page_text.split())
                    })
            
            doc.close()
            return pages
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def get_content_cache_path(self, filename):
        """Get the path for the content cache file"""
        base_name = os.path.splitext(filename)[0]
        return os.path.join(self.content_cache_dir, f"{base_name}_content.json")
    
    def save_content_to_cache(self, filename, pages, full_content):
        """Save extracted content to a separate cache file"""
        cache_path = self.get_content_cache_path(filename)
        content_data = {
            "filename": filename,
            "pages": pages,
            "full_content": full_content,
            "cached_at": datetime.now().isoformat()
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(content_data, f, indent=2, ensure_ascii=False)
    
    def load_content_from_cache(self, filename):
        """Load content from cache file"""
        cache_path = self.get_content_cache_path(filename)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def create_chatbot_summary_json(self, filename, pages):
        """Create a relational JSON structure for the chatbot"""
        relational_data = {
            "filename": filename,
            "summaries": []
        }

        for page in pages:
            page_keywords = self.extract_keywords(page["content"])
            page_relations = self.extract_relations(page["content"])
            relational_data["summaries"].append({
                "page_number": page["page_number"],
                "summary": page["summary"],
                "keywords": page_keywords,
                "relations": page_relations
            })

        hash_object = hashlib.sha256(filename.encode())
        json_filename = f"{hash_object.hexdigest()}_chatbot_summary.json"
        save_path = os.path.join(self.content_cache_dir, json_filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(relational_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Chatbot summary JSON saved as {json_filename}")
        return save_path

    def create_document_index(self):
        """Create or update the document index"""
        # Load existing index if it exists
        existing_index = {}
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    existing_index = json.load(f)
            except:
                existing_index = {}
        
        # Get all PDF files from documents folder
        pdf_files = []
        pdf_files.extend(glob.glob(os.path.join(self.documents_dir, "*.pdf")))
        # Also include other document types
        pdf_files.extend(glob.glob(os.path.join(self.documents_dir, "*.txt")))
        pdf_files.extend(glob.glob(os.path.join(self.documents_dir, "*.docx")))
        
        updated_index = {}
        print(f"🔄 Processing {len(pdf_files)} PDF files...")
        
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            file_hash = self.get_file_hash(pdf_path)
            
            # Check if file has changed
            if filename in existing_index and existing_index[filename].get("file_hash") == file_hash:
                print(f"✅ {filename} - No changes, using cached data")
                updated_index[filename] = existing_index[filename]
                continue
            
            print(f"🔄 Processing {filename}...")
            
            # Extract page content
            pages = self.extract_page_content_parallel(pdf_path)
            
            if pages:
                # Create document summary from page summaries
                all_page_summaries = " ".join([page["summary"] for page in pages])
                doc_summary = all_page_summaries[:1000] + "..." if len(all_page_summaries) > 1000 else all_page_summaries
                
                # Save content to separate cache file
                all_content = " ".join([page["content"] for page in pages])
                self.save_content_to_cache(filename, pages, all_content)
                
                # Create chatbot summary JSON for relational data
                self.create_chatbot_summary_json(filename, pages)
                
                # Store only metadata in the main index
                updated_index[filename] = {
                    "filename": filename,
                    "file_path": pdf_path,
                    "file_hash": file_hash,
                    "total_pages": len(pages),
                    "total_words": sum(page["word_count"] for page in pages),
                    "document_summary": doc_summary,
                    "last_updated": datetime.now().isoformat(),
                    "content_cache_path": self.get_content_cache_path(filename)
                }
                print(f"✅ {filename} - Processed {len(pages)} pages")
            else:
                print(f"⚠️ {filename} - Could not extract content")
        
        # Save updated index and chatbot summary JSONs
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(updated_index, f, indent=2, ensure_ascii=False)
        
        # Create or update chatbot summary JSONs
        for filename, doc_data in updated_index.items():
            cache_data = self.load_content_from_cache(filename)
            if cache_data:
                self.create_chatbot_summary_json(filename, cache_data['pages'])

        print(f"💾 Document index saved with {len(updated_index)} documents")
        return updated_index
    
    def load_index(self):
        """Load the document index"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def search_in_index(self, query, index_data):
        """Search for relevant documents in the index using simple text matching"""
        query_lower = query.lower()
        relevant_docs = []
        
        for filename, doc_data in index_data.items():
            # Load content from cache to search
            content_cache = self.load_content_from_cache(filename)
            if not content_cache:
                continue
                
            # Search in document summary and full content
            full_content = content_cache.get("full_content", "").lower()
            doc_summary = doc_data.get("document_summary", "").lower()
            
            # Simple relevance scoring
            relevance_score = 0
            
            # Check for query words in content
            query_words = query_lower.split()
            for word in query_words:
                if word in full_content:
                    relevance_score += full_content.count(word)
                if word in doc_summary:
                    relevance_score += doc_summary.count(word) * 2  # Summary matches are more important
            
            if relevance_score > 0:
                relevant_docs.append({
                    "filename": filename,
                    "relevance_score": relevance_score,
                    "doc_data": doc_data
                })
        
        # Sort by relevance score
        relevant_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_docs
    
    def get_relevant_content(self, query, max_docs=3):
        """Get relevant content from top matching documents"""
        index_data = self.load_index()
        
        if not index_data:
            return []
        
        relevant_docs = self.search_in_index(query, index_data)
        
        # Return top documents with their content
        result = []
        for doc in relevant_docs[:max_docs]:
            doc_info = doc["doc_data"]
            
            # Load full content from cache
            content_cache = self.load_content_from_cache(doc_info["filename"])
            if not content_cache:
                continue
                
            result.append({
                "filename": doc_info["filename"],
                "file_path": doc_info["file_path"],
                "relevance_score": doc["relevance_score"],
                "pages": content_cache["pages"],
                "full_content": content_cache["full_content"],
                "document_summary": doc_info["document_summary"]
            })
        
        return result

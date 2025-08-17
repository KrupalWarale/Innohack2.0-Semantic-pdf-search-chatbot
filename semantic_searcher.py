import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict
import time

# Load environment variables
load_dotenv()

class SemanticSearch:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.client = self.model  # Add client attribute for compatibility
        
    def get_relevant_sentences(self, query: str, text_chunks: List[str], top_k: int = 5) -> Dict:
        """Get relevant sentences from text chunks based on a query"""
        if not text_chunks:
            raise ValueError("Text chunks cannot be empty.")

        prompt = (
            f"You are an expert at finding relevant text in documents. "
            f"The user is searching for: '{query}'. "
            f"Find and extract up to {top_k} of the most relevant sentences from the text below. "
            f"CRITICAL: You must copy the sentences EXACTLY as they appear in the text - do not paraphrase, summarize, or modify them in any way. "
            f"Return only the exact sentences as they are written, numbered 1., 2., etc.\n\n"
            f"Text:\n---\n" + "\n\n".join(text_chunks) + "\n---"
        )

        try:
            response = self.model.generate_content(prompt)
            relevant_sentences = self.parse_response(response.text)
        except Exception as e:
            raise Exception(f"Error processing query with generative AI: {e}")

        return {
            'query': query,
            'relevant_sentences': relevant_sentences
        }
    
    def parse_response(self, response_text: str) -> List[str]:
        """Parse the numbered list response from the AI model"""
        sentences = []
        for line in response_text.split('\n'):
            # Check if the line starts with a number and a period (e.g., '1.')
            if line.strip().startswith(tuple(f'{i}.' for i in range(1, 11))):
                # Remove the number and period, and strip whitespace
                sentence = line.split('.', 1)[1].strip()
                sentences.append(sentence)
        return sentences


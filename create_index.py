from document_indexer import DocumentIndexer
import time
from dotenv import load_dotenv
import os

def main():
    print("🚀 Starting document indexing process...")
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("API_KEY")
    
    if not api_key:
        print("⚠️ API key not found - AI-powered summaries will be disabled")
    
    start_time = time.time()
    
    indexer = DocumentIndexer(api_key=api_key)
    index_data = indexer.create_document_index()
    
    end_time = time.time()
    
    print(f"\n✨ Indexing complete in {end_time - start_time:.2f} seconds")
    print(f"✅ {len(index_data)} documents indexed successfully")

if __name__ == "__main__":
    main()

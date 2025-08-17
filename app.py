import streamlit as st
import os
import base64
import glob
import json
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from semantic_searcher import SemanticSearch
from highlighter import PDFHighlighter
from document_indexer import DocumentIndexer
# Removed online PDF downloader - not needed
from ocr_processor import OCRProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🚀 Semantic Search Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to minimize spacing and fit viewport
st.markdown("""
<style>
    /* Aggressive removal of ALL top spacing - based on Streamlit community solutions */
    .stApp {
        margin-top: -80px !important;
        padding-top: 0rem !important;
    }
    
    /* Hide Streamlit header completely */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0rem !important;
    }
    
    /* Remove top padding from main container */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        margin-top: 0rem !important;
        max-width: 100%;
        width: 100%;
    }
    
    /* Target the main app view container */
    div[data-testid="stAppViewContainer"] {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    
    /* Target main content section */
    section[data-testid="stMain"] {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    
    /* Remove toolbar spacing */
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Remove decoration (three dots menu) */
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    
    /* Additional targeting for stubborn elements */
    .main {
        padding-left: 0rem;
        margin-left: 0rem;
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    
    /* Force remove any remaining top margins */
    .stApp > div:first-child {
        margin-left: 0;
        padding-left: 0;
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
    .full-height-container {
        height: calc(100vh - 8rem);
        overflow-y: auto;
        margin-left: 0;
        padding-left: 0;
    }
    .no-scroll-block {
        overflow: hidden;
    }
    h1 {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
        margin-top: 0;
        margin-left: 0;
        padding-left: 0;
    }
    h3 {
        font-size: 1.1rem;
        margin-bottom: 0.2rem;
        margin-top: 0.3rem;
        margin-left: 0;
        padding-left: 0;
    }
    .stButton > button {
        width: 100%;
        margin-top: 0rem;
        padding: 0.35rem;
        border: 1px solid #ccc;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .element-container {
        margin-bottom: 0.2rem;
        margin-left: 0;
        padding-left: 0;
    }
    .stTextInput > div > div > input {
        padding: 0.25rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background-color: #f3e5f5;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 10px 0;
        border-left: 4px solid #9c27b0;
    }
    .document-reference {
        background-color: #fff3e0;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 5px 0;
        border-left: 3px solid #ff9800;
        font-size: 0.9em;
    }
    /* Remove left gaps from columns */
    .stColumn {
        padding-left: 0rem !important;
        margin-left: 0rem !important;
    }
    .stColumn > div {
        padding-left: 0rem !important;
        margin-left: 0rem !important;
    }
    /* Adjustments for sidebar */
    section[data-testid="stSidebar"] {
        padding-top: 0.5rem;
    }
    .stSidebar > div:first-child {
        padding-top: 0;
    }
    .stSidebar [data-testid="stVerticalBlock"] > div {
        gap: 0.3rem;
    }
    .stSidebar .stButton {
        margin-bottom: 0.3rem;
    }
    .stSidebar .stMarkdown {
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }
    /* Hide sidebar when not visible */
    .sidebar-hidden section[data-testid="stSidebar"] {
        display: none !important;
    }
    /* Toggle button styling */
    .sidebar-toggle {
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 999;
        background: #ff4b4b;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 12px;
        cursor: pointer;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .sidebar-toggle:hover {
        background: #ff3333;
    }
    /* Remove gaps from containers */
    .stContainer {
        padding-left: 0rem !important;
        margin-left: 0rem !important;
    }
    /* Remove gaps from tabs */
    .stTabs {
        margin-left: 0rem !important;
        padding-left: 0rem !important;
    }
    .stTabs > div {
        margin-left: 0rem !important;
        padding-left: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
load_dotenv()

@st.cache_resource
def init_components():
    api_key = os.getenv("API_KEY")
    if not api_key:
        st.error("API key not found in .env file.")
        st.stop()
    return (
        PDFProcessor(),
        SemanticSearch(api_key),
        PDFHighlighter(),
        DocumentIndexer(),
        OCRProcessor(api_key=api_key)
    )

pdf_processor, semantic_searcher, highlighter, indexer, ocr_processor = init_components()

# --- UTILITY FUNCTIONS ---
def load_pdf_file(filename):
    """Load PDF file from documents or downloads folder"""
    for folder in ["documents", "downloads"]:
        file_path = os.path.join(os.path.dirname(__file__), folder, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return f.read()
    return None

@st.cache_data
def get_document_index():
    """Get the document index, create if doesn't exist"""
    index_data = indexer.load_index()
    if not index_data:
        st.info("🔄 Creating document index for the first time...")
        index_data = indexer.create_document_index()
    return index_data

def load_chatbot_summaries():
    """Load all chatbot summaries"""
    content_cache_dir = os.path.join(os.path.dirname(__file__), "content_cache")
    if not os.path.exists(content_cache_dir):
        return {}
    
    summary_files = [f for f in os.listdir(content_cache_dir) if f.endswith("_chatbot_summary.json")]
    summaries = {}
    for summary_file in summary_files:
        try:
            with open(os.path.join(content_cache_dir, summary_file), 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
                summaries[summary_data['filename']] = summary_data
        except:
            continue
    return summaries

def search_summaries(query, summaries):
    """Search through chatbot summaries to find relevant content"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    relevant_sections = []
    
    for filename, doc_data in summaries.items():
        for page_summary in doc_data.get('summaries', []):
            summary_text = page_summary['summary'].lower()
            score = 0
            
            # Score based on keyword matches
            for word in query_words:
                if word in summary_text:
                    score += summary_text.count(word) * 3
            
            if score > 0:
                relevant_sections.append({
                    'filename': filename,
                    'page_number': page_summary['page_number'],
                    'summary': page_summary['summary'],
                    'keywords': page_summary.get('keywords', []),
                    'relations': page_summary.get('relations', []),
                    'relevance_score': score
                })
    
    relevant_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
    return relevant_sections[:5]

def generate_ai_response(query, relevant_sections, semantic_searcher):
    """Generate AI response using found relevant sections"""
    if not relevant_sections:
        return "I couldn't find any relevant information in your documents for that query."
    
    context = "Based on the following information from your documents:\n\n"
    for section in relevant_sections:
        context += f"From {section['filename']} (Page {section['page_number']}): {section['summary']}\n\n"
    
    prompt = f"""You are an AI assistant helping a user understand their documents. Based on the context provided below, please answer the user's question in a helpful and informative way.\n\nUser Question: {query}\n\nContext from documents:\n{context}\n\nPlease provide a comprehensive answer based on the information available. If the information is incomplete, mention what additional details might be helpful. Be conversational and helpful."""
    
    try:
        response = semantic_searcher.client.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            return "I'm having trouble generating a response right now. Please try again."
    except Exception as e:
        return f"Error generating response: {str(e)}"

# --- SESSION STATE ---
if "gemini_results" not in st.session_state:
    st.session_state.gemini_results = []
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "📊 Dashboard" # Default page
if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = {}
# Sidebar is always visible - no toggle needed


# --- NO TOGGLE FUNCTIONALITY ---
# Sidebar is always visible

# --- DOCUMENT STATS CALCULATION ---
documents_dir = os.path.join(os.path.dirname(__file__), "documents")
content_cache_dir = os.path.join(os.path.dirname(__file__), "content_cache")

docs_count = len([f for f in os.listdir(documents_dir) if f.lower().endswith(('.pdf', '.txt', '.docx'))]) if os.path.exists(documents_dir) else 0
json_count = len([f for f in os.listdir(content_cache_dir) if f.endswith('.json')]) if os.path.exists(content_cache_dir) else 0

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 Semantic Search")
st.sidebar.markdown("---")

# Navigation buttons for 4 pages
if st.sidebar.button("📊 Dashboard", use_container_width=True):
    st.session_state.current_page = "📊 Dashboard"
    st.rerun()

if st.sidebar.button("🔍 Semantic Search with Filter", use_container_width=True):
    st.session_state.current_page = "🔍 Semantic Search with Filter"
    st.rerun()

if st.sidebar.button("📊 JSON Viewer", use_container_width=True):
    st.session_state.current_page = "📊 JSON Viewer"
    st.rerun()

if st.sidebar.button("🤖 Advanced Chatbot", use_container_width=True):
    st.session_state.current_page = "🤖 Advanced Chatbot"
    st.rerun()

if st.sidebar.button("📁 Document Manager", use_container_width=True):
    st.session_state.current_page = "📁 Document Manager"
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.title("🛠️ Quick Actions")

if st.sidebar.button("🔄 Preprocess All Documents", use_container_width=True):
    with st.spinner("Preprocessing all documents..."):
        indexer.create_document_index()
        st.success("All documents preprocessed successfully!")
        st.rerun()

# --- MAIN CONTENT BASED ON SELECTION ---

if st.session_state.current_page == "📊 Dashboard":
    st.title("📊 Dashboard")
    
    # Welcome message
    st.markdown("Welcome to your **Semantic Search Hub**! Manage and search your documents with AI-powered tools.")
    
    # Stats Cards Row
    st.markdown("### 📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📄 Documents",
            value=docs_count,
            help="Total documents in your library"
        )
    
    with col2:
        st.metric(
            label="📊 Processed", 
            value=json_count,
            help="Processed document files"
        )
    
    with col3:
        st.metric(
            label="🔧 Status",
            value="Ready",
            help="System status"
        )
    
    with col4:
        st.metric(
            label="💾 Storage",
            value=f"{docs_count} Files",
            help="Total files in storage"
        )
    
    # Action Buttons Row
    st.markdown("### 🚀 Quick Actions")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔄 Preprocess Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing documents..."):
                indexer.create_document_index()
                st.success("Documents processed successfully!")
                st.rerun()
    
    with action_col2:
        if st.button("🔍 Start Searching", use_container_width=True):
            st.session_state.current_page = "🔍 Semantic Search with Filter"
            st.rerun()
    
    with action_col3:
        if st.button("🤖 Chat with AI", use_container_width=True):
            st.session_state.current_page = "🤖 Advanced Chatbot"
            st.rerun()
    
    # Recent Activity
    st.markdown("### 📋 Recent Activity")
    
    activity_col1, activity_col2 = st.columns(2)
    
    with activity_col1:
        st.info("🔍 **Search Activity**\n\nNo recent searches. Start exploring your documents!")
    
    with activity_col2:
        st.info("🤖 **AI Conversations**\n\nNo recent chats. Ask AI about your documents!")

elif st.session_state.current_page == "🔍 Semantic Search with Filter":
    st.title("🔍 Semantic Search with Filter")
    with st.container():
        st.markdown("Search and analyze your documents with AI-powered semantic search")
        
        # Two-column layout
        col1, col2 = st.columns([0.7, 0.3])
        
        with col2:
            st.markdown("### 🛠️ Search Controls")
            
            # Document index status
            with st.spinner("Loading document index..."):
                document_index = get_document_index()
            
            if document_index:
                st.success(f"📚 Indexed {len(document_index)} documents")
                
                with st.expander("📋 Available Documents", expanded=False):
                    for filename, doc_data in document_index.items():
                        st.write(f"• {filename} ({doc_data['total_pages']} pages)")
            
            # Search input
            st.markdown("### 🔍 Search")
            query = st.text_input("Enter your search query:", key="main_search")
            
            if st.button("🚀 Search Documents") and query:
                with st.spinner("Searching documents..."):
                    try:
                        relevant_docs = indexer.get_relevant_content(query, max_docs=3)
                        
                        if relevant_docs:
                            # Process documents for search results
                            def process_single_document(doc, query):
                                try:
                                    text_chunks = pdf_processor.split_into_chunks(doc['full_content'])
                                    results = semantic_searcher.get_relevant_sentences(query, text_chunks)
                                    relevant_sentences = results.get('relevant_sentences', [])
                                    
                                    if relevant_sentences:
                                        result_data = {
                                            "filename": doc["filename"],
                                            "relevance_score": doc.get("relevance_score", 0),
                                            "search_results": relevant_sentences,
                                            "page_summaries": [p['summary'] for p in doc.get('pages', [])]
                                        }
                                        
                                        if doc['filename'].lower().endswith('.pdf'):
                                            pdf_bytes = load_pdf_file(doc['filename'])
                                            if pdf_bytes:
                                                highlighted_pdf = highlighter.highlight_text_in_pdf(pdf_bytes, relevant_sentences)
                                                result_data["pdf_bytes"] = pdf_bytes
                                                result_data["highlighted_pdf"] = highlighted_pdf
                                        
                                        return result_data
                                except Exception as e:
                                    st.error(f"Error processing {doc['filename']}: {str(e)}")
                                return None
                            
                            # Process all documents
                            results = []
                            # Using ThreadPoolExecutor for concurrent processing
                            with ThreadPoolExecutor(max_workers=3) as executor:
                                future_to_doc = {executor.submit(process_single_document, doc, query): doc for doc in relevant_docs}
                                for future in as_completed(future_to_doc):
                                    result = future.result()
                                    if result:
                                        results.append(result)
                            
                            st.session_state.gemini_results = results
                            st.session_state.search_query = query
                            
                            if results:
                                total_results = sum(len(result['search_results']) for result in results)
                                st.success(f"🎯 Found {total_results} relevant results across {len(results)} documents")
                            else:
                                st.warning("❌ No specific matches found.")
                        else:
                            st.warning("❌ No relevant documents found.")
                            
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
            
            # Clear results
            if st.session_state.gemini_results and st.button("🗑️ Clear Results"):
                st.session_state.gemini_results = []
                st.session_state.search_query = ""
                st.rerun()
        
        with col1:
            st.markdown("### 📄 Search Results")
            
            if st.session_state.gemini_results:
                # Create tabs for each document
                tab_titles = [f"📄 {result['filename']} ({len(result['search_results'])} matches)" for result in st.session_state.gemini_results]
                tabs = st.tabs(tab_titles)
                
                for i, result in enumerate(st.session_state.gemini_results):
                    with tabs[i]:
                        st.subheader(f"{result['filename']}")
                        
                        if 'highlighted_pdf' in result:
                            # Show highlighted PDF
                            base64_pdf = base64.b64encode(result['highlighted_pdf']).decode('utf-8')
                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="calc(100vh - 250px)" type="application/pdf" style="border: none; border-radius: 5px;"></iframe>'
                            st.markdown(pdf_display, unsafe_allow_html=True)
                        else:
                            # Show highlighted text
                            st.markdown("**📝 Highlighted Text**")
                            highlighted_text = "<div style='height: calc(100vh - 250px); overflow-y: auto; padding: 16px; background-color: #f9f9f9; border-radius: 8px; border: 1px solid #e0e0e0; font-size: 14px; line-height: 1.6;'>"
                            
                            for idx, sentence in enumerate(result['search_results'], 1):
                                highlighted_text += f"<div style='margin-bottom: 12px; padding: 8px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>"
                                highlighted_text += f"<strong style='color: #856404; font-size: 12px;'>Match {idx}:</strong><br>"
                                highlighted_text += f"<span style='color: #333;'>{sentence}</span>"
                                highlighted_text += "</div>"
                            
                            highlighted_text += "</div>"
                            st.markdown(highlighted_text, unsafe_allow_html=True)
            else:
                st.markdown('<div style="height: calc(100vh - 200px); display: flex; align-items: center; justify-content: center; background-color: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 5px;"><p style="color: #6c757d; font-size: 18px; text-align: center;">🚀 Enter a search query to find and view relevant content with highlighted results</p></div>', unsafe_allow_html=True)

elif st.session_state.current_page == "📊 JSON Viewer":
    st.title("📊 JSON Viewer")
    
    # Initialize with all JSON data by default
    if "json_view_mode" not in st.session_state:
        st.session_state.json_view_mode = "all"
    
    # Get all JSON files and prepare combined data
    all_json_data = {}
    if os.path.exists(content_cache_dir):
        files = os.listdir(content_cache_dir)
        json_files = [f for f in files if f.endswith('.json')]
        
        # Load document index
        try:
            index_data = indexer.load_index()
            all_json_data["Document Index"] = index_data
        except:
            pass
        
        # Load all JSON files
        for file in json_files:
            try:
                with open(os.path.join(content_cache_dir, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_json_data[file] = data
            except:
                continue
    
    # Two column layout - JSON on left, document selection on right
    col_json, col_select = st.columns([2, 1])
    
    with col_json:
        # Show JSON content
        if st.session_state.json_view_mode == "all":
            # Show all JSON files combined
            json_str = json.dumps(all_json_data, indent=2, ensure_ascii=False)
        else:
            # Show specific document
            if st.session_state.json_view_mode in all_json_data:
                json_str = json.dumps(all_json_data[st.session_state.json_view_mode], indent=2, ensure_ascii=False)
            else:
                json_str = "{}"
        
        # Display JSON directly
        st.code(json_str, language='json')
    
    with col_select:
        st.markdown("### 📁 Select Document")
        
        # Show All button
        if st.button("📊 Show All JSON", use_container_width=True, 
                    type="primary" if st.session_state.json_view_mode == "all" else "secondary"):
            st.session_state.json_view_mode = "all"
            st.rerun()
        
        st.markdown("---")
        
        # Individual document buttons
        for doc_name in all_json_data.keys():
            button_type = "primary" if st.session_state.json_view_mode == doc_name else "secondary"
            
            # Clean up display name
            display_name = doc_name
            if doc_name.endswith('.json'):
                display_name = doc_name[:-5]  # Remove .json extension
            
            if st.button(f"📄 {display_name}", use_container_width=True, type=button_type, key=f"select_{doc_name}"):
                st.session_state.json_view_mode = doc_name
                st.rerun()
        
        # Show current selection info
        if all_json_data:
            st.markdown("---")
            st.markdown("### ℹ️ Current View")
            if st.session_state.json_view_mode == "all":
                st.info(f"📊 Showing all {len(all_json_data)} JSON files combined")
            else:
                st.info(f"📄 Showing: {st.session_state.json_view_mode}")
        else:
            st.warning("📂 No JSON files found")

elif st.session_state.current_page == "🤖 Advanced Chatbot":
    # Remove Streamlit page title and intro text from outside columns
    with st.container():
        # Load chatbot summaries
        chatbot_summaries = load_chatbot_summaries()

        # Always initialize selected_source before any use
        if "selected_source" not in st.session_state:
            st.session_state.selected_source = None
        
        if chatbot_summaries:
            # --- Two-column layout: Left = Chat, Right = Source Viewer ---
            chat_col, source_col = st.columns([0.6, 0.4], gap="medium")

            with chat_col:
                # Move title and subtitle into left container
                st.markdown("""
                <div style='font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2em;'>🤖 Advanced Chatbot</div>
                <div style='font-size: 1.1rem; color: #555; margin-bottom: 1.2em;'>Chat with your documents using advanced AI</div>
                """, unsafe_allow_html=True)
                st.markdown("### 💬 Conversation")
                # Chat input
                query = st.text_input(
                    "Ask me anything about your documents:",
                    key="chatbot_query",
                    placeholder="e.g., What are the main findings in my research papers?"
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🚀 Ask AI", type="primary"):
                        if query:
                            with st.spinner("🔍 Searching documents and generating response..."):
                                relevant_sections = search_summaries(query, chatbot_summaries)
                                ai_response = generate_ai_response(query, relevant_sections, semantic_searcher)
                                st.session_state.chat_history.append((query, ai_response, relevant_sections))
                                st.rerun()
                        else:
                            st.error("Please enter a question.")
                with col_b:
                    if st.button("🗑️ Clear Chat"):
                        st.session_state.chat_history = []
                        st.session_state.selected_source = None
                        st.rerun()

                # --- Chat history (no outer div container) ---
                if st.session_state.chat_history:
                    for i, (user_msg, ai_msg, references) in enumerate(st.session_state.chat_history):
                        st.markdown(f'<div class="user-message"><strong>You:</strong> {user_msg}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="ai-message"><strong>AI:</strong> {ai_msg}</div>', unsafe_allow_html=True)
                        # Show sources as clickable icons/buttons
                        if references:
                            st.markdown("**📖 Sources:**", unsafe_allow_html=True)
                            for j, ref in enumerate(references):
                                btn_label = f"{ref['filename']} (Page {ref['page_number']})"
                                if st.button(f"🔗 {btn_label}", key=f"srcbtn_{i}_{j}"):
                                    st.session_state.selected_source = ref
                        st.markdown("---")
                else:
                  
                    st.markdown("💡 **Tip:** The more specific your question, the better I can help you find relevant information in your documents.")

            with source_col:
                # Remove the 'Source Viewer' title and reduce top margin for better alignment
                st.markdown("""
                <div style='background-color: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; padding: 16px 24px 16px 24px; margin-top: 0; min-height: 600px;'>
                """, unsafe_allow_html=True)
                selected = st.session_state.selected_source
                if selected:
                    st.markdown(f"**{selected['filename']} - Page {selected['page_number']}**")
                    # Try to show highlighted PDF if available, else show highlighted text
                    pdf_bytes = load_pdf_file(selected['filename'])
                    if pdf_bytes and selected['filename'].lower().endswith('.pdf'):
                        # Highlight the summary text in the PDF
                        highlighted_pdf = highlighter.highlight_text_in_pdf(pdf_bytes, [selected['summary']])
                        base64_pdf = base64.b64encode(highlighted_pdf).decode('utf-8')
                        pdf_display = f'<iframe src=\"data:application/pdf;base64,{base64_pdf}\" width=\"100%\" height=\"600px\" type=\"application/pdf\"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    else:
                        # Show highlighted summary text
                        st.markdown("**📝 Highlighted Text**")
                        st.markdown(f"<div style='background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; padding: 12px; margin-bottom: 8px;'><span style='color: #333;'>{selected['summary']}</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown('<div style="height: 100%; display: flex; align-items: center; justify-content: center; color: #a0a0a0; font-size: 18px; text-align: center;">🔗 Click a source in the chat to view it here with highlights.</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("📁 No document summaries found. Please process documents first.")
            if st.button("🔄 Process Documents Now"):
                with st.spinner("Processing documents..."):
                    indexer.create_document_index()
                    st.success("Documents processed! Please refresh the page.")
                    st.rerun()

elif st.session_state.current_page == "📁 Document Manager":
    st.title("📁 Document Manager")
    
    # Top action bar - compact
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        # File upload
        uploaded_files = st.file_uploader("📤 Upload Documents", 
                                        type=['pdf', 'txt', 'docx'], 
                                        accept_multiple_files=True,
                                        help="Upload documents to add to your collection")
        
        if uploaded_files:
            # Ensure documents directory exists
            os.makedirs(documents_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                # Save uploaded file to documents folder
                file_path = os.path.join(documents_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
            st.rerun()
    
    with col2:
        if st.button("✅ Select All", use_container_width=True):
            for key in st.session_state.selected_documents:
                st.session_state.selected_documents[key] = True
            st.rerun()
    
    with col3:
        if st.button("❌ Deselect All", use_container_width=True):
            for key in st.session_state.selected_documents:
                st.session_state.selected_documents[key] = False
            st.rerun()
    
    with col4:
        # Count selected documents
        total_docs = len(st.session_state.selected_documents)
        selected_count = sum(1 for selected in st.session_state.selected_documents.values() if selected)
        
        if selected_count > 0:
            if st.button(f"🔄 Process ({selected_count})", type="primary", use_container_width=True):
                with st.spinner(f"Processing {selected_count} documents..."):
                    indexer.create_document_index()
                    st.success(f"✅ Processed {selected_count} documents!")
                    st.rerun()
        else:
            st.button("⚠️ No Selection", disabled=True, use_container_width=True)
    
    st.markdown("---")
    
    # Get all documents from documents directory
    all_documents = {}
    
    # Documents folder
    if os.path.exists(documents_dir):
        docs_files = [f for f in os.listdir(documents_dir) if f.lower().endswith(('.pdf', '.txt', '.docx'))]
        if docs_files:
            all_documents["📁 Documents"] = docs_files
    
    if not all_documents:
        st.info("📂 No documents found. Upload files above to get started.")
    else:
        # Initialize selected documents if not exists
        for folder, files in all_documents.items():
            for file in files:
                file_key = f"{folder}/{file}"
                if file_key not in st.session_state.selected_documents:
                    st.session_state.selected_documents[file_key] = True
        
        # Compact document list in single scrollable container
        st.markdown("### 📋 Document Collection")
        
        # Create scrollable container
        with st.container():
            st.markdown("""
            <div style='height: 400px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; background-color: #fafafa;'>
            """, unsafe_allow_html=True)
            
            for folder, files in all_documents.items():
                # Compact folder header
                st.markdown(f"**{folder}** ({len(files)} files)")
                
                # Compact file list in 2 columns
                for i in range(0, len(files), 2):
                    col_left, col_right = st.columns(2)
                    
                    # Left file
                    if i < len(files):
                        file = files[i]
                        file_key = f"{folder}/{file}"
                        
                        with col_left:
                            # Get file size
                            file_path = os.path.join(documents_dir, file)
                            size_mb = 0
                            if os.path.exists(file_path):
                                size_mb = os.path.getsize(file_path) / (1024*1024)
                            
                            # Compact checkbox with file info
                            selected = st.checkbox(
                                f"📄 {file[:25]}{'...' if len(file) > 25 else ''} ({size_mb:.1f}MB)",
                                value=st.session_state.selected_documents.get(file_key, True),
                                key=f"doc_{file_key}"
                            )
                            st.session_state.selected_documents[file_key] = selected
                    
                    # Right file
                    if i + 1 < len(files):
                        file = files[i + 1]
                        file_key = f"{folder}/{file}"
                        
                        with col_right:
                            # Get file size
                            file_path = os.path.join(documents_dir, file)
                            size_mb = 0
                            if os.path.exists(file_path):
                                size_mb = os.path.getsize(file_path) / (1024*1024)
                            
                            # Compact checkbox with file info
                            selected = st.checkbox(
                                f"📄 {file[:25]}{'...' if len(file) > 25 else ''} ({size_mb:.1f}MB)",
                                value=st.session_state.selected_documents.get(file_key, True),
                                key=f"doc_{file_key}"
                            )
                            st.session_state.selected_documents[file_key] = selected
                
                st.markdown("---")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Bottom summary bar
        st.markdown("---")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("📄 Total", total_docs)
        with summary_col2:
            st.metric("✅ Selected", selected_count)
        with summary_col3:
            st.metric("❌ Excluded", total_docs - selected_count)

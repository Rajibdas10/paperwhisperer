import streamlit as st
# --- Available Summarization Models ---
AVAILABLE_SUMMARIZERS = ["minilm", "bart", "pegasus", "falcon"]

# --- Page settings ---
st.set_page_config(
    page_title="PaperWhisperer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import sys
import asyncio
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from llms.llm_factory import load_model
from summarizer.summarizer_factory import load_summarizer
from modules.chunking import split_info_semantic_chunks, clean_chunk
from modules.qa_engine import QAEngine
from modules.cheatsheet_engine import generate_cheat_sheet
from utils.difficulty_score import estimate_difficulty
from utils.pdf_reader import extract_text_from_pdf
from utils.reference_parser import parse_multiple_references_to_bibtex
from utils.bibtex_utils import generate_bibtex_from_arxiv
from utils.bibtex_generator import robust_bibtex_fetch
from io import BytesIO

# --- Load LLM for Smart Headline ---
llm = load_model("minilm")
generate_smart_headline = llm.generate_headline

qa_engine = QAEngine()

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .stApp {
            background-color: #0f1117;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }

        h1, h2, h3 {
            color: #00e0ff;
        }

        .main-title {
            font-size: 3em;
            font-weight: bold;
            background: -webkit-linear-gradient(45deg, #00e0ff, #a3ffe9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: -10px;
        }

        .tagline {
            font-size: 1.2em;
            color: #cccccc;
            margin-bottom: 25px;
        }

        .stFileUploader > div:first-child {
            background-color: #1e1e1e;
            border: 2px dashed #00e0ff;
            padding: 20px;
            border-radius: 10px;
            color: white;
        }

        #MainMenu, header, footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
selected_tab = st.sidebar.radio("Choose a Feature", ["Home", "Summarizer", "Cheat Sheet", "Q&A", "Smart Headline", "Paper Difficulty Score", "Citation & Reference Generator"])


# --- Main Area ---
if selected_tab == "Home":
    st.markdown("""
        <div class="main-title">
            🧠 <span style="background: -webkit-linear-gradient(45deg, #00e0ff, #a3ffe9);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;">
            PaperWhisperer</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="tagline">Your smart AI research paper companion ⚡</div>', unsafe_allow_html=True)
    st.markdown("---")

    # File Upload Section
    st.markdown("### 📤 Upload your research paper (PDF)")
    uploaded_file = st.file_uploader(" ", type=["pdf"])

    # Handle restoration after tab switch
    if uploaded_file is None and "uploaded_file_bytes" in st.session_state:
        uploaded_file = BytesIO(st.session_state.uploaded_file_bytes)
        uploaded_file.name = st.session_state.get("file_name", "Restored_File.pdf")

    # Upload and Extract
    if uploaded_file is not None:
        if "uploaded_file_bytes" not in st.session_state or uploaded_file.name != st.session_state.get("file_name"):
            with st.spinner("📚 Analyzing and decoding your research paper..."):
                file_bytes = uploaded_file.read()
                st.session_state.uploaded_file_bytes = file_bytes
                st.session_state.file_name = uploaded_file.name
                st.session_state.pdf_text = extract_text_from_pdf(BytesIO(file_bytes))
        st.success(f"✅ PDF Loaded: {uploaded_file.name}")

    elif "uploaded_file_bytes" in st.session_state:
        st.success(f"✅ Restored file: {st.session_state.get('file_name', 'Uploaded PDF')}")

    if "pdf_text" in st.session_state:
        st.subheader("📄 Extracted Text Preview:")
        st.write(st.session_state.pdf_text[:1000] + "...")

        if st.button("🗑️ Clear Extraction"):
            for key in ["uploaded_file_bytes", "pdf_text", "file_name"]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        st.warning("📎 Please upload a PDF file to continue.")

# --- Summarizer Page ---
if selected_tab == "Summarizer":
    summarizer_choice = st.sidebar.selectbox(
        "Select a Summarization Model",
        ["minilm", "bart", "pegasus", "falcon"],
        index=0
    )
    summarizer = load_summarizer(summarizer_choice)
    st.sidebar.caption(f"📌 Currently using: **{summarizer_choice.upper()}** model")

    st.title("📚 Summarizer")

    if "pdf_text" not in st.session_state or not st.session_state["pdf_text"].strip():
        st.warning("⚠️ No PDF text available. Please upload a valid PDF from the Home tab.")
    else:
        with st.spinner("Generating summary..."):
            # 1. Set a limit (like 1024 tokens roughly = ~3000-4000 characters)
            max_input_chars = 3000  # safer side

            # 2. Truncate if text is too long
            input_text = st.session_state["pdf_text"]
            if len(input_text) > max_input_chars:
                input_text = input_text[:max_input_chars]

            summary = summarizer(
                input_text,
                max_length=512,
                min_length=50,
                do_sample=False
            )[0]["summary_text"]


        st.subheader("Summary:")
        st.write(summary)

        st.download_button(
            label="💾 Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )

elif selected_tab == "Cheat Sheet":
    st.header("🧾 Cheat Sheet Generator")
    
    if "pdf_text" in st.session_state and st.session_state["pdf_text"].strip():
        pdf_text = st.session_state["pdf_text"]

        if st.button("Generate Cheat Sheet"):
            with st.spinner("Generating Cheat Sheet..."):
                cheat_sheet = generate_cheat_sheet(pdf_text)
                st.markdown("### 📋 Cheat Sheet")
                st.markdown(cheat_sheet)
    else:
        st.warning("Please upload pdf first")
    
    
elif selected_tab == "Q&A":
    st.header("❓ Ask Me Anything")
    st.write("Ask questions based on the content of your uploaded research paper.")
    
    if "pdf_text" not in st.session_state or not st.session_state["pdf_text"].strip():
        st.warning("PLease upload a PDF file in the Home tab first")
    else:
        if "qa_index_built" not in st.session_state:
            with st.spinner ("🔍 Indexing content for question-answering..."):
                qa_engine.prepare_context(st.session_state["pdf_text"])
                st.session_state.qa_engine = qa_engine
                st.session_state.qa_index_built = True
        else:
            qa_engine = st.session_state.qa_engine
            
        user_question = st.text_input("💬 Ask a question about the paper:")
        
        if st.button("Get Answer") and user_question.strip():
            with st.spinner("🤖 Thinking..."):
                answer = qa_engine.answer_question(user_question)
            st.success("Answer:")
            st.markdown(f"**{answer}**")       
        
elif selected_tab == "Smart Headline":
    st.header("Smart Headline Generator")
    st.write("Enter your text and let AI handle it to save your time.")

    user_text = st.text_area("Paste your paragraph here:")

    if st.button("🚀 Generate Headline"):
        if user_text.strip():
            with st.spinner("Thinking..."):
                headline = generate_smart_headline(user_text)
            st.success("✨ Here's your smart headline:")
            st.markdown(f"### 📰 **{headline}**")
        else:
            st.warning("⚠️ Please enter some text before generating.")

elif selected_tab == "Paper Difficulty Score":
    st.header("Paper Difficulty Analyzer")
    st.write("Paste a chunk of your paper to estimate the difficulty.")
    
    user_input = st.text_area("Enter paper content here")
    
    if st.button("Analyze Difficulty"):
        if user_input.strip():
            with st.spinner ("Analyzing..."):
                result = estimate_difficulty(user_input)
            st.success("Difficult Analysis Complete")
            st.write(f"**Difficulty Level:** {result['difficulty_level']}")
            st.write(f"**Flesch Score:** {result['flesch_score']}")
            st.write(f"**Average Sentence Length:** {result['avg_sentence_length']}")
            st.write(f"**Vocabulary Richness:** {result['vocab_richness']}")
            st.write(f"**Jargon Density:** {result['jargon_density']}")
            st.write(f"**Transformer-Based Semantic Score:** {result['semantic_score']}")
        else:
            st.warning("Please enter some text.")
            

elif selected_tab == "Citation & Reference Generator":
    st.header("📚 Citation & Reference Generator")
    st.write("Paste the reference section or a single reference below to generate BibTeX entries.")

    input_references = st.text_area("✏️ Paste the reference(s) here", height=200)

    if st.button("🧾 Generate BibTeX"):
        if input_references.strip():
            with st.spinner("📚 Parsing and generating BibTeX..."):
                try:
                    bibtex_entries = parse_multiple_references_to_bibtex(input_references)
                    formatted_bibtex = "\n\n".join(bibtex_entries) if isinstance(bibtex_entries, list) else bibtex_entries

                    st.markdown("### 📌 Generated BibTeX:")
                    st.code(formatted_bibtex, language="bibtex")

                    st.download_button(
                        label="💾 Download BibTeX File",
                        data=formatted_bibtex,
                        file_name="references.bib",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ Failed to parse references: {e}")
        else:
            st.warning("⚠️ Please enter reference(s) before generating.")

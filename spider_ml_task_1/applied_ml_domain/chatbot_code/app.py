import streamlit as st
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
st.set_page_config(page_title="SPML RAG System", layout="wide")
st.title("📚 SPML Task 1: Research Paper RAG")

# EXPLICIT ABSOLUTE PATH FOR THE DATABASE
CHROMA_PATH = "/content/chroma_db"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    openrouter_api_key = st.text_input("OpenRouter API Key", type="password")
    pdf_directory = st.text_input("PDF Directory path", value="/content/papers")
    
    if st.button("Initialize Database"):
        if not os.path.exists(pdf_directory) or not os.listdir(pdf_directory):
            st.error(f"No PDFs found in '{pdf_directory}'. Please upload them first.")
        else:
            with st.spinner("Processing PDFs and building vector database..."):
                try:
                    # 1. Load and Split
                    loader = PyPDFDirectoryLoader(pdf_directory)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    
                    # 2. Clear old database safely
                    if os.path.exists(CHROMA_PATH):
                        shutil.rmtree(CHROMA_PATH)
                        
                    # 3. Build Vector Store into the writable /content drive
                    vectorstore = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embeddings, 
                        persist_directory=CHROMA_PATH
                    )
                    st.success(f"Database initialized! Processed {len(documents)} pages into {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error building database: {e}")

# --- MAIN CHAT INTERFACE ---
st.subheader("Chat with the Research Papers")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the papers:"):
    if not openrouter_api_key:
        st.info("Please enter your OpenRouter API Key in the sidebar to chat.")
        st.stop()
        
    if not os.path.exists(CHROMA_PATH):
        st.info("Please initialize the database first using the sidebar button.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            # Using the free auto-router to bypass 429 rate limits
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
                model="openrouter/free",
                temperature=0.1
            )

            system_prompt = (
                "You are an expert AI assistant tasked with answering questions based on research papers. "
                "Use the following retrieved context to answer the user's question. "
                "If you don't know the answer or if the context does not provide enough information, explicitly state that you don't know to avoid hallucination. "
                "CRITICAL INSTRUCTION: You must cite the source of your information by mentioning the paper name/source and the page number at the end of your answer. "
                "\n\nContext:\n{context}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            try:
                sources = retriever.invoke(prompt)
                context_text = "\n\n".join(doc.page_content for doc in sources)
                rag_chain = prompt_template | llm | StrOutputParser()
                answer = rag_chain.invoke({"context": context_text, "input": prompt})
                
                st.markdown(answer)
                
                if sources:
                    with st.expander("View Retrieved Context & Sources"):
                        for i, doc in enumerate(sources):
                            source_file = os.path.basename(doc.metadata.get('source', 'Unknown File'))
                            page_num = doc.metadata.get('page', 'Unknown Page')
                            st.markdown(f"**Source {i+1}:** {source_file} (Page {page_num})")
                            st.text(doc.page_content[:300] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

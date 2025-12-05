import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from agent.config import embeddings_cohere

def process_document(uploaded_file, doc_type, chunk_size, chunk_overlap, pdf_temp_path, vector_store_base_path):
    """
    Procesa un archivo PDF cargado, lo divide en chunks y crea/guarda un vector store.
    """
    # Guardado temporal del archivo
    temp_file_path = os.path.join(pdf_temp_path, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Lógica de procesamiento
        loader = PyPDFLoader(temp_file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(loader.load_and_split())
        
        if chunks:
            ruta_vector_store = os.path.join(vector_store_base_path, doc_type)
            vector_store = FAISS.from_documents(chunks, embeddings_cohere)
            vector_store.save_local(ruta_vector_store)
            return len(chunks)
        else:
            return 0
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

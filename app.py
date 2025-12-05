# app.py

# IMPORTACIONES
import agent.config # Carga variables de entorno
import os
import streamlit as st
import uuid
from langchain_core.messages import AIMessage, HumanMessage

# Importaciones de módulos refactorizados
from agent.graph import workflow
from utils.ingestion import process_document

# Directorios
VECTOR_STORE_BASE_PATH = "vector_stores"
PDF_TEMP_PATH = "temp_pdf"
os.makedirs(PDF_TEMP_PATH, exist_ok=True)
os.makedirs(VECTOR_STORE_BASE_PATH, exist_ok=True)

# LÓGICA DE LA INTERFAZ DE USUARIO CON STREAMLIT

# Configuración de la página
st.set_page_config(page_title="Ingelect", page_icon="🤖", layout="wide")
st.title("Ingelect 🤖 - Engennier Assistant")

# Barra lateral para carga de documentos
with st.sidebar:
    st.header("Gestión de Documentos")
    
    doc_type = st.selectbox("Elige el tipo de documento a cargar", ["documento_01", "documento_02"])
    
    uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf", key=f"uploader_{doc_type}")
    
    # Expander para agrupar la configuración avanzada
    with st.expander("Configuración de Procesamiento (Chunking)"):
        
        # st.number_input para el tamaño del chunk
        chunk_size = st.number_input(
            "Tamaño del Chunk", 
            min_value=100, 
            max_value=6000, 
            value=500, 
            step=50, # El valor que se incrementa/decrementa con los botones
            help="El número máximo de caracteres en cada chunk. Afecta el detalle de la información."
        )

        # st.number_input para el overlap del chunk
        chunk_overlap = st.number_input(
            "Solapamiento de Chunks", 
            min_value=0, 
            max_value=2000, 
            value=125, 
            step=25, # El salto más pequeño aquí
            help="Cuántos caracteres se superponen entre chunks para mantener el contexto."
        )
        # ---------------------

    if st.button(f"Procesar Documento: {doc_type}"):
        if uploaded_file is not None:
            with st.spinner(f"Procesando '{uploaded_file.name}'..."):
                num_chunks = process_document(
                    uploaded_file, 
                    doc_type, 
                    chunk_size, 
                    chunk_overlap, 
                    PDF_TEMP_PATH, 
                    VECTOR_STORE_BASE_PATH
                )
                
                if num_chunks > 0:
                    st.success(f"¡'{doc_type}' procesado! ({num_chunks} chunks)")
                else:
                    st.error("No se pudo extraer texto del PDF.")
        else:
            st.warning("Por favor, sube un archivo PDF.")

# Interfaz de Chat
st.divider()

# Inicialización del thread_id y el historial de mensajes en el estado de la sesión de Streamlit
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes del historial guardado en la sesión de Streamlit
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Entrada de usuario
if prompt := st.chat_input("Pregunta sobre proyectos (doc_01) o procedimientos (doc_02)..."):
    # Añadir el mensaje del usuario a la lista de la UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            
            #    Se construye el historial de entrada para el grafo.
            #    Se convierte el historial de diccionarios de la UI al formato que LangGraph espera (HumanMessage, AIMessage).
            #    Esto asegura que el grafo reciba TODA la conversación anterior.
            input_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    input_messages.append(HumanMessage(content=msg["content"]))
                else:
                    input_messages.append(AIMessage(content=msg["content"]))

            #    Se prepara el input y la configuración para la llamada.
            #    El input contiene el historial completo.
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            input_data = {"messages": input_messages}
            
            #    Se invoca el workflow con el historial completo.
            final_state = workflow.invoke(input_data, config)
            
            #    Se obtiene solo la última respuesta (la que acaba de generar el agente).
            response_content = final_state["messages"][-1].content
            
            #    Se muestra la respuesta y se añade al historial de la UI.
            st.write(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})
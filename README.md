# Ingelect 🤖 - Engineer Assistant

Sistema de asistente inteligente basado en RAG (Retrieval-Augmented Generation) que permite consultar documentos técnicos mediante un chat interactivo construido con Streamlit y LangGraph.

## 🚀 Características

- **Arquitectura Multi-Agente**: Sistema supervisor que delega consultas a agentes especializados
- **RAG con FAISS**: Búsqueda semántica eficiente en documentos PDF
- **Interfaz Streamlit**: UI interactiva y fácil de usar
- **Modular y Escalable**: Código organizado en módulos independientes
- **Configuración Flexible**: Parámetros ajustables de chunking para optimizar resultados

## 📋 Requisitos

- Python 3.12+
- API Keys:
  - Google API Key (para Gemini)
  - Cohere API Key (para embeddings)

## 🛠️ Instalación

1. **Clonar el repositorio**
```bash
git clone <repository-url>
cd workflow_delegacion_sintesis_streamlit
```

2. **Crear entorno virtual**
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**

Crear un archivo `.env` en la raíz del proyecto:
```env
GOOGLE_API_KEY=tu_google_api_key
COHERE_API_KEY=tu_cohere_api_key
```

## 🏗️ Estructura del Proyecto

```
workflow_delegacion_sintesis_streamlit/
├── app.py                      # Interfaz Streamlit (Frontend)
├── agent/
│   ├── config.py              # Configuración de LLM y embeddings
│   ├── graph.py               # Construcción del grafo LangGraph
│   ├── nodes.py               # Lógica de los nodos del grafo
│   ├── state.py               # Definición del estado
│   ├── supervisor.py          # Modelo de decisión del supervisor
│   ├── specialists.py         # Agentes especializados
│   └── tools/
│       └── rag_tools.py       # Herramientas de búsqueda RAG
├── utils/
│   └── ingestion.py           # Procesamiento de documentos
├── .env                       # Variables de entorno (no versionado)
├── .gitignore
├── requirements.txt
└── README.md
```

## 🎯 Uso

1. **Iniciar la aplicación**
```bash
streamlit run app.py
```

2. **Cargar documentos**
   - Selecciona el tipo de documento (`documento_01` o `documento_02`)
   - Sube un archivo PDF
   - Ajusta los parámetros de chunking si es necesario
   - Haz clic en "Procesar Documento"

3. **Hacer consultas**
   - Escribe tu pregunta en el chat
   - El sistema enrutará automáticamente tu consulta al agente especializado correcto
   - Recibe respuestas basadas en el contenido de los documentos

## 🔧 Tecnologías Utilizadas

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: Cohere (embed-english-light-v3.0)
- **Vector Store**: FAISS
- **Framework de Agentes**: LangGraph
- **Procesamiento de PDFs**: PyPDF

## 📝 Configuración Avanzada

### Parámetros de Chunking

- **Tamaño del Chunk**: 100-6000 caracteres (default: 500)
- **Solapamiento**: 0-2000 caracteres (default: 125)

Ajusta estos valores según la naturaleza de tus documentos para optimizar la recuperación de información.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría realizar.

## 📄 Licencia

Este proyecto es de uso educativo.

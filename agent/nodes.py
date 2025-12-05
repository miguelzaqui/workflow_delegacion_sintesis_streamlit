from typing import Dict
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from agent.state import GraphState
from agent.supervisor import DecisionSupervisor
from agent.specialists import agente_documento_01, agente_documento_02
from agent.config import llm_gemini

# Nodos del Grafo (supervisor_router, specialist_executor, supervisor_synthesizer)
def supervisor_router_node(state: GraphState) -> Dict:
    print(">> Supervisor (Router): Evaluando a quién delegar...")
    llm_router = llm_gemini.with_structured_output(DecisionSupervisor)
    prompt = SystemMessage(content="""Eres un supervisor de un equipo de agentes. Analiza la consulta del usuario y enrútala al especialista correcto.

Reglas de Enrutamiento:
- 'agente_documento_01': Para preguntas sobre el directorio de proyectos de ingeniería.
- 'agente_documento_02': Para preguntas sobre el manual de procedimientos.
- Si el usuario te pide que verifiques o corrobores información, DEBES enrutar la tarea al especialista correspondiente.
- 'finalizar': Si el usuario se despide.
        """)
    decision = llm_router.invoke([prompt] + state["messages"])
    print(f"   Decisión: Delegar a '{decision.siguiente_agente}'")
    return {"siguiente_agente": decision.siguiente_agente}

def specialist_executor_node(state: GraphState) -> Dict:
    agent_name = state["siguiente_agente"]
    print(f">> Ejecutando Especialista: {agent_name}")
    specialist = {"agente_documento_01": agente_documento_01, "agente_documento_02": agente_documento_02}.get(agent_name)
    if not specialist: raise ValueError(f"Agente desconocido: {agent_name}")
    
    # Llamada síncrona '.invoke()' con el historial completo
    result = specialist.invoke({"messages": state["messages"]})
    
    informe = result['messages'][-1].content
    print(f"   Informe del especialista: {informe[:300]}...")
    return {"informe_especialista": informe}

def supervisor_synthesizer_node(state: GraphState) -> Dict:
    print(">> Supervisor (Synthesizer): Formulando respuesta final...")
    informe = state["informe_especialista"]
    pregunta_usuario = state["messages"][-1].content
    print(f"   Pregunta del usuario: {pregunta_usuario}")
    template = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente de atención al cliente. Tu única función es tomar la información técnica recuperada por un especialista y presentarla de forma clara y amable al usuario.

Sigue estas reglas de forma OBLIGATORIA:
1.  Basa tu respuesta EXCLUSIVAMENTE en la "Información recuperada por el especialista". NO uses conocimiento previo ni inventes datos.
2.  Si el informe del especialista indica que no se encontró información, tu única respuesta debe ser informar al usuario de que no se encontró la información en el documento.
3.  Si la pregunta del usuario es una continuación de la conversación (p. ej., "y sobre ella?", "dime más"), usa el "Historial de la conversación" para entender el contexto, pero la información para la respuesta DEBE provenir únicamente del informe actual del especialista.
4.  Mantén siempre un tono servicial y dirígete al usuario por su nombre si lo conoces por el historial.
5.  NO justifiques ni inventes explicaciones si los datos del usuario contradicen los del especialista. Simplemente presenta la información del especialista como la fuente de verdad. Por ejemplo, si el usuario dice "La puntuación es X" y el informe dice "La puntuación es Y", tu respuesta debe ser "Según mis datos, la puntuación es Y."."""),
        MessagesPlaceholder(variable_name="historial_chat"),
        ("user", f"Pregunta del usuario:\n{pregunta_usuario}\n\nInformación recuperada por el especialista:\n{informe}")
    ])
    synthesis_chain = template | llm_gemini
    response = synthesis_chain.invoke({
        "historial_chat": state["messages"],
        "informe_tecnico": informe
    })
    print(f"   Respuesta final generada: {response.content[:300]}...")
    return {"messages": [response]}

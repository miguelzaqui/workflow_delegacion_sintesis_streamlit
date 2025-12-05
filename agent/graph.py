from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from agent.state import GraphState
from agent.nodes import supervisor_router_node, specialist_executor_node, supervisor_synthesizer_node

# Ensamblado del grafo
builder = StateGraph(GraphState)
builder.add_node("supervisor_router", supervisor_router_node)
builder.add_node("specialist_executor", specialist_executor_node)
builder.add_node("supervisor_synthesizer", supervisor_synthesizer_node)

builder.set_entry_point("supervisor_router")
builder.add_conditional_edges(
    "supervisor_router",
    lambda state: state["siguiente_agente"],
    {"agente_documento_01": "specialist_executor", "agente_documento_02": "specialist_executor", "finalizar": END}
)
builder.add_edge("specialist_executor", "supervisor_synthesizer")
builder.add_edge("supervisor_synthesizer", END)

checkpointer = InMemorySaver()
workflow = builder.compile(checkpointer=checkpointer)
print("Grafo Supervisor Multi-RAG (Delegación/Síntesis) compilado y listo para Streamlit.")

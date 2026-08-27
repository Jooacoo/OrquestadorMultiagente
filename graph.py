"""
graph.py
--------
Arma el StateGraph jerárquico: Supervisor (router inteligente) +
2 especialistas (researcher, analyst).

Diseño clave:
- El Supervisor es el ÚNICO nodo que decide el flujo. Nunca ejecuta
  trabajo de investigación o análisis él mismo (evita el error
  "Supervisor Todólogo").
- La decisión de ruteo usa salida estructurada con Pydantic
  (`Route`, con `next: Literal[...]`) en vez de parsear texto libre:
  así el grafo nunca recibe un nombre de nodo inválido.
- Cada especialista recibe solo la `instruction` puntual que arma el
  Supervisor, no todo el historial de mensajes (evita "Contaminación
  de Contexto"). El resultado del especialista sí se agrega al
  historial, para que el Supervisor pueda leerlo en la próxima vuelta.
- `iteration_count` + `MAX_ITERATIONS` cortan el "Supervisor Infinito".
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agents.analyst_agent import build_analyst_agent
from agents.research_agent import build_research_agent
from state import MAX_ITERATIONS, OrchestratorState

SUPERVISOR_PROMPT = """Sos el Supervisor de un orquestador multi-agente jerárquico.
No investigás ni analizás nada vos mismo: tu único trabajo es decidir quién actúa
a continuación, y con qué instrucción puntual, o si ya hay suficiente información
para finalizar.

Especialistas disponibles:
- researcher: busca información externa (no analiza ni calcula).
- analyst: procesa datos ya recolectados (sentimiento, estadísticas, validación
  de esquemas). No sale a buscar información nueva.

Pedido original del usuario:
{user_request}

Estado actual:
- research_results: {research_results}
- analysis_results: {analysis_results}
- vuelta número: {iteration_count} de {max_iterations}

Rúbrica de suficiencia para finalizar (FINISH):
- Hay al menos un hallazgo de research_results relevante al pedido, Y
- Hay al menos un resultado de analysis_results que procese ese hallazgo.
- Si falta investigación, mandá a "researcher".
- Si hay investigación pero falta procesarla, mandá a "analyst".
- Si ambas partes están cubiertas y son coherentes entre sí, elegí "FINISH".
- Si se llegó a la última vuelta permitida, elegí "FINISH" aunque esté incompleto.

Cuando rutees a un especialista, escribí una instrucción concreta y autocontenida
(no le va a llegar el resto de la conversación, solo esa instrucción).
"""


class Route(BaseModel):
    """Salida estructurada del Supervisor: valida el nombre del nodo destino."""

    next: Literal["researcher", "analyst", "FINISH"] = Field(
        description="Próximo nodo a ejecutar, o FINISH si la tarea está completa."
    )
    instruction: str = Field(
        default="",
        description="Instrucción puntual y autocontenida para el especialista elegido. Vacío si next=FINISH.",
    )
    rationale: str = Field(
        description="Justificación breve (1 línea) de por qué se rutea así, para trazabilidad."
    )


def build_graph(
    llm: BaseChatModel,
    *,
    supervisor_llm: BaseChatModel | None = None,
    research_llm: BaseChatModel | None = None,
    analyst_llm: BaseChatModel | None = None,
    supervisor_router=None,
):
    """`llm` es el modelo por defecto para los 3 roles (Supervisor +
    2 especialistas). Se puede pasar un modelo distinto por rol.

    `supervisor_router`, si se pasa, reemplaza directamente a
    `llm.with_structured_output(Route)` (cualquier Runnable con
    `.invoke(...) -> Route` sirve). Lo usa demo.py para ensayar el
    flujo completo con decisiones de ruteo pre-cargadas, sin pegarle
    a una API real."""
    research_agent = build_research_agent(research_llm or llm)
    analyst_agent = build_analyst_agent(analyst_llm or llm)
    structured_supervisor = supervisor_router or (supervisor_llm or llm).with_structured_output(Route)

    # ---------------- Nodo Supervisor ----------------
    def supervisor_node(state: OrchestratorState) -> dict:
        iteration_count = state.get("iteration_count", 0) + 1
        user_request = state["messages"][0].content

        if iteration_count > MAX_ITERATIONS:
            # Criterio de "Suficiencia" estricto: cortamos el bucle sí o sí.
            return {
                "next_agent": "FINISH",
                "task_completed": True,
                "iteration_count": iteration_count,
                "messages": [
                    AIMessage(
                        content="[Supervisor] Límite de iteraciones alcanzado. "
                        "Cierro con la mejor información disponible."
                    )
                ],
            }

        prompt = SUPERVISOR_PROMPT.format(
            user_request=user_request,
            research_results=state.get("research_results") or "(sin datos todavía)",
            analysis_results=state.get("analysis_results") or "(sin datos todavía)",
            iteration_count=iteration_count,
            max_iterations=MAX_ITERATIONS,
        )
        route: Route = structured_supervisor.invoke([HumanMessage(content=prompt)])

        return {
            "next_agent": route.next,
            "task_completed": route.next == "FINISH",
            "iteration_count": iteration_count,
            "messages": [
                AIMessage(
                    content=f"[Supervisor] -> {route.next} | {route.rationale}"
                    + (f" | instrucción: {route.instruction}" if route.instruction else "")
                )
            ],
        }

    # ---------------- Nodo Investigador ----------------
    def research_node(state: OrchestratorState) -> dict:
        instruction = _last_instruction(state)
        result = research_agent.invoke({"messages": [HumanMessage(content=instruction)]})
        summary = result["messages"][-1].content
        contribution = {
            "agent": "researcher",
            "summary": summary,
            "iteration": state["iteration_count"],
        }
        return {
            "research_results": summary,
            "contributions": state.get("contributions", []) + [contribution],
            "messages": [AIMessage(content=f"[Investigador] {summary}")],
        }

    # ---------------- Nodo Analista ----------------
    def analyst_node(state: OrchestratorState) -> dict:
        instruction = _last_instruction(state)
        context = state.get("research_results") or "(no hay datos de investigación aún)"
        full_instruction = f"{instruction}\n\nDatos disponibles para analizar:\n{context}"
        result = analyst_agent.invoke({"messages": [HumanMessage(content=full_instruction)]})
        summary = result["messages"][-1].content
        contribution = {
            "agent": "analyst",
            "summary": summary,
            "iteration": state["iteration_count"],
        }
        return {
            "analysis_results": summary,
            "contributions": state.get("contributions", []) + [contribution],
            "messages": [AIMessage(content=f"[Analista] {summary}")],
        }

    def _last_instruction(state: OrchestratorState) -> str:
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content.startswith("[Supervisor]"):
                if "instrucción:" in msg.content:
                    return msg.content.split("instrucción:", 1)[1].strip()
        # fallback: si el supervisor no dio instrucción explícita, usar el pedido original
        return state["messages"][0].content

    def route_after_supervisor(state: OrchestratorState) -> Literal["researcher", "analyst", "FINISH"]:
        return state["next_agent"]

    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", research_node)
    graph.add_node("analyst", analyst_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"researcher": "researcher", "analyst": "analyst", "FINISH": END},
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")

    return graph

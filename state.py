"""
state.py
--------
Esquema de estado compartido para el Orquestador Multi-Agente.

Hereda de MessagesState (historial de chat) y agrega los campos que
el Supervisor necesita para rutear y para saber cuándo cortar el flujo:

- next_agent: a quién le toca actuar ahora (o "FINISH").
- task_completed: bandera de salida limpia.
- research_results / analysis_results: "parcelas" de estado, cada una
  escrita únicamente por su agente dueño (evita condiciones de carrera:
  Investigador nunca toca analysis_results y viceversa).
- contributions: bitácora de quién aportó qué, para poder auditar el
  flujo (requerimiento de "rastrear qué agente ha contribuido con qué
  información").
- iteration_count: contador de vueltas por el supervisor, para cortar
  el patrón "Supervisor Infinito".
"""

from typing import Literal, Optional, TypedDict
from langgraph.graph import MessagesState


# Nombres válidos de nodo. Definir esto como Literal es lo que nos permite
# tipar el retorno de la función de ruteo del supervisor y que LangGraph
# valide las aristas condicionales en tiempo de construcción del grafo.
AgentName = Literal["researcher", "analyst", "FINISH"]

MAX_ITERATIONS = 6  # criterio de "Suficiencia" estricto (ver README, Error 1)


class Contribution(TypedDict):
    agent: str
    summary: str
    iteration: int


class OrchestratorState(MessagesState):
    """Estado compartido del grafo. Extiende MessagesState (historial)."""

    next_agent: Optional[AgentName]
    task_completed: bool
    research_results: Optional[str]
    analysis_results: Optional[str]
    contributions: list[Contribution]
    iteration_count: int


def initial_state(user_request: str) -> OrchestratorState:
    """Crea el estado inicial a partir del pedido del usuario."""
    from langchain_core.messages import HumanMessage

    return OrchestratorState(
        messages=[HumanMessage(content=user_request)],
        next_agent=None,
        task_completed=False,
        research_results=None,
        analysis_results=None,
        contributions=[],
        iteration_count=0,
    )

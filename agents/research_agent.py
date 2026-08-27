"""
agents/research_agent.py
-------------------------
Agente de Búsqueda/Investigación. Su única responsabilidad es consultar
fuentes externas y devolver hallazgos: no analiza, no calcula, no valida.

Herramienta: Tavily real si hay TAVILY_API_KEY en el entorno; si no,
cae a la búsqueda simulada sobre la Vector DB local (agents/tools.py),
tal como habilita la consigna ("Tavily o una búsqueda simulada").
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from agents.tools import vector_db_search

RESEARCH_PROMPT = """Sos el Agente de Investigación de un orquestador multi-agente.
Tu único trabajo es buscar información relevante usando tu herramienta de búsqueda
y resumir los hallazgos de forma concisa y factual.

Reglas:
- No hagas análisis, cálculos ni validaciones: eso es trabajo del Agente de Análisis.
- No inventes datos que no vengan de la herramienta de búsqueda.
- Devolvé un resumen breve (3-5 líneas) con los datos más relevantes que encontraste.
"""


def _get_search_tool():
    if os.environ.get("TAVILY_API_KEY"):
        # Fuente externa real, tal como sugiere la consigna.
        from langchain_community.tools.tavily_search import TavilySearchResults

        return TavilySearchResults(max_results=3)
    # Fallback: búsqueda simulada sobre la Vector DB local del curso.
    return vector_db_search


def build_research_agent(llm: BaseChatModel):
    """Construye el agente especialista de investigación con create_react_agent,
    con una herramienta acotada (solo búsqueda)."""
    tools = [_get_search_tool()]
    return create_react_agent(llm, tools=tools, prompt=RESEARCH_PROMPT, name="researcher")

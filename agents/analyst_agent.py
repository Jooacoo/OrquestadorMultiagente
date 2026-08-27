"""
agents/analyst_agent.py
-------------------------
Agente de Análisis/Cómputo. Procesa los datos que ya trajo el Investigador:
análisis de sentimiento, cálculos matemáticos o validación de esquemas.
No sale a buscar información por su cuenta.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from agents.tools import analyze_sentiment, compute_stats, validate_schema

ANALYST_PROMPT = """Sos el Agente de Análisis de un orquestador multi-agente.
Recibís datos ya recolectados (por lo general del Agente de Investigación) y tu
trabajo es procesarlos: análisis de sentimiento, cálculos estadísticos o
validación de esquemas, según lo que pida la tarea.

Reglas:
- No busques información nueva: trabajá solo con los datos que te pasaron.
- Elegí la herramienta que corresponda a la tarea (sentimiento, stats o schema).
- Devolvé una conclusión breve y accionable, no solo el output crudo de la herramienta.
"""


def build_analyst_agent(llm: BaseChatModel):
    """Construye el agente especialista de análisis con create_react_agent,
    con herramientas acotadas a procesamiento (no búsqueda)."""
    tools = [analyze_sentiment, compute_stats, validate_schema]
    return create_react_agent(llm, tools=tools, prompt=ANALYST_PROMPT, name="analyst")

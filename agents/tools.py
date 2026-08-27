"""
agents/tools.py
----------------
Herramientas funcionales para los dos especialistas.

Investigador:
    - vector_db_search: búsqueda simulada sobre una "Vector DB" local
      (TF-IDF + similitud coseno, igual que en las pre-entregas de
      embeddings del curso) que hace de stand-in de Tavily/una fuente
      externa. Si existe TAVILY_API_KEY en el entorno, se usa
      TavilySearchResults real en su lugar (ver research_agent.py).

Analista:
    - analyze_sentiment: análisis de sentimiento simple basado en léxico.
    - compute_stats: cálculos matemáticos (media, desvío, etc.) sobre listas.
    - validate_schema: valida un JSON contra un esquema Pydantic dinámico.
"""

from __future__ import annotations

import json
import statistics
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, ValidationError, create_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# "Vector DB" local: reutiliza el corpus de las pre-entregas anteriores
# del curso (microservicios / despliegue) como stand-in de una fuente
# externa real. En producción esto se reemplaza por Tavily o por la
# ChromaDB/Pinecone ya construidos en las Pre-entregas 3 y 4.
# ---------------------------------------------------------------------
_DOCS = [
    "El despliegue de microservicios con contenedores mejora la escalabilidad horizontal.",
    "Kubernetes orquesta el ciclo de vida de contenedores en clústeres distribuidos.",
    "El patrón Supervisor centraliza el control en sistemas multi-agente jerárquicos.",
    "La topología cooperativa usa un pizarrón compartido para coordinar agentes pares.",
    "El uso de RAG combina recuperación de documentos con generación de texto.",
    "La similitud coseno mide el ángulo entre vectores de embeddings, no su magnitud.",
    "Un chunking demasiado grande diluye la relevancia semántica de cada fragmento.",
    "LangGraph permite ciclos y aristas condicionales, a diferencia de una cadena lineal.",
    "El chequeo de esquemas con Pydantic evita que datos alucinados contaminen el estado.",
    "Asyncio permite procesar múltiples agentes sin bloquear el hilo principal.",
]

_vectorizer = TfidfVectorizer().fit(_DOCS)
_doc_matrix = _vectorizer.transform(_DOCS)


@tool
def vector_db_search(query: str, top_k: int = 3) -> str:
    """Busca en la Vector DB local (simulada) los documentos más relevantes
    para la query, usando similitud coseno sobre embeddings TF-IDF.
    Devuelve los top_k resultados con su score, como stand-in de una
    búsqueda externa (Tavily) cuando no hay API key configurada."""
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _doc_matrix)[0]
    ranked = sorted(zip(_DOCS, scores), key=lambda x: x[1], reverse=True)[:top_k]
    lines = [f"[{score:.2f}] {doc}" for doc, score in ranked]
    return "Resultados (Vector DB simulada):\n" + "\n".join(lines)


# ---------------------------------------------------------------------
# Herramientas del Analista
# ---------------------------------------------------------------------
_POSITIVE_WORDS = {
    "mejora", "escalabilidad", "eficiente", "resiliente", "flexible",
    "robusto", "clara", "claro", "beneficio", "ventaja", "óptimo",
    "fácil", "rápido", "estable",
}
_NEGATIVE_WORDS = {
    "cuello", "bottleneck", "falla", "riesgo", "caos", "bucle",
    "infinito", "error", "lento", "costoso", "complejo", "inconsistente",
}


@tool
def analyze_sentiment(text: str) -> str:
    """Análisis de sentimiento simple basado en léxico (positivo/negativo/neutro)
    sobre un texto en español. Devuelve la polaridad y las palabras clave detectadas."""
    words = {w.strip(".,;:()").lower() for w in text.split()}
    pos = words & _POSITIVE_WORDS
    neg = words & _NEGATIVE_WORDS
    if len(pos) > len(neg):
        polarity = "positivo"
    elif len(neg) > len(pos):
        polarity = "negativo"
    else:
        polarity = "neutro"
    return (
        f"Polaridad: {polarity} | señales positivas: {sorted(pos) or '—'} "
        f"| señales negativas: {sorted(neg) or '—'}"
    )


@tool
def compute_stats(numbers: list[float]) -> str:
    """Calcula estadísticas descriptivas (media, mediana, desvío estándar,
    mínimo y máximo) sobre una lista de números."""
    if not numbers:
        return "Lista vacía: no hay estadísticas para calcular."
    mean = statistics.mean(numbers)
    median = statistics.median(numbers)
    stdev = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
    return (
        f"n={len(numbers)} | media={mean:.2f} | mediana={median:.2f} "
        f"| desvío={stdev:.2f} | min={min(numbers)} | max={max(numbers)}"
    )


@tool
def validate_schema(data_json: str, required_fields_json: str) -> str:
    """Valida que un JSON (data_json) contenga los campos requeridos
    (required_fields_json, ej. '{"titulo": "str", "score": "float"}') usando
    un modelo Pydantic generado dinámicamente. Devuelve OK o el detalle del error,
    para evitar que datos alucinados por otro agente contaminen el estado."""
    type_map = {"str": str, "float": float, "int": int, "bool": bool}
    try:
        data = json.loads(data_json)
        required_fields = json.loads(required_fields_json)
    except json.JSONDecodeError as e:
        return f"JSON inválido: {e}"

    fields: dict[str, Any] = {
        name: (type_map.get(type_name, str), ...)
        for name, type_name in required_fields.items()
    }
    DynamicModel: type[BaseModel] = create_model("DynamicSchema", **fields)  # type: ignore

    try:
        DynamicModel(**data)
        return "OK: el JSON cumple el esquema requerido."
    except ValidationError as e:
        return f"Esquema inválido: {e.errors()}"

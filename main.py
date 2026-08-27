"""
main.py
-------
Punto de entrada. Arma el LLM (Gemini) y compila el grafo con un
checkpointer sqlite, igual que en la Pre-entrega 5, para poder retomar
un thread_id si hiciera falta.

Uso:
    export GOOGLE_API_KEY="tu-api-key"
    python main.py "¿Qué ventajas y riesgos tiene la topología jerárquica?"

Sin GOOGLE_API_KEY seteada, avisa cómo correr la demo en modo mock
(ver demo.py), que no requiere API key y genera una traza real de ejecución.
"""

import os
import sys

from state import initial_state

try:
    from dotenv import load_dotenv

    load_dotenv()  # lee el archivo .env de esta carpeta y carga sus variables
except ImportError:
    pass  # si no está instalado python-dotenv, seguimos con las variables de entorno normales


def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta GOOGLE_API_KEY en el entorno. Para correr con Gemini real:\n"
            "  export GOOGLE_API_KEY='tu-api-key'\n"
            "Para probar el flujo sin API key (modo mock con LLM simulado), "
            "corré: python demo.py"
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        max_retries=5,  
    )


def run(user_request: str, thread_id: str = "default-thread"):
    from graph import build_graph
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3

    llm = get_llm()
    graph = build_graph(llm)

    conn = sqlite3.connect("orchestrator_memory.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    final_state = app.invoke(initial_state(user_request), config=config)

    print("\n=== Trazo de contribuciones ===")
    for c in final_state["contributions"]:
        print(f"[vuelta {c['iteration']}] {c['agent']}: {c['summary'][:200]}")

    print("\n=== Respuesta final ===")
    print(final_state["messages"][-1].content)
    return final_state


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or (
        "Investigá qué ventajas tiene la topología jerárquica frente a la "
        "cooperativa y analizá el sentimiento de esa comparación."
    )
    run(query)
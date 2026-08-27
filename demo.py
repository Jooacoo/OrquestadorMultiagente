"""
demo.py
-------
Demo del flujo de delegación que corre SIN necesidad de GOOGLE_API_KEY.

No mockea el grafo ni las herramientas: usa el StateGraph real
(graph.py) y las tools reales (vector_db_search, analyze_sentiment,
etc. en agents/tools.py). Lo único "de mentira" es el LLM: se
reemplaza por `ScriptedLLM`/`ScriptedRouter`, que devuelven una
secuencia de respuestas pre-cargadas en vez de llamar a una API
(mismo patrón que un test de integración con un fake chat model).

Sirve para demostrar que el cableado del grafo funciona de punta a
punta (Supervisor -> researcher -> Supervisor -> analyst -> Supervisor
-> FINISH) y deja una traza real de ejecución en demo_trace.json.

Para correr contra Gemini de verdad: `python main.py "tu pregunta"`.
"""

import json
from collections import deque
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from graph import Route, build_graph
from state import initial_state


class ScriptedLLM(BaseChatModel):
    """LLM de juguete: devuelve, en orden, los AIMessage precargados en
    `script`. Ignora bind_tools (las tool_calls ya vienen adentro de
    cada AIMessage scripteado), así que sirve como `model` dentro de
    create_react_agent sin llamar a ninguna API real."""

    script: Any = None

    def __init__(self, script: list[AIMessage], **kwargs: Any):
        super().__init__(script=deque(script), **kwargs)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake-llm"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        if not self.script:
            raise RuntimeError("ScriptedLLM se quedó sin respuestas pre-cargadas.")
        message = self.script.popleft()
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self  # no-op: las tool_calls ya están hardcodeadas en el script


class ScriptedRouter(Runnable):
    """Reemplaza a `llm.with_structured_output(Route)` del Supervisor:
    devuelve, en orden, los objetos Route precargados."""

    def __init__(self, routes: list[Route]):
        self.routes = deque(routes)

    def invoke(self, *_args, **_kwargs) -> Route:
        if not self.routes:
            raise RuntimeError("ScriptedRouter se quedó sin decisiones pre-cargadas.")
        return self.routes.popleft()


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def build_demo_app():
    supervisor_router = ScriptedRouter(
        [
            Route(
                next="researcher",
                instruction=(
                    "Buscá información sobre la topología jerárquica (patrón Supervisor) "
                    "en sistemas multi-agente y sus ventajas frente a la topología cooperativa."
                ),
                rationale="Todavía no hay datos de investigación en el estado.",
            ),
            Route(
                next="analyst",
                instruction=(
                    "Analizá el sentimiento del resumen de investigación sobre la "
                    "topología jerárquica (ventajas y riesgos mencionados)."
                ),
                rationale="Ya hay investigación disponible; falta procesarla.",
            ),
            Route(
                next="FINISH",
                instruction="",
                rationale="research_results y analysis_results están completos y son coherentes entre sí.",
            ),
        ]
    )

    research_llm = ScriptedLLM(
        [
            _tool_call(
                "vector_db_search",
                {"query": "topología jerárquica supervisor multi-agente ventajas", "top_k": 3},
                "call_research_1",
            ),
            AIMessage(
                content=(
                    "Hallazgo: la topología jerárquica centraliza el control en un nodo "
                    "Supervisor, lo que facilita auditar decisiones y modularizar a los "
                    "especialistas, pero introduce un posible cuello de botella si el "
                    "Supervisor falla o se satura."
                )
            ),
        ]
    )

    analyst_llm = ScriptedLLM(
        [
            _tool_call(
                "analyze_sentiment",
                {
                    "text": (
                        "la topologia jerarquica centraliza el control facilita auditar "
                        "decisiones modulariza especialistas pero introduce cuello de "
                        "botella si falla"
                    )
                },
                "call_analyst_1",
            ),
            AIMessage(
                content=(
                    "Conclusión: el balance es mixto — hay señales positivas claras "
                    "(control, modularidad) junto con un riesgo explícito (cuello de "
                    "botella), así que la recomendación es monitorear la carga del "
                    "Supervisor en producción."
                )
            ),
        ]
    )

    graph = build_graph(
        research_llm,  # llm "default": no se usa porque cada rol tiene el propio
        research_llm=research_llm,
        analyst_llm=analyst_llm,
        supervisor_router=supervisor_router,
    )
    return graph.compile()


def run_demo(user_request: str) -> dict:
    app = build_demo_app()
    return app.invoke(initial_state(user_request), config={"recursion_limit": 25})


if __name__ == "__main__":
    query = (
        "Investigá qué ventajas tiene la topología jerárquica frente a la "
        "cooperativa y analizá el sentimiento de esa comparación."
    )
    print(f"Pedido del usuario:\n  {query}\n")

    final_state = run_demo(query)

    print("=== Traza de ejecución (quién contribuyó qué, en orden) ===")
    for c in final_state["contributions"]:
        print(f"  vuelta {c['iteration']} | {c['agent']}: {c['summary'][:160]}...")

    print(f"\nVueltas totales del Supervisor: {final_state['iteration_count']}")
    print(f"task_completed: {final_state['task_completed']}")

    print("\n=== Historial completo de mensajes ===")
    for m in final_state["messages"]:
        role = m.__class__.__name__
        print(f"  [{role}] {m.content}")

    trace = {
        "user_request": query,
        "iteration_count": final_state["iteration_count"],
        "task_completed": final_state["task_completed"],
        "research_results": final_state["research_results"],
        "analysis_results": final_state["analysis_results"],
        "contributions": final_state["contributions"],
        "messages": [
            {"role": m.__class__.__name__, "content": m.content} for m in final_state["messages"]
        ],
    }
    with open("demo_trace.json", "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    print("\nTraza guardada en demo_trace.json")

# Orquestador Multi-Agente de Análisis e Investigación

Pre-entrega 6 — Módulo 6 (Sistemas multi-agente: colaboración y especialización).
Prototipo funcional de un orquestador **jerárquico** con un nodo Supervisor y
dos agentes especialistas (Investigador y Analista), implementado con LangGraph.

## Topología elegida: Jerárquica (patrón Supervisor)

Elegí jerarquía y no colaboración peer-to-peer por tres razones, tal como las
plantea el material del módulo:

1. **El workflow es semi-determinista.** El pedido siempre implica "buscar
   algo -> procesar ese algo -> responder", no una exploración abierta donde
   convenga que los agentes reaccionen entre sí sin orden fijo.
2. **Necesito un punto único de validación.** El Supervisor es el único que
   decide si research_results + analysis_results alcanzan para cerrar
   (`FINISH`) o si falta una vuelta más. En una topología cooperativa esa
   responsabilidad quedaría difusa entre los pares.
3. **Especialización de modelos.** El Supervisor es el único nodo que razona
   sobre "quién sigue", así que en producción puede correr un modelo más caro
   mientras los especialistas usan modelos más chicos — algo natural en
   jerarquía y forzado en colaboración.

El trade-off que acepto a cambio: el Supervisor es un cuello de botella (si
falla, todo el grafo se frena) y su carga cognitiva crece con cada
especialista nuevo. Lo mitigo limitando estrictamente su prompt a "rutear y
validar" (nunca "ejecutar", ver Error 1 más abajo) y con un tope duro de
iteraciones.

## Cómo se maneja el estado compartido (`state.py`)

- `OrchestratorState` hereda de `MessagesState` y agrega `next_agent`,
  `task_completed`, `research_results`, `analysis_results`, `contributions`
  e `iteration_count`.
- **Aislamiento de escritura**: `researcher` solo escribe
  `research_results`; `analyst` solo escribe `analysis_results`. Ningún
  agente pisa la parcela del otro (evita condiciones de carrera, tal como
  recomienda el módulo).
- **Trazabilidad**: cada nodo especialista agrega una entrada a
  `contributions` (`{agent, summary, iteration}`), así queda registro de
  quién aportó qué y en qué vuelta — el requerimiento explícito de "rastrear
  qué agente ha contribuido con qué información, evitando la pérdida de
  contexto".
- **Validación con Pydantic**: el Supervisor nunca devuelve texto libre para
  rutear. Usa `llm.with_structured_output(Route)`, donde `Route.next` es un
  `Literal["researcher", "analyst", "FINISH"]`. Si el LLM "alucina" un nombre
  de nodo inválido, Pydantic lo rechaza antes de que llegue al grafo.

## Cómo se manejan los conflictos entre agentes

- **Contaminación de contexto**: los especialistas no reciben todo el
  historial de mensajes, solo la `instruction` puntual que arma el
  Supervisor para esa vuelta (`agents/research_agent.py`,
  `agents/analyst_agent.py` reciben un `HumanMessage` nuevo, no
  `state["messages"]` completo). El Analista sí recibe explícitamente
  `research_results` como contexto de trabajo, porque su tarea depende de
  eso — es la única dependencia intencional entre especialistas.
- **Supervisor Infinito**: `MAX_ITERATIONS = 6` en `state.py`. Si el
  Supervisor llega a esa vuelta sin haber decidido `FINISH`, el nodo
  `supervisor_node` fuerza el cierre igual (`next_agent = "FINISH"`), en vez
  de confiar ciegamente en que el LLM corte el bucle por su cuenta.
- **Supervisor Todólogo**: el prompt del Supervisor prohíbe explícitamente
  que investigue o analice; su única salida posible es la estructura `Route`
  (next + instruction + rationale), no texto de respuesta al usuario.
- **Rúbrica de suficiencia**: el prompt del Supervisor define por escrito
  cuándo dar `FINISH` (hay research_results relevante Y hay
  analysis_results que lo procesa y es coherente con él), en vez de dejar el
  criterio de corte librado a la intuición del modelo en cada llamada.

## Estructura del repositorio

```
state.py                 # Esquema de estado compartido (OrchestratorState)
graph.py                 # StateGraph: nodo Supervisor + aristas condicionales
agents/
  research_agent.py       # Agente de Investigación (create_react_agent)
  analyst_agent.py         # Agente de Análisis (create_react_agent)
  tools.py                 # vector_db_search, analyze_sentiment, compute_stats, validate_schema
main.py                  # Entry point real: Gemini + checkpointer sqlite (thread_id)
demo.py                  # Demo/traza SIN API key (LLM scripteado, tools reales)
requirements.txt
.env.example
```

## Cómo correrlo

### 1. Demo sin API key (deja una traza real de ejecución)

```bash
pip install -r requirements.txt
python demo.py
```

Esto corre el **grafo real** (mismo `graph.py` de producción) con las
**herramientas reales** (`vector_db_search` con TF-IDF + similitud coseno
sobre una Vector DB local, `analyze_sentiment` léxico), reemplazando
únicamente el LLM por `ScriptedLLM`/`ScriptedRouter` (mismo patrón que un
test de integración con un *fake chat model*). Imprime la traza completa de
delegación y la guarda en `demo_trace.json`.

Salida esperada (resumen):

```
[Supervisor] -> researcher | Todavía no hay datos de investigación...
[Investigador] Hallazgo: la topología jerárquica centraliza el control...
[Supervisor] -> analyst | Ya hay investigación disponible; falta procesarla.
[Analista] Conclusión: el balance es mixto...
[Supervisor] -> FINISH | research_results y analysis_results están completos...
```

### 2. Corrida real con Gemini

```bash
cp .env.example .env   # completar GOOGLE_API_KEY (y TAVILY_API_KEY si querés Tavily real)
export GOOGLE_API_KEY="tu-api-key"
python main.py "¿Qué ventajas y riesgos tiene la topología jerárquica frente a la cooperativa?"
```

Usa `gemini-flash-latest` para los 3 roles (Supervisor + 2 especialistas) y
persiste el estado con `SqliteSaver` (`orchestrator_memory.sqlite`) bajo un
`thread_id`, igual que en la Pre-entrega 5.

## Herramientas por agente

| Agente | Herramienta(s) | Notas |
|---|---|---|
| researcher | `vector_db_search` (o `TavilySearchResults` si hay `TAVILY_API_KEY`) | Búsqueda simulada con TF-IDF + coseno sobre un corpus local |
| analyst | `analyze_sentiment`, `compute_stats`, `validate_schema` | El Supervisor elige cuál de las tres corresponde vía la `instruction` |

## Video mostrando en la terminal el funcionamiento
LinkURL: https://youtu.be/8pLu2hQGHGg

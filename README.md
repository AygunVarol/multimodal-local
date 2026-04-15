# Multimodal Local LLMs in Smart Environments for Visual IoT Analytics

> Local-first framework for interrogating indoor air quality (IAQ) data, spotting anomalies, and chatting with a multimodal LLM across text, images, plots, and knowledge graph facts.

## Overview

This repository packages everything needed to run a privacy-preserving ambient intelligence agent on a single laptop. The system ingests hourly IAQ measurements from multiple rooms, enriches them with a semantic knowledge graph, and exposes a multimodal conversational interface backed by locally hosted Ollama models. Users can:

- Ask natural-language questions that compile to DuckDB SQL and run against merged sensor datasets.
- Generate time-series plots with comfort-band overlays and immediately ask the LLM to interpret the resulting chart.
- Combine text questions with image uploads (for example, screenshots of dashboard widgets) for richer context.
- Query a knowledge graph of locations, displays, comfort policies, and datasets, or allow the LLM to translate NL -> SPARQL.
- Drive simple pervasive display summaries that flag comfort threshold violations.

Everything stays on the edge device: data, models, and analytics remain local to support privacy-sensitive smart environments.

## System Highlights

- **Local multimodal reasoning** - Default Ollama models include `ministral-3:3b`, `qwen3-vl:4b`, `gemma3:4b-it-qat`, and `granite3.2-vision:latest` (set via `OLLAMA_MODELS`), with `OLLAMA_MODEL` choosing the default for requests.
- **IAQ trend analysis** - Hourly measurements are merged into an in-memory DuckDB table so the LLM can surface trends, aggregates, or outliers.
- **Visual analytics loop** - Natural language drives automated plot generation; plots are fed back into the LLM for narrative explanations.
- **Semantic context** - RDF/Turtle knowledge graph (`data/kg.ttl`) stores comfort ranges, display placements, and policies that inform responses and UI badges.
- **Inline comfort badges** - Data answers automatically surface pervasive-display statuses (for temperature, humidity, pressure, and gas resistance) using KG thresholds.
- **Live latency telemetry** - Every UI panel reports its last round-trip time (chat, SQL answers, plots, chart Q&A, and KG load) so you can spot bottlenecks while iterating.
- **Display-ready insights** - `/api/display/cards` produces status cards and ticker strings tuned for ambient or kiosk displays.
- **Zero cloud dependencies** - No external APIs are required once Ollama is running locally.

## Architecture at a Glance

```
+-----------+     text / images     +----------------------+
|  Web UI   | --------------------> | FastAPI Backend      |
| (Tailwind |                       |  /api routes         |
|   + D3)   | <----- JSON/PNG ----- |                      |
+-----+-----+                       |  chat      data/ask  |
      |                             |  data/plot  kg/ask   |
      |                             |  display/cards       |
      |                             +----------+-----------+
      |                                        |
      |                             DuckDB (trend analytics)
      |                                        |
      |             +-------------+            |
      +-----------> | Ollama LLMs | <----------+
                    +-------------+
                           ^
                           | knowledge graph facts
                      RDFLib KG store
```

## Repository Layout

```text
.
|-- backend/                   # FastAPI service that orchestrates data, KG, and LLM prompts
|   |-- app/
|   |   |-- main.py            # API bootstrap + CORS
|   |   |-- models/            # Pydantic request/response schemas
|   |   |-- routes/            # Chat, data, KG, and display endpoints
|   |   |-- services/          # DuckDB wrapper, KG store, Ollama HTTP client
|   |   `-- utils/             # Shared helpers (image base64 conversion)
|   |-- Dockerfile             # Containerized backend
|   `-- requirements.txt
|-- data/                      # Sample IAQ CSVs + turtle knowledge graph
|-- experiments/               # IAQ paper experiment runner, results, and grading UI
|-- experiment/                # Safety-experiment image runner and image sets
|-- scripts/dev.sh             # Helper script to spin up the backend locally
|-- web/                       # Static HTML + JS front-end (TailwindCSS + D3)
|-- docker-compose.yml         # Compose file for backend + mounted data
|-- LICENSE
`-- README.md                  # You are here
```

## Data & Semantic Layer

### Merged IAQ Dataset

- Files matching `*_measurements_IAQ_1h_avg.csv` are auto-discovered under `data/`.
- Each CSV is normalized, coerced to include a lowercase `location` column (inferred from filename when missing), and concatenated into a single DuckDB-registered table named `iaq`. See `backend/app/services/duckdb_engine.py`.
- Core columns: `Date`, `location`, `temperature_c`, `pressure_hpa`, `humidity_rh`, `gas_resistance_ohms`, `heat_stable`, `IAQ_proxy`, `IAQ_class`, `samples_per_hour`.
- The LLM prompt includes schema and sample rows so generated SQL remains grounded in the actual data layout.

### Knowledge Graph (`data/kg.ttl`)

- Declares `Location`, `Metric`, `PervasiveDisplay`, `Policy`, and `Dataset` classes along with object properties linking them.
- Stores recommended comfort ranges (for example, temperature 20-26 deg C, humidity 30-60 percent), dataset provenance, and ambient display placements.
- Links each location to the core IAQ metrics (temperature, humidity, pressure, gas resistance, IAQ score) to align the UI, kiosk summaries, and KG viewer overlays.
- `backend/app/services/kg_store.py` wraps RDFLib queries, offering helpers such as `recommended_range()` and `short_context_for()` to inject relevant triples into LLM prompts.

### Ambient Display Facts

- `/api/display/cards` aggregates IAQ readings per day and location, measures them against KG-defined comfort ranges, and returns card payloads for kiosks (see `backend/app/routes/display.py`).
- The response includes a ticker string (for example, `temperature_c: 25.4  humidity_rh: 41.2`) that can be scrolled across a display.

## Backend Services

- **`backend/app/main.py`** - Configures the FastAPI app, global CORS, and mounts API routers under `/api`.

- **Chat endpoint (`backend/app/routes/chat.py`)**
  - Accepts text plus optional image uploads via multipart form-data.
  - Prepends KG-derived snippets related to the user query before delegating to `ollama_client.chat`.
  - Returns assistant text and the raw Ollama payload to the browser. Errors surface actionable hints (for example, switching to a vision-enabled model).

- **Data exploration (`backend/app/routes/dataqa.py`)**
  - `POST /api/data/ask`: NL -> DuckDB SQL. Only `SELECT` (including `WITH` CTE) statements are allowed; safety checks prevent schema-altering commands before execution via `DuckDBRunner.safe_execute`. Accepts an optional `model` override (defaults to `OLLAMA_MODEL`). The response now includes a `pervasive_status` list that mirrors the kiosk display logic so the UI can surface comfort warnings inline.
  - `POST /api/data/plot`: Parses metric/date/location hints, filters the combined dataframe, overlays comfort bands from the KG, and streams the resulting PNG (base64 encoded) alongside the generated SQL.
  - `POST /api/data/value`: Computes daily means for a metric/date pair, which is handy for quick anomaly detection or threshold checks.

- **Knowledge graph access (`backend/app/routes/kg.py`)**
  - Direct SPARQL endpoint for power users.
  - NL -> SPARQL translator using the local LLM with a fallback generic query if generation fails.

- **Display support (`backend/app/routes/display.py`)**
  - Summarizes temperature and humidity means for kiosk dashboards and labels their status (`ok`, `warn-low`, `warn-high`) using KG thresholds.

- **Services**
  - `ollama_client.py` - Thin HTTP client for Ollama `/api/chat` and `/api/generate`.
  - `duckdb_engine.py` - Handles CSV ingestion, schema introspection, and safe SQL execution.
  - `kg_store.py` - RDFLib wrapper plus prompt helpers to keep the LLM grounded in building-specific context.

## Front-end Experience (`web/`)

- `index.html` arranges four widgets (chat, data Q&A, plot-and-ask, KG viewer) using TailwindCSS for rapid layout and D3.js for interactive knowledge graph visuals.
- `script.js` wires forms to the API, renders data tables, injects SQL snippets, supports the "ask about this chart" loop by attaching generated plots as vision inputs to the LLM, surfaces pervasive display comfort badges beneath query results, and logs per-panel latency metrics.
- Works as a static file, so open it directly in your browser once the backend is running (CORS is already enabled).

## Running Locally

### Prerequisites

- [Ollama](https://ollama.com/) running locally (tested at `http://localhost:11434`).
- Python 3.11 or later (for the backend) and a modern browser.
- Optional: Docker if you prefer containerized deployment.

### 1. Pull or select your model

```bash
ollama pull ministral-3:3b
ollama pull gemma3:4b-it-qat
ollama pull qwen3-vl:4b
ollama pull llava:latest
```

### 2. Start the backend (virtual environment)

```bash
cd backend
python -m venv .venv
. .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

The helper script `scripts/dev.sh` automates these steps on Unix-like shells.

### 3. Launch the front-end

- Open `web/index.html` directly in the browser, or serve the `web/` folder with any static file server.

You can check connectivity via `http://localhost:8000/health`.

### Docker Compose

```bash
ollama serve &
docker compose up --build
```

`docker-compose.yml` mounts `backend/app` for live code reloads and binds `./data` into the container at `/data`.

## Experiments & Evaluation

The experiment runner assumes the backend is already running at `http://localhost:8000` and does not start or stop it for you. Results are written to `experiments/results/` as timestamped JSONL files, with chart PNGs and chart-interpretation text responses stored alongside them for review.

Install the extra experiment dependencies in the same backend virtual environment:

```bash
cd backend
python -m pip install -r requirements.txt
```

Run the paper experiments from the repository root. If `--models` is omitted in an interactive terminal, the runner asks you to choose from the numbered model menu:

```bash
python experiments/experiment.py --task nl_sql
python experiments/experiment.py --task display_cards
python experiments/experiment.py --task plot_gen
python experiments/experiment.py --task chart_interp
python experiments/experiment.py --task kg_ablation
```

The KG ablation task runs only chart interpretation records for temperature and humidity safe-limit decisions, comparing `kg_on` and `kg_off`.

Useful flags:

```bash
python experiments/experiment.py --task nl_sql --models 1,3
python experiments/experiment.py --task nl_sql --models gemma3:4b-it-qat,ministral-3:3b
python experiments/experiment.py --task chart_interp --backend-url http://localhost:8000 --output-dir experiments/results
```

Run the safety-experiment image evaluation from the repository root:

```bash
python experiment/run_safety_experiment.py
```

It uses the same numbered model menu when `--models` is omitted. Results are written under `results/<timestamp>/<model>/` using `raw_safety_experiment.csv`, `summary_safety_experiment.csv`, and `table_safety_experiment.tex`. The older `experiment/run_safety_image_eval.py` command remains as a compatibility wrapper.

Start the evaluation backend and open the grading UI:

```bash
python -m uvicorn experiments.eval_server:app --port 8001
```

Then visit `http://localhost:8001` in the browser. The UI lists every JSONL file under `experiments/results/`, lets you grade pending records in-place, and shows a live accuracy dashboard that refreshes from `/api/summary` every 5 seconds.

To export an aggregate paper table, query the summary endpoint directly:

```bash
curl http://localhost:8001/api/summary
```

The JSON payload groups accuracy, mean latency, and graded counts by model and task, which is suitable for downstream table generation.

## Configuration

Environment variables accepted by the backend:

| Variable          | Default                             | Purpose |
|-------------------|-------------------------------------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434`            | Endpoint for the local Ollama daemon. |
| `OLLAMA_MODELS`   | `ministral-3:3b,qwen3-vl:4b,gemma3:4b-it-qat,llava:latest` | Comma-separated list of available Ollama models. |
| `OLLAMA_MODEL`    | `gemma3:4b-it-qat`                  | Default model sent to Ollama for chat and generation. |
| `DATA_DIR`        | `../data` (or `/data` in Docker)    | Directory containing IAQ CSVs. |
| `CSV_PATH`        | (unset)                             | Force loading a single CSV instead of merging. |
| `KG_PATH`         | `../data/kg.ttl`                    | Optional override for the knowledge graph file. |

Export variables before starting Uvicorn or pass them via Docker environment configuration to tailor the setup.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | `POST` (multipart) | Send `message` text plus optional `images[]` and `model` to the LLM. Returns assistant content and raw Ollama response. |
| `/api/data/ask` | `POST` (JSON) | `{"question": "...", "model": "..."}` -> generated SQL plus query results from DuckDB (uses selected `model` if provided). |
| `/api/data/plot` | `POST` (JSON) | Returns base64 PNG plots with the SQL used and parsed filters (metric/date/location). |
| `/api/data/value` | `POST` (JSON) | Computes mean value for the requested metric and date range. |
| `/api/kg/sparql` | `POST` (JSON) | Executes raw SPARQL against the Turtle graph. |
| `/api/kg/ask` | `POST` (JSON) | NL question translated to SPARQL via LLM; accepts optional `model`. |
| `/api/display/cards` | `GET` | `?date=YYYY-MM-DD&location=office` -> kiosk cards and ticker string. |
| `/health` | `GET` | Lightweight readiness probe exposing the active model list. |

Example call:

```bash
curl -X POST http://localhost:8000/api/data/ask \
     -H "Content-Type: application/json" \
     -d '{"question":"Average humidity by hour for kitchen on 2025-08-19"}'
```

## Working with the Web UI

- **Chat** - Free-form conversation; drag in images of dashboards or sensor placements for contextual reasoning.
- **Ask the IAQ Data** - Generates SQL transparently and displays both the query and result table for auditability. Uses the selected model from the UI (or `OLLAMA_MODEL` by default).
- **Plot & Ask** - Natural language plot requests automatically highlight recommended comfort ranges. Immediately follow up with qualitative questions about the trend.
- **Knowledge Graph Viewer** - Interactive D3 force graph showing relationships between locations, displays, policies, and metrics sourced from the Turtle file.

## Extending & Customizing

1. **Add new sensors** - Drop additional IAQ CSVs into `data/`; restart the backend to refresh the merged DuckDB table. If filenames encode new locations, the loader auto-creates matching `location` values.
2. **Enrich the KG** - Update `data/kg.ttl` with new policies, thresholds, or displays. Front-end badges and backend prompts adapt immediately.
3. **Swap LLMs** - Update `OLLAMA_MODELS` to advertise multiple models and set `OLLAMA_MODEL` to choose the default. Pass an optional `model` field to `/api/chat`, `/api/data/ask`, or `/api/kg/ask` to override per request.
4. **Build anomaly detectors** - The current plot overlay plus `/api/data/value` mean calculations provide the foundation; extend `dataqa.py` with custom thresholds or integrate additional analytics libraries as needed.

## Continuous Integration

A minimal GitHub Actions workflow (`.github/workflows/ci.yml`) is provided to ensure Python tooling (currently Ruff) is available. Extend it with formatting, unit tests, or container builds as the project evolves.

## License

This project is released under the terms of the [MIT License](LICENSE).

---

Questions or ideas? Open an issue, or experiment locally by asking the agent about your latest IAQ trends and policy compliance.

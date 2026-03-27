# LangGraph AI Agent Project

A starter project for building AI agents using LangGraph and LangChain.

## Setup

### Prerequisites

- Python 3.11+
- uv or pip

### Installation

```bash
# Clone the repository
cd langgraph-app

# Install dependencies (with dev tools)
pip install -e ".[dev]"

# Set up environment
cp .env.example .env
# Edit .env (API keys are optional if you use local free mode)
# Optional secure override for local secrets:
# cp .env .env.local
```

`load_dotenv` now loads `.env` and then `.env.local` (override), so secrets can
stay in `.env.local` while `.env` remains shareable/safe.

### Database Backends (SQLite or MongoDB)

By default, the app uses SQLite (`AGENT_DB_BACKEND=sqlite`).

To run with MongoDB:

```bash
# from langgraph-app/
docker compose up -d mongodb
```

Set in `.env`:

```env
AGENT_DB_BACKEND=mongodb
MONGODB_URI=mongodb://127.0.0.1:27017
MONGODB_DATABASE=langgraph_agent
MONGODB_FALLBACK_TO_SQLITE=true
```

Health endpoint now reports active backend:

- `GET /health` -> `db_backend: "sqlite"` or `db_backend: "mongodb"`

If MongoDB is configured but unavailable, backend can fallback to SQLite when
`MONGODB_FALLBACK_TO_SQLITE=true`.

### Free Local Mode (No Account)

This project now supports fully local OSS inference with no account by default.

1. Install Ollama: <https://ollama.com/download>
2. Pull free Asian open-source models:

```bash
ollama pull qwen2.5:latest
ollama pull deepseek-r1:8b
```

1. Keep these values in `.env`:

```env
FREE_LLM_PROVIDER=ollama
FORCE_FREE_LLM=true
FREE_THOUGHT_MODEL=qwen2.5:latest
FREE_REVIEW_MODEL=qwen2.5:latest
FREE_ACTION_MODEL=qwen2.5:latest
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_AUTO_START=true
OLLAMA_AUTO_START_TIMEOUT_SECONDS=25
```

With `OLLAMA_AUTO_START=true`, the app will attempt to start Ollama automatically
before local model calls.

Optional: instead of Ollama, you can use a local OpenAI-compatible server
(LM Studio, vLLM, llama.cpp server) with:

```env
FREE_LLM_PROVIDER=openai_compatible
LOCAL_LLM_BASE_URL=http://localhost:1234/v1
LOCAL_LLM_API_KEY=
```

Optional: you can also use Hugging Face Inference:

```env
FREE_LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=hf_xxx
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
HUGGINGFACE_BASE_URL=https://router.huggingface.co/v1
```

### Live Thermal Regulation

The app includes live thermal regulation to keep runtime behavior more stable
under high temperature or system load.

- `src/nodes.py` applies automatic cooldown before model calls.
- `control_center.py` shows a live thermal panel (temperature, load, level).
- `src/agent.py` prints current thermal status at startup.

Configure these values in `.env`:

```env
THERMAL_MONITOR_ENABLED=true
THERMAL_POLL_INTERVAL_SECONDS=2.0
THERMAL_WARNING_C=75
THERMAL_CRITICAL_C=85
THERMAL_COOLDOWN_WARN_SECONDS=0.8
THERMAL_COOLDOWN_CRITICAL_SECONDS=2.5
THERMAL_TIMEOUT_SCALE_WARNING=1.0
THERMAL_TIMEOUT_SCALE_CRITICAL=0.85
THERMAL_TIMEOUT_MIN_SECONDS=12
```

Notes:

- On macOS, installing `osx-cpu-temp` (Homebrew) enables direct CPU temperature reads.
- If no sensor is available, the regulator falls back to system load ratio.
- Under `critical`, timeout scaling now favors faster failover and cooldown instead of long blocking calls.

### Ollama Stability Tuning

If local inference feels heavy or unstable, tune these:

```env
OLLAMA_CHAT_TIMEOUT_SECONDS=25
OLLAMA_CHAT_RETRIES=0
OLLAMA_BACKOFF_SECONDS=10
OLLAMA_NUM_PREDICT=320
OLLAMA_TAGS_TIMEOUT_SECONDS=4
```

### Running the Application

#### Development Mode

```bash
# Start the LangGraph dev server (watches for changes)
langgraph dev
```

The server will be available at `http://localhost:8123`

#### Production Mode

```bash
# Install without dev dependencies
pip install -e .

# Run your application
python src/agent.py
```

### API Mode (FastAPI)

```bash
# Start HTTP API for UI integrations
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

Optional `.env` for browser frontends:

```env
CORS_ALLOW_ORIGINS=http://localhost:5173
```

Security note:

- If `CORS_ALLOW_ORIGINS=*`, API disables credentialed CORS automatically.

### Command Runner Security

The command endpoint supports hardened controls:

```env
COMMAND_EXECUTION_ENABLED=true
COMMAND_RUN_USE_SHELL=false
# COMMAND_RUN_ALLOWLIST_REGEX=^(npm|pnpm|yarn|python|pytest|uv|git|ls|cat|echo)\b
```

- `COMMAND_RUN_USE_SHELL=false` blocks shell operators (`|`, `&&`, `;`, redirects, etc.).
- High-risk patterns (`rm -rf /`, `shutdown`, `reboot`, `mkfs`, etc.) are blocked.

Main endpoints:

- `GET /health`
- `GET /projects`, `POST /projects`
- `GET /profiles`, `POST /profiles`, `PATCH /profiles/{agent_key}`
- `POST /agent/run`
- `POST /agents/team/run`
- `POST /assistant/chat`
- `POST /commands/run`
- `GET /orchestrator/workflows`
- `POST /orchestrator/run`
- `GET /memories`, `GET /conversations`
- `GET /models/available`

Interactive docs: `http://localhost:8000/docs`

Free OSS references for extending multi-agent workflows:

- `free-agent-code.md`

### Thunder Client Integration (VS Code)

You can test the API directly from Thunder Client using the prepared files in:

- `thunder-client/langgraph-app.postman_collection.json`
- `thunder-client/langgraph-local.postman_environment.json`

Steps:

1. Install VS Code extension: `rangav.vscode-thunder-client`
2. Open Thunder Client in VS Code
3. Import the collection file and the environment file above
4. Select environment `LangGraph Local (8010)`
5. Run `Health` first, then continue with `Projects`, `Profiles`, and `Assistant Chat`

If your backend runs on another port, update `base_url` in the imported environment.

### Visual UI Mode (React + Vite)

```bash
# from langgraph-app/
cd ui
cp .env.example .env
npm install
npm run dev
```

UI runs at `http://localhost:5173` and talks to API via `VITE_API_BASE_URL`.
Default value is `http://localhost:8000`.
The UI includes an Assistant Copilot chat. You can send natural language tasks
or use `comando: <instruction>` to run a project command through the assistant.
It also includes a LangChain-style Orchestrator panel to run predefined workflows
(`diagnostic`, `assistant_quick`, `agent_full`) from the page.

Run backend + UI together:

1. Terminal A: `uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload`
2. Terminal B: `cd ui && npm run dev`

### Always-On Safe Mode (Auto-Restart)

To keep the page and backend active with automatic recovery when either process crashes:

```bash
# from langgraph-app/
./run_keepalive.sh
```

Defaults:

- API: `http://127.0.0.1:8010`
- UI: `http://127.0.0.1:5173`
- Logs: `logs/keepalive/api.log` and `logs/keepalive/ui.log`

Security defaults in keepalive mode:

- Binds services to localhost only (`127.0.0.1`)
- Restricts CORS to local UI origins
- Sets `COMMAND_EXECUTION_ENABLED=false` unless you override it

Useful options:

```bash
# API only
./run_keepalive.sh --no-ui

# Custom ports
./run_keepalive.sh --api-port 8011 --ui-port 5175
```

## Project Structure

```
langgraph-app/
├── src/
│   ├── __init__.py
│   ├── agent.py          # Main agent definition
│   ├── api_server.py     # FastAPI backend for UI integration
│   ├── graph.py          # LangGraph state and graph setup
│   ├── nodes.py          # Node implementations
│   └── tools.py          # Tool definitions
├── ui/                   # React visual frontend
│   ├── src/App.tsx       # Main control surface
│   ├── src/api.ts        # API client
│   └── package.json
├── tests/
│   └── test_agent.py
├── pyproject.toml        # Project configuration
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

## API Keys

API keys are optional. If you use free local mode, you can leave them empty.
For cloud providers, set them in your `.env` file:

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **LangSmith** (optional): `LANGSMITH_API_KEY`

## Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangSmith](https://smith.langchain.com/)

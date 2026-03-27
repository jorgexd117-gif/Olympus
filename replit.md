# LangGraph AI Agent Project

## Overview

A LangGraph + LangChain multi-agent AI application with a FastAPI backend and React/Vite frontend. Features an ethics framework, context acquisition system, and specialized sub-agents.

## Architecture

- **Backend**: FastAPI server (`src/api_server.py`) running on port 8000 (localhost)
- **Frontend**: React + Vite UI (`ui/`) running on port 5000 (0.0.0.0)
- **Database**: PostgreSQL (via `DATABASE_URL` env var, async via asyncpg), plus legacy SQLite (`data/agent_memory.db`)
- **Migrations**: Alembic with async PostgreSQL support (`migrations/`)

## Key Files

- `src/database.py` - Async SQLAlchemy engine, session factory, Base model
- `src/models.py` - SQLAlchemy ORM models (Agent, Folder, AgentAssignment)
- `src/api_server.py` - FastAPI HTTP API facade
- `src/agent.py` - Main LangGraph agent
- `src/graph.py` - LangGraph state and graph setup
- `src/nodes.py` - Graph node implementations
- `src/tools.py` - Tool definitions (calculator, get_current_info, query_db)
- `src/multi_agent.py` - Multi-agent coordinator with ethics + context integration
- `src/prompt_orchestrator.py` - PromptOrchestrator class (LLM-powered intent translation with Pydantic schemas)
- `src/ethics.py` - Ethics framework (input/output filtering, principles, audit logging)
- `src/context_acquisition.py` - Context acquisition engine (file search, DB schema, memory, env)
- `src/subagents.py` - Specialized sub-agents (Planner, Researcher, Coder, Reviewer, Executor, Synthesizer)
- `src/persistence.py` - Database abstraction (SQLite/MongoDB)
- `src/process_orchestrator.py` - Workflow orchestration
- `ui/src/App.tsx` - Main React UI component
- `ui/src/components/AgentList.tsx` - Agent listing and creation UI
- `ui/src/components/FolderManager.tsx` - Folder tree management
- `ui/src/components/AgentAssigner.tsx` - Assign agents to folder processes
- `ui/src/components/SubAgentPanel.tsx` - Sub-agent pipeline execution, ethics viewer, DB query UI
- `ui/src/api.ts` - API client functions
- `ui/src/types.ts` - TypeScript type definitions

## Sub-Agent System

Six specialized sub-agents that collaborate in pipelines:
- **Planificador** (Planner) - Task decomposition and prioritization
- **Investigador** (Researcher) - Information gathering and gap analysis
- **Programador** (Coder) - Code generation and modification
- **Revisor** (Reviewer) - Quality, security, and correctness review
- **Ejecutor** (Executor) - Tool execution and action planning
- **Sintetizador** (Synthesizer) - Result integration and final output

Pipeline templates: full_analysis, code_task, research, quick_answer, execute

## Ethics Framework

7 principles: beneficence, non-maleficence, autonomy, transparency, fairness, privacy, accountability. Input/output filtering with pattern-based detection for harmful content, privacy violations, deception, prompt injection. Sensitive data redaction in outputs. Audit logging.

## Prompt Orchestrator

LLM-powered intent translation layer (`src/prompt_orchestrator.py`). Translates natural language prompts into structured JSON schemas (Pydantic-validated) containing:
- Objective, intent classification, parameters
- Tool activation (calculator, get_current_info, query_db)
- Priority (low/medium/high/critical), expected output format
- Sub-task decomposition, ambiguity scoring with clarification questions
- Uses free Ollama/DeepSeek LLM for translation, with deterministic rule-based fallback
- Integrated as first step in `assistant_turn` pipeline (replaces static `_translate_prompt_to_machine_ir` for general queries)

## Context Acquisition

Automatic context gathering when agents don't know something:
- Local file search (docs, configs)
- Database schema discovery
- Memory context analysis
- Environment configuration
- Uncertainty detection and human question generation

## Workflows

- **Start application**: `cd ui && npm run dev` -> port 5000 (webview)
- **Backend API**: `uvicorn src.api_server:app --host localhost --port 8000 --reload` -> console

## API Endpoints

- `GET /healthz` - Database health check
- `GET /health` - Health check
- `GET /projects`, `POST /projects` - Project management
- `GET /profiles`, `POST /profiles`, `PATCH /profiles/{key}` - Agent profile management
- `GET /api/agents`, `POST /api/agents`, `PATCH /api/agents/{id}` - Agent CRUD
- `GET /api/folders`, `POST /api/folders`, `DELETE /api/folders/{id}` - Folder management
- `GET /api/folders/{id}/assignments`, `POST /api/folders/{id}/assignments` - Agent assignments
- `DELETE /api/assignments/{id}` - Remove assignment
- `POST /api/subagents/run` - Run sub-agent pipeline
- `GET /api/subagents/configs` - Get sub-agent configurations
- `GET /api/subagents/pipelines` - Get pipeline templates
- `GET /api/ethics/principles` - View ethics principles and audit
- `POST /api/ethics/check` - Check text against ethics framework
- `POST /api/context/acquire` - Acquire context for a query
- `POST /api/query-db` - Execute read-only SQL against PostgreSQL
- `POST /api/orchestrator/translate` - Translate prompt to structured JSON via PromptOrchestrator
- `POST /agent/run` - Run single agent
- `POST /agents/team/run` - Run agent team
- `POST /assistant/chat` - Assistant chat
- `POST /commands/run` - Run shell commands
- `GET /orchestrator/workflows`, `POST /orchestrator/run` - Workflow orchestration
- `GET /memories`, `GET /conversations` - History
- `GET /models/available` - Available LLM models

## Dependencies

- Python: langgraph, langchain-core, langchain-openai, fastapi, uvicorn, pymongo, python-dotenv, httpx, sqlalchemy[asyncio], asyncpg, psycopg2-binary, alembic, structlog
- Node: react, react-dom, vite, @vitejs/plugin-react, typescript

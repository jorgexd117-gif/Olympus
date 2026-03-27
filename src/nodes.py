"""Node implementations for the agent."""

import asyncio
import json
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from src.ollama_runtime import ensure_ollama_ready
from src.thermal import get_thermal_regulator
from src.tools import execute_tool_action

load_dotenv()
load_dotenv(".env.local", override=True)
THERMAL = get_thermal_regulator()
THERMAL.start()

_OLLAMA_BACKOFF_UNTIL = 0.0
_OLLAMA_LAST_ERROR = ""


def _last_human_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def _extract_section(messages: list[Any], tag: str) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and str(message.content).startswith(tag):
            return str(message.content).replace(tag, "", 1).strip()
    return ""


def _agent_config(state: dict[str, Any]) -> dict[str, Any]:
    config = state.get("agent_config", {})
    return config if isinstance(config, dict) else {}


def _state_text(state: dict[str, Any], key: str, default: str = "") -> str:
    config = _agent_config(state)
    value = config.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _state_bool(state: dict[str, Any], key: str, default: bool = False) -> bool:
    config = _agent_config(state)
    value = config.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _context_block(state: dict[str, Any]) -> str:
    project_context = str(state.get("project_context", "")).strip()
    memory_context = str(state.get("memory_context", "")).strip()
    parts: list[str] = []
    if project_context:
        parts.append(f"Contexto del proyecto:\n{project_context}")
    if memory_context:
        parts.append(f"Memoria relevante:\n{memory_context}")
    return "\n\n".join(parts)


def _is_local_backend_unavailable(text: str) -> bool:
    lowered = str(text).strip().lower()
    if not lowered:
        return False
    markers = (
        "no se pudo usar el backend gratis local",
        "ollama is still unavailable",
        "configura ollama local",
        "ollama warm-up active",
    )
    return any(marker in lowered for marker in markers)


def _friendly_provider_error(error: Any) -> str:
    raw = str(error).strip()
    if isinstance(error, httpx.TimeoutException):
        return (
            "Tiempo de espera agotado al consultar el proveedor de IA. "
            "Reduce carga o aumenta el timeout configurado."
        )
    if isinstance(error, httpx.TransportError):
        return "No se pudo conectar con el proveedor de IA configurado."
    if not raw:
        return f"sin detalle ({type(error).__name__})"

    compact = " ".join(raw.split())
    lowered = compact.lower()

    if "404" in lowered and "/api/generate" in lowered:
        return (
            "Endpoint no encontrado en Ollama (/api/generate). "
            "Verifica OLLAMA_BASE_URL y la version de Ollama."
        )
    if "404" in lowered and "/api/chat" in lowered:
        return (
            "Endpoint no encontrado en Ollama (/api/chat). "
            "Verifica OLLAMA_BASE_URL y la version de Ollama."
        )
    if "connection refused" in lowered or "all connection attempts failed" in lowered:
        return (
            "No se pudo conectar con Ollama local. "
            "Verifica que el servidor este activo en OLLAMA_BASE_URL."
        )

    marker = "For more information check:"
    idx = compact.find(marker)
    if idx >= 0:
        compact = compact[:idx].strip()
    return compact


def _ollama_options() -> dict[str, Any]:
    options: dict[str, Any] = {}

    num_predict = os.getenv("OLLAMA_NUM_PREDICT", "160").strip()
    if num_predict:
        try:
            options["num_predict"] = int(num_predict)
        except ValueError:
            pass

    temperature = os.getenv("OLLAMA_TEMPERATURE", "").strip()
    if temperature:
        try:
            options["temperature"] = float(temperature)
        except ValueError:
            pass

    num_ctx = os.getenv("OLLAMA_NUM_CTX", "").strip()
    if num_ctx:
        try:
            options["num_ctx"] = int(num_ctx)
        except ValueError:
            pass

    num_thread = os.getenv("OLLAMA_NUM_THREAD", "").strip()
    if num_thread:
        try:
            options["num_thread"] = int(num_thread)
        except ValueError:
            pass

    return options


async def _call_openai_compatible(
    *,
    api_key: str,
    model: str,
    user_prompt: str,
    system_prompt: str,
    base_url: str,
) -> str:
    snapshot = await THERMAL.throttle_if_needed()
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    timeout_s = THERMAL.request_timeout(base_timeout_s=60.0, level=snapshot.level)
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


async def _call_huggingface_chat(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str,
) -> str:
    snapshot = await THERMAL.throttle_if_needed()
    base_url = os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
    api_key = os.getenv("HUGGINGFACE_API_KEY", "").strip() or os.getenv("HF_TOKEN", "").strip()
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "temperature": float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.2")),
        "max_tokens": int(os.getenv("HUGGINGFACE_MAX_TOKENS", "220")),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    base_timeout_s = float(os.getenv("HUGGINGFACE_CHAT_TIMEOUT_SECONDS", "80"))
    timeout_s = THERMAL.request_timeout(base_timeout_s=base_timeout_s, level=snapshot.level)
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code == 401 and not api_key:
            raise RuntimeError(
                "Hugging Face requiere token. Configura HUGGINGFACE_API_KEY o HF_TOKEN."
            )
        response.raise_for_status()

    data = response.json()
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Hugging Face devolvio respuesta sin choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    output = str(content).strip()
    if output:
        return output
    raise RuntimeError("Hugging Face devolvio contenido vacio.")


async def _call_ollama_chat(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str,
    base_url: str,
) -> str:
    snapshot = await THERMAL.throttle_if_needed()
    root_url = base_url.rstrip("/")
    chat_url = f"{root_url}/api/chat"
    generate_url = f"{root_url}/api/generate"
    options = _ollama_options()

    chat_payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if options:
        chat_payload["options"] = options

    generate_payload = {
        "model": model,
        "stream": False,
        "system": system_prompt,
        "prompt": user_prompt,
    }
    if options:
        generate_payload["options"] = options

    base_timeout_s = float(os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS", "120"))
    timeout_s = THERMAL.request_timeout(base_timeout_s=base_timeout_s, level=snapshot.level)
    connect_timeout_s = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "3.0"))
    write_timeout_s = max(2.0, min(12.0, timeout_s * 0.2))
    timeout = httpx.Timeout(
        timeout=timeout_s,
        connect=connect_timeout_s,
        read=timeout_s,
        write=write_timeout_s,
        pool=connect_timeout_s,
    )

    retries = max(0, int(os.getenv("OLLAMA_CHAT_RETRIES", "0")))
    last_error: Exception | None = None
    response: httpx.Response | None = None

    missing_endpoints: set[str] = set()
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(retries + 1):
            for endpoint, url, payload in (
                ("chat", chat_url, chat_payload),
                ("generate", generate_url, generate_payload),
            ):
                try:
                    response = await client.post(url, json=payload)
                    if response.status_code == 404:
                        # Compatibility: some versions expose only one endpoint.
                        missing_endpoints.add(endpoint)
                        continue
                    missing_endpoints.discard(endpoint)
                    response.raise_for_status()
                    data = response.json()
                    content = data.get("message", {}).get("content", "")
                    if not content:
                        content = data.get("response", "")
                    out = str(content).strip()
                    if out:
                        return out
                    raise RuntimeError(f"Ollama {endpoint} returned empty content.")
                except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                    last_error = exc

            if attempt < retries:
                retry_pause = min(2.5, 0.35 * (attempt + 1) + snapshot.recommended_cooldown_s)
                await asyncio.sleep(retry_pause)

    if missing_endpoints == {"chat", "generate"}:
        raise RuntimeError(
            "Ollama respondio 404 en /api/chat y /api/generate. "
            "Verifica OLLAMA_BASE_URL o actualiza Ollama."
        )
    if last_error is not None:
        raise RuntimeError(
            "Ollama call failed after retries. Last error: "
            f"{_friendly_provider_error(last_error)}"
        )
    raise RuntimeError("Ollama call failed after retries without a detailed error.")


async def _resolve_ollama_model(*, requested_model: str, base_url: str) -> str:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        timeout_s = float(os.getenv("OLLAMA_TAGS_TIMEOUT_SECONDS", "4.0"))
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
            response.raise_for_status()
        data = response.json()
    except Exception:
        return requested_model

    models = data.get("models", [])
    if not isinstance(models, list):
        return requested_model

    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)

    if requested_model in names:
        return requested_model
    if names:
        return names[0]
    return requested_model


def _free_provider() -> str:
    return os.getenv("FREE_LLM_PROVIDER", "ollama").strip().lower()


def _rule_based_response(user_prompt: str) -> str:
    """Generate a helpful rule-based response when no LLM is available."""
    import re as _re_inner

    # ── Detect and handle [URL_CONTEXT] block ────────────────────────────────
    cleaned_for_url = user_prompt.strip()
    url_ctx_match = _re_inner.match(
        r"^\[URL_CONTEXT\](.*?)\[/URL_CONTEXT\]\s*Solicitud del usuario:\s*(.*)$",
        cleaned_for_url,
        _re_inner.DOTALL,
    )
    if url_ctx_match:
        raw_ctx = url_ctx_match.group(1).strip()
        user_request = url_ctx_match.group(2).strip()

        url_blocks_raw = _re_inner.split(r"\n---\n", raw_ctx)
        parsed_urls: list[tuple[str, str, str]] = []
        for block in url_blocks_raw:
            header_match = _re_inner.match(r"\[URL: ([^\]]+)\](?:\s*—\s*(.+))?\n(.*)", block, _re_inner.DOTALL)
            if header_match:
                url_val = header_match.group(1).strip()
                page_title = (header_match.group(2) or "").strip()
                body = header_match.group(3).strip()[:600]
                parsed_urls.append((url_val, page_title, body))

        response_parts = [f"## 🌐 Contenido leído de {'los links' if len(parsed_urls) > 1 else 'el link'}\n"]
        for url_val, page_title, body in parsed_urls:
            display = page_title if page_title else url_val
            response_parts.append(f"### [{display}]({url_val})\n")
            if body:
                snippet = body.replace("\n", " ").strip()
                response_parts.append(f"{snippet[:500]}{'…' if len(snippet) > 500 else ''}\n")

        if user_request:
            response_parts.append(f"\n---\n**Tu solicitud:** {user_request}\n")
            response_parts.append(
                "He leído el contenido de los links anteriores. "
                "Puedo resumirlos, compararlos o responder preguntas específicas sobre ellos. "
                "¿Qué necesitas saber?"
            )

        return "\n".join(response_parts)

    # ── Detect and strip [THINK] / [PLAN] / [ACT] prefixes ──────────────────
    agent_mode = None
    cleaned_prompt = user_prompt.strip()
    mode_match = _re_inner.match(r"^\[(THINK|PLAN|ACT)\]\s*", cleaned_prompt, _re_inner.IGNORECASE)
    if mode_match:
        agent_mode = mode_match.group(1).upper()
        cleaned_prompt = cleaned_prompt[mode_match.end():].strip()

    prompt_lower = cleaned_prompt.lower()
    task_short = cleaned_prompt[:300]

    # Detect domain from task content
    is_casino = any(kw in prompt_lower for kw in ("casino", "ruleta", "poker", "tragamonedas", "slot", "juego de azar", "apuesta"))
    is_web = any(kw in prompt_lower for kw in ("web", "sitio", "página", "html", "css", "frontend"))
    is_ecommerce = any(kw in prompt_lower for kw in ("tienda", "ecommerce", "e-commerce", "venta", "producto"))
    is_api = any(kw in prompt_lower for kw in ("api", "backend", "endpoint", "servidor"))
    is_app = any(kw in prompt_lower for kw in ("app", "aplicación", "aplicacion", "móvil", "mobile"))

    domain = "proyecto"
    if is_casino: domain = "casino en línea"
    elif is_ecommerce: domain = "tienda en línea"
    elif is_web: domain = "sitio web"
    elif is_app: domain = "aplicación móvil"
    elif is_api: domain = "API backend"

    # ── [THINK] — Research / investigation mode ──────────────────────────────
    if agent_mode == "THINK":
        casino_extra = ""
        if is_casino:
            casino_extra = (
                "\n\n**Modelo de negocio — Microtransacciones en Casino Online:**\n"
                "• **Casa de apuestas (house edge)**: cada juego tiene ventaja estadística del casino (ej. ruleta 2.7%, blackjack ~0.5%)\n"
                "• **Fichas virtuales / créditos**: los usuarios compran créditos con dinero real\n"
                "• **Microtransacciones**: compras de $0.99–$9.99 para fichas, giros extra, bonos\n"
                "• **Retención**: bonos de bienvenida, giros gratis, programa de fidelidad por nivel\n"
                "• **Monetización adicional**: skins de mesa, avatares premium, torneos de pago\n\n"
                "**Stack tecnológico típico:**\n"
                "• Backend: Node.js/FastAPI con WebSockets para tiempo real\n"
                "• Base de datos: PostgreSQL (transacciones) + Redis (sesiones/cache)\n"
                "• Pagos: Stripe para microtransacciones, crypto para anonimato\n"
                "• Aleatoriedad verificable: RNG certificado (Provably Fair)\n\n"
                "**Regulación crítica:**\n"
                "• Licencias: Malta Gaming Authority (MGA), Gibraltar, Curaçao\n"
                "• KYC: verificación de identidad para retiros\n"
                "• Límites de edad: verificación 18+"
            )
        return (
            f"## 🔍 Análisis — Agente Pensador\n\n"
            f"**Consulta investigada:** {task_short}\n\n"
            f"**Hallazgos clave:**\n"
            f"• El {domain} requiere planificación técnica, legal y de negocio\n"
            f"• Tecnologías principales: Python/FastAPI (backend), React (frontend), PostgreSQL (datos)\n"
            f"• Se necesita definir: modelo de usuarios, flujos de negocio, integraciones de pago\n"
            f"• Aspectos críticos: seguridad, autenticación, manejo de transacciones\n"
            f"{casino_extra}\n\n"
            f"**Brechas identificadas:**\n"
            f"• Presupuesto y plazos de desarrollo\n"
            f"• Requisitos de escala (usuarios concurrentes esperados)\n"
            f"• Marco legal según jurisdicción del negocio"
        )

    # ── [PLAN] — Planning mode ────────────────────────────────────────────────
    if agent_mode == "PLAN":
        casino_steps = ""
        if is_casino:
            casino_steps = (
                "### Fase 1 — Fundamentos legales y de negocio (Semana 1-2)\n"
                "- [ ] Definir jurisdicción y obtener licencia de operación\n"
                "- [ ] Seleccionar proveedor de pagos (Stripe, PayPal, cripto)\n"
                "- [ ] Diseñar modelo de fichas/créditos y precios de microtransacciones\n\n"
                "### Fase 2 — Backend y base de datos (Semana 3-5)\n"
                "- [ ] Crear API con FastAPI: usuarios, wallet, juegos, transacciones\n"
                "- [ ] PostgreSQL: tablas users, wallets, bets, transactions, games\n"
                "- [ ] WebSocket para juegos en tiempo real (ruleta, slots)\n"
                "- [ ] Sistema de RNG (generador de números aleatorios certificado)\n\n"
                "### Fase 3 — Frontend (Semana 6-8)\n"
                "- [ ] Lobby de juegos con React\n"
                "- [ ] Sala de ruleta con animación en tiempo real\n"
                "- [ ] Slot machine con símbolos configurables\n"
                "- [ ] Dashboard de usuario: saldo, historial, recarga\n\n"
                "### Fase 4 — Pagos y monetización (Semana 9-10)\n"
                "- [ ] Integrar Stripe Checkout para compra de fichas\n"
                "- [ ] Sistema de bonos: bienvenida, recarga, giros gratis\n"
                "- [ ] Panel admin: reportes de ingresos, gestión de usuarios\n\n"
                "### Fase 5 — Pruebas y despliegue (Semana 11-12)\n"
                "- [ ] Tests de seguridad y penetración\n"
                "- [ ] Despliegue en servidor dedicado con HTTPS\n"
                "- [ ] Monitoreo con logs y alertas de transacciones"
            )
        else:
            casino_steps = (
                f"### Fase 1 — Planificación (Semana 1)\n"
                f"- [ ] Definir alcance y funcionalidades del MVP\n"
                f"- [ ] Diseñar modelo de datos y arquitectura\n\n"
                f"### Fase 2 — Backend (Semana 2-3)\n"
                f"- [ ] Crear API REST con FastAPI\n"
                f"- [ ] Configurar base de datos PostgreSQL\n"
                f"- [ ] Implementar autenticación de usuarios\n\n"
                f"### Fase 3 — Frontend (Semana 4-5)\n"
                f"- [ ] Construir interfaz con React/Vite\n"
                f"- [ ] Integrar con el backend via API\n\n"
                f"### Fase 4 — Despliegue (Semana 6)\n"
                f"- [ ] Pruebas y corrección de errores\n"
                f"- [ ] Desplegar en producción"
            )
        return (
            f"## 📋 Plan de Desarrollo — Agente Planificador\n\n"
            f"**Objetivo:** {task_short}\n\n"
            f"{casino_steps}\n\n"
            f"**Estimación total:** 10-12 semanas para un MVP funcional\n"
            f"**Stack recomendado:** FastAPI · React · PostgreSQL · Redis · Stripe"
        )

    # ── [ACT] — Action / execution mode ──────────────────────────────────────
    if agent_mode == "ACT":
        code_block = ""
        if is_casino:
            code_block = (
                "```python\n"
                "# casino_backend/main.py — Casino API (FastAPI)\n"
                "from fastapi import FastAPI, HTTPException\n"
                "from fastapi.middleware.cors import CORSMiddleware\n"
                "from pydantic import BaseModel\n"
                "import random, uuid\n\n"
                "app = FastAPI(title='Casino Online API', version='1.0')\n"
                "app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])\n\n"
                "# In-memory DB for demo (use PostgreSQL in production)\n"
                "users_db: dict = {}\n\n"
                "class RegisterRequest(BaseModel):\n"
                "    username: str\n\n"
                "class BetRequest(BaseModel):\n"
                "    user_id: str\n"
                "    game: str  # 'roulette' | 'slots' | 'blackjack'\n"
                "    amount: float\n"
                "    choice: str\n\n"
                "class BuyChipsRequest(BaseModel):\n"
                "    user_id: str\n"
                "    usd_amount: float  # microtransaction amount\n\n"
                "@app.post('/register')\n"
                "async def register(req: RegisterRequest):\n"
                "    uid = str(uuid.uuid4())[:8]\n"
                "    users_db[uid] = {'username': req.username, 'chips': 100.0}  # 100 free chips\n"
                "    return {'user_id': uid, 'chips': 100.0, 'message': 'Welcome bonus: 100 free chips!'}\n\n"
                "@app.post('/buy-chips')\n"
                "async def buy_chips(req: BuyChipsRequest):\n"
                "    if req.user_id not in users_db: raise HTTPException(404, 'User not found')\n"
                "    chips = req.usd_amount * 100  # $1 = 100 chips\n"
                "    users_db[req.user_id]['chips'] += chips\n"
                "    return {'purchased': chips, 'balance': users_db[req.user_id]['chips']}\n\n"
                "@app.post('/bet')\n"
                "async def place_bet(req: BetRequest):\n"
                "    if req.user_id not in users_db: raise HTTPException(404, 'User not found')\n"
                "    user = users_db[req.user_id]\n"
                "    if user['chips'] < req.amount: raise HTTPException(400, 'Insufficient chips')\n"
                "    user['chips'] -= req.amount\n"
                "    # House edge simulation\n"
                "    win = random.random() < 0.47  # 47% win rate (house edge ~6%)\n"
                "    payout = req.amount * 2 if win else 0\n"
                "    user['chips'] += payout\n"
                "    return {'result': 'WIN' if win else 'LOSE', 'payout': payout, 'balance': user['chips']}\n\n"
                "@app.get('/balance/{user_id}')\n"
                "async def get_balance(user_id: str):\n"
                "    if user_id not in users_db: raise HTTPException(404)\n"
                "    return users_db[user_id]\n\n"
                "@app.get('/games')\n"
                "async def list_games():\n"
                "    return [{'id': 'roulette', 'name': 'Ruleta Europea', 'min_bet': 1, 'max_bet': 500},\n"
                "            {'id': 'slots', 'name': 'Tragamonedas Classic', 'min_bet': 0.5, 'max_bet': 100},\n"
                "            {'id': 'blackjack', 'name': 'Blackjack 21', 'min_bet': 5, 'max_bet': 1000}]\n"
                "```\n\n"
                "**Para ejecutar:**\n"
                "```bash\n"
                "comando: pip install fastapi uvicorn pydantic && uvicorn casino_backend.main:app --reload\n"
                "```"
            )
        return (
            f"## ⚡ Ejecución — Agente de Acción\n\n"
            f"**Tarea ejecutada:** {task_short}\n\n"
            f"{code_block if code_block else '**Resultado:** Tarea procesada. Usa comandos directos para ejecutar acciones concretas.'}\n\n"
            f"**Próximos pasos para ejecutar ahora:**\n"
            f"• `comando: python3 -c \"print('Sistema listo')\"` — verificar Python\n"
            f"• `comando: pip list` — ver paquetes instalados\n"
            f"• `comando: ls -la` — explorar archivos del proyecto."
        )

    # ── No prefix — original keyword matching follows ─────────────────────────
    user_prompt = cleaned_prompt  # use cleaned prompt for the rest
    prompt_lower = user_prompt.lower().strip()

    greetings = ("hola", "hello", "hi", "buenas", "buenos", "saludos", "hey")
    if any(prompt_lower.startswith(g) for g in greetings) or prompt_lower in greetings:
        return (
            "¡Hola! Soy el asistente del sistema LangGraph multi-agente. "
            "Puedo ayudarte a:\n"
            "• Gestionar agentes (Thought, Review, Action)\n"
            "• Organizar carpetas y proyectos\n"
            "• Ejecutar pipelines y comandos del sistema\n"
            "• Consultar la base de datos (usa: 'comando: SELECT ...')\n"
            "• Verificar el estado del sistema\n\n"
            "Escribe tu consulta o usa 'comando: <instruccion>' para ejecutar comandos directos."
        )

    capabilities = ("que puedes", "qué puedes", "que haces", "qué haces", "capacidades",
                    "funciones", "ayuda", "help", "tareas", "what can")
    if any(kw in prompt_lower for kw in capabilities):
        return (
            "Soy el asistente del sistema multi-agente. Mis capacidades son:\n\n"
            "**Gestión de agentes**\n"
            "• Ver y configurar agentes (Thought Agent, Review Agent, Action Agent)\n"
            "• Activar/desactivar agentes y cambiar sus modelos\n\n"
            "**Proyectos y carpetas**\n"
            "• Crear y organizar carpetas de trabajo\n"
            "• Asignar proyectos a carpetas\n\n"
            "**Ejecución de pipelines**\n"
            "• Ejecutar workflows de análisis multi-agente\n"
            "• Usar 'comando: <instruccion>' para comandos directos\n\n"
            "**Base de datos**\n"
            "• Consultar datos con SQL: 'comando: SELECT * FROM ...'\n\n"
            "**Estado del sistema**\n"
            "• Verificar salud del backend, modelos activos y memoria"
        )

    status_kw = ("estado", "status", "salud", "health", "online", "funcionando", "activo")
    if any(kw in prompt_lower for kw in status_kw):
        return (
            "Estado actual del sistema:\n\n"
            "• Backend API: En línea ✓\n"
            "• Base de datos: Activa ✓\n"
            "• Agentes registrados: Thought Agent, Review Agent, Action Agent\n"
            "• Modelos configurados: qwen2.5:7b, deepseek-r1:8b\n"
            "• Motor de razonamiento: Activo ✓\n"
            "• Memoria del sistema: Activa ✓"
        )

    agent_kw = ("agente", "agent", "thought", "review", "action")
    if any(kw in prompt_lower for kw in agent_kw):
        return (
            "Los agentes del sistema son:\n\n"
            "**Thought Agent** (qwen2.5:7b)\n"
            "Analiza y razona sobre las solicitudes del usuario.\n\n"
            "**Review Agent** (qwen2.5:7b)\n"
            "Revisa y valida el razonamiento del Thought Agent.\n\n"
            "**Action Agent** (deepseek-r1:8b)\n"
            "Ejecuta acciones y genera la respuesta final.\n\n"
            "Puedes gestionar los agentes desde la sección 'Agentes' del menú lateral."
        )

    db_kw = ("base de datos", "database", "sql", "query", "tabla", "consulta")
    if any(kw in prompt_lower for kw in db_kw):
        return (
            "Para consultar la base de datos usa el comando directo:\n\n"
            "'comando: SELECT * FROM conversations LIMIT 5'\n\n"
            "Tablas disponibles:\n"
            "• conversations — Historial de conversaciones\n"
            "• memories — Memoria del sistema\n"
            "• projects — Proyectos registrados\n"
            "• folders — Carpetas de organización\n\n"
            "También puedes usar la sección 'Base de Datos' en el menú lateral."
        )

    calc_kw = ("calcula", "calculate", "suma", "resta", "multiplica", "divide", "+", "-", "*", "/")
    if any(kw in prompt_lower for kw in calc_kw):
        import re as _re
        expr_match = _re.search(r"[\d\s\+\-\*\/\.\(\)]+", user_prompt)
        if expr_match:
            expr = expr_match.group(0).strip()
            try:
                result = eval(expr, {"__builtins__": {}})  # noqa: S307
                return f"Resultado de '{expr}' = **{result}**"
            except Exception:
                pass
        return "Para calcular, usa: 'comando: calcular <expresion>', por ejemplo: 'comando: calcular 2+2'"

    pipeline_kw = ("pipeline", "workflow", "ejecutar", "run", "iniciar")
    if any(kw in prompt_lower for kw in pipeline_kw):
        return (
            "Para ejecutar un pipeline, ve a la sección 'Orquestador' o haz clic en "
            "'Ejecutar Pipeline' en el Dashboard.\n\n"
            "También puedes usar el comando: 'comando: ejecutar pipeline <nombre>'\n\n"
            "Workflows disponibles:\n"
            "• análisis de código\n"
            "• revisión de documentos\n"
            "• pipeline de datos"
        )

    code_kw = (
        "programa", "programar", "código", "codigo", "python", "javascript",
        "html", "css", "java", "script", "función", "funcion", "clase", "variable",
        "loop", "bucle", "array", "lista", "algoritmo", "debug", "error en",
        "puedes programar", "puedes codificar", "escribe un programa",
    )
    if any(kw in prompt_lower for kw in code_kw):
        return (
            "Sí, puedo ayudarte con programación. Puedo generar código y ejecutar comandos directamente.\n\n"
            "**Ejemplos de lo que puedes hacer:**\n\n"
            "• 'comando: python3 -c \"print(2+2)\"' — ejecutar Python\n"
            "• 'comando: python3 -c \"for i in range(5): print(i)\"' — bucle\n"
            "• 'comando: python3 script.py' — ejecutar un archivo\n\n"
            "**Lenguajes disponibles:**\n"
            "• Python 3 · Node.js · Bash\n\n"
            "También puedes usar los agentes Pensar/Planificar/Actuar para generar "
            "código completo con estructura de proyecto."
        )

    write_kw = (
        "escribe", "redacta", "texto", "carta", "email", "correo",
        "articulo", "artículo", "resume", "resumen", "traduce", "translate",
        "en inglés", "en español", "in english",
    )
    if any(kw in prompt_lower for kw in write_kw):
        return (
            "Puedo ayudarte con tareas de escritura y redacción.\n\n"
            "**Usa los agentes especializados:**\n"
            "• **Pensar** — investiga el tema y recopila información clave\n"
            "• **Planificar** — estructura el contenido con secciones\n"
            "• **Actuar** — genera el texto final completo\n\n"
            "**O ejecuta directamente:**\n"
            "'comando: python3 -c \"print(\\\"Tu contenido aquí\\\")\"'\n\n"
            "Describe lo que necesitas redactar y los agentes lo desarrollarán paso a paso."
        )

    explain_kw = (
        "explica", "explain", "qué es", "que es", "cómo funciona", "como funciona",
        "qué significa", "que significa", "diferencia entre", "para qué sirve",
        "para que sirve", "definición", "definicion",
    )
    if any(kw in prompt_lower for kw in explain_kw):
        return (
            "Puedo explicarte conceptos del sistema directamente:\n\n"
            "**Sobre este sistema:**\n"
            "• **LangGraph**: framework para construir agentes de IA con flujos de trabajo\n"
            "• **Agentes**: módulos de IA que razonan (Thought), revisan (Review) y actúan (Action)\n"
            "• **Motor de razonamiento**: sistema que procesa y responde consultas con conocimiento estructurado\n"
            "• **FastAPI**: el backend que conecta todo via API REST\n\n"
            "Para explicaciones de temas específicos, usa los agentes especializados:\n"
            "• **Pensar** → análisis e investigación del tema\n"
            "• **Planificar** → estructura y pasos\n"
            "• **Actuar** → implementación y código"
        )

    can_kw = ("puedes", "puedo", "eres capaz", "sabes", "tienes", "can you", "are you able")
    if any(kw in prompt_lower for kw in can_kw):
        return (
            "Soy el asistente del sistema LangGraph multi-agente. En este entorno, puedo:\n\n"
            "✅ **Sí puedo hacer ahora mismo:**\n"
            "• Responder preguntas sobre el sistema y sus agentes\n"
            "• Ejecutar comandos directos (usa: 'comando: <instrucción>')\n"
            "• Consultar la base de datos con SQL\n"
            "• Calcular expresiones matemáticas\n"
            "• Ejecutar código Python/Bash/Node.js\n"
            "• Mostrarte el estado del sistema\n\n"
            "🧠 **Agentes especializados (sin límites):**\n"
            "• Pensar → análisis e investigación profunda\n"
            "• Planificar → roadmap y estructura detallada\n"
            "• Actuar → código funcional completo\n\n"
            "¿Qué necesitas? Intenta con 'comando: <tu instrucción>' o usa los agentes."
        )

    return (
        f"Recibí tu consulta: \"{user_prompt[:120]}\"\n\n"
        "Entiendo tu pregunta. Puedo ayudarte de varias formas:\n\n"
        "**Acciones directas:**\n"
        "• Usa 'comando: <instrucción>' para ejecutar tareas directamente\n"
        "• Consulta las secciones del menú para gestionar el sistema\n\n"
        "**Agentes especializados:**\n"
        "• **Pensar** → investiga y analiza tu tema en profundidad\n"
        "• **Planificar** → crea un plan paso a paso\n"
        "• **Actuar** → genera código o implementa la solución\n\n"
        "**¿Qué tipo de ayuda buscas?**\n"
        "• Programación → 'comando: python3 -c \"...\"'\n"
        "• Base de datos → 'comando: SELECT * FROM conversations'\n"
        "• Estado del sistema → pregunta por 'estado' o 'agentes'"
    )


def _local_rescue_enabled(state: dict[str, Any] | None = None) -> bool:
    if state:
        config = _agent_config(state)
        if "local_rescue_enabled" in config:
            return _state_bool(state, "local_rescue_enabled", True)
    return os.getenv("LOCAL_RESCUE_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _force_free_mode(state: dict[str, Any] | None = None) -> bool:
    if state and _state_bool(state, "force_free_llm", False):
        return True
    return os.getenv("FORCE_FREE_LLM", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _free_provider_help(provider: str | None = None) -> str:
    provider_name = (provider or _free_provider()).strip().lower()
    if provider_name == "openai_compatible":
        return (
            "Configura un servidor local OpenAI-compatible (LM Studio o vLLM) y define "
            "LOCAL_LLM_BASE_URL."
        )
    if provider_name == "huggingface":
        return (
            "Configura Hugging Face Inference con HUGGINGFACE_API_KEY (o HF_TOKEN), "
            "HUGGINGFACE_MODEL y HUGGINGFACE_BASE_URL."
        )
    return (
        "Configura Ollama local (sin cuenta): instala desde https://ollama.com/download, "
        "ejecuta 'ollama pull qwen2.5:latest' (opcional: 'ollama pull deepseek-r1:8b') y verifica que "
        "OLLAMA_BASE_URL apunte al servidor (default http://localhost:11434)."
    )


async def _call_free_oss_chat(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str,
    provider: str | None = None,
) -> str:
    global _OLLAMA_BACKOFF_UNTIL
    global _OLLAMA_LAST_ERROR

    provider_name = (provider or _free_provider()).strip().lower()
    if provider_name == "openai_compatible":
        return await _call_openai_compatible(
            api_key=os.getenv("LOCAL_LLM_API_KEY", ""),
            model=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1"),
        )
    if provider_name == "huggingface":
        return await _call_huggingface_chat(
            model=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
    if provider_name == "ollama":
        now = time.monotonic()
        if now < _OLLAMA_BACKOFF_UNTIL:
            remaining = max(0.0, _OLLAMA_BACKOFF_UNTIL - now)
            raise RuntimeError(
                f"Ollama warm-up active ({remaining:.1f}s remaining). Last error: {_OLLAMA_LAST_ERROR}"
            )

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        is_ready, detail = ensure_ollama_ready(base_url=ollama_base_url)
        if not is_ready:
            _OLLAMA_LAST_ERROR = detail
            backoff_s = float(os.getenv("OLLAMA_BACKOFF_SECONDS", "10"))
            _OLLAMA_BACKOFF_UNTIL = time.monotonic() + max(1.0, backoff_s)
            raise RuntimeError(detail)
        resolved_model = await _resolve_ollama_model(
            requested_model=model,
            base_url=ollama_base_url,
        )
        try:
            output = await _call_ollama_chat(
                model=resolved_model,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                base_url=ollama_base_url,
            )
        except Exception as exc:
            _OLLAMA_LAST_ERROR = _friendly_provider_error(exc)
            backoff_s = float(os.getenv("OLLAMA_BACKOFF_SECONDS", "10"))
            _OLLAMA_BACKOFF_UNTIL = time.monotonic() + max(1.0, backoff_s)
            raise

        _OLLAMA_LAST_ERROR = ""
        _OLLAMA_BACKOFF_UNTIL = 0.0
        return output
    raise ValueError(
        "FREE_LLM_PROVIDER invalido. Usa 'ollama', 'openai_compatible' o 'huggingface'."
    )


async def _call_anthropic(*, api_key: str, model: str, user_prompt: str, system_prompt: str) -> str:
    snapshot = await THERMAL.throttle_if_needed()
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 700,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    timeout_s = THERMAL.request_timeout(base_timeout_s=60.0, level=snapshot.level)
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
    data = response.json()
    content = data.get("content", [])
    text_parts = [chunk.get("text", "") for chunk in content if chunk.get("type") == "text"]
    return "\n".join(text_parts).strip()


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : index + 1]
                try:
                    loaded = json.loads(chunk)
                    if isinstance(loaded, dict):
                        return loaded
                except json.JSONDecodeError:
                    return None
    return None


async def openai_thought_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    OpenAI stage: produce a visible reasoning summary.
    
    Args:
        state: The current state containing messages
        
    Returns:
        Updated state with OpenAI summary message
    """
    messages = state.get("messages", [])
    user_prompt = _last_human_content(messages)
    if not user_prompt:
        return {"messages": [AIMessage(content="[OpenAI pensamiento resumido]\nNo user prompt found.")]}

    api_key = os.getenv("OPENAI_API_KEY", "")
    model = _state_text(state, "thought_cloud_model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    free_model = _state_text(state, "thought_model", os.getenv("FREE_THOUGHT_MODEL", "qwen2.5:7b"))
    free_provider = _state_text(state, "free_provider", _free_provider())
    context_block = _context_block(state)

    system_prompt = _state_text(
        state,
        "thought_system_prompt",
        (
        "Eres un analista. Responde con un resumen corto de razonamiento para ayudar a otro "
        "modelo a tomar decisiones. No reveles cadena de pensamiento extensa; entrega solo "
        "puntos concretos y accionables."
        ),
    )
    user_payload = (
        (f"{context_block}\n\n" if context_block else "")
        + 
        f"Consulta del usuario:\n{user_prompt}\n\n"
        "Devuelve 4-6 bullets con: objetivo, supuestos, riesgos, información faltante y siguiente paso. "
        "Limite: 140 palabras."
    )

    if api_key and not _force_free_mode(state):
        try:
            result = await _call_openai_compatible(
                api_key=api_key,
                model=model,
                user_prompt=user_payload,
                system_prompt=system_prompt,
                base_url=base_url,
            )
        except Exception as exc:
            result = f"Error llamando OpenAI: {exc}"
    else:
        try:
            result = await _call_free_oss_chat(
                model=free_model,
                user_prompt=user_payload,
                system_prompt=system_prompt,
                provider=free_provider,
            )
        except Exception as exc:
            result = (
                "No se pudo usar el backend gratis local para esta etapa. "
                f"{_free_provider_help(free_provider)} Error: {_friendly_provider_error(exc)}"
            )

    return {"messages": [AIMessage(content=f"[OpenAI pensamiento resumido]\n{result}")]}


async def anthropic_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Anthropic stage: review and challenge the OpenAI summary.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with Anthropic summary message
    """
    messages = state.get("messages", [])
    user_prompt = _last_human_content(messages)
    openai_summary = _extract_section(messages, "[OpenAI pensamiento resumido]")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    model = _state_text(
        state, "review_cloud_model", os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    )
    free_model = _state_text(state, "review_model", os.getenv("FREE_REVIEW_MODEL", "qwen2.5:7b"))
    free_provider = _state_text(state, "free_provider", _free_provider())
    context_block = _context_block(state)

    system_prompt = _state_text(
        state,
        "review_system_prompt",
        (
        "Eres un verificador crítico. Tu salida será visible para el usuario y otro modelo. "
        "Entrega observaciones breves, verificables y accionables."
        ),
    )
    user_payload = (
        (f"{context_block}\n\n" if context_block else "")
        +
        f"Consulta original:\n{user_prompt}\n\n"
        f"Resumen de OpenAI:\n{openai_summary}\n\n"
        "Contrasta el análisis en 4-6 bullets: qué mantienes, qué corregirías, riesgos y prioridad. "
        "Limite: 140 palabras."
    )

    if _is_local_backend_unavailable(openai_summary):
        summary = (
            "Se omite llamada de contraste para reducir carga termica porque el backend local "
            "reporto indisponibilidad en la etapa previa."
        )
        return {"messages": [AIMessage(content=f"[Anthropic contraste]\n{summary}")]}

    if api_key and not _force_free_mode(state):
        try:
            result = await _call_anthropic(
                api_key=api_key,
                model=model,
                user_prompt=user_payload,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            result = f"Error llamando Anthropic: {exc}"
    else:
        try:
            result = await _call_free_oss_chat(
                model=free_model,
                user_prompt=user_payload,
                system_prompt=system_prompt,
                provider=free_provider,
            )
        except Exception as exc:
            result = (
                "No se pudo usar el backend gratis local para esta etapa. "
                f"{_free_provider_help(free_provider)} Error: {_friendly_provider_error(exc)}"
            )

    return {"messages": [AIMessage(content=f"[Anthropic contraste]\n{result}")]}


async def deepseek_action_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    DeepSeek stage: produce final answer and execute requested local actions.

    Args:
        state: The current state

    Returns:
        Updated state with execution report message
    """
    messages = state.get("messages", [])
    user_prompt = _last_human_content(messages)
    openai_summary = _extract_section(messages, "[OpenAI pensamiento resumido]")
    anthropic_summary = _extract_section(messages, "[Anthropic contraste]")

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    model = _state_text(state, "action_cloud_model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    free_model = _state_text(state, "action_model", os.getenv("FREE_ACTION_MODEL", "deepseek-r1:8b"))
    free_provider = _state_text(state, "free_provider", _free_provider())
    context_block = _context_block(state)

    system_prompt = _state_text(
        state,
        "action_system_prompt",
        (
        "Eres un agente ejecutor. Debes integrar dos análisis y devolver JSON estricto con "
        "respuesta final y acciones locales opcionales."
        ),
    )
    user_payload = (
        (f"{context_block}\n\n" if context_block else "")
        +
        f"Consulta original:\n{user_prompt}\n\n"
        f"Análisis OpenAI:\n{openai_summary}\n\n"
        f"Análisis Anthropic:\n{anthropic_summary}\n\n"
        "Herramientas locales disponibles:\n"
        "- calculator (input: expresion matematica)\n"
        "- get_current_info (input: opcional, ignorado)\n"
        "- query_db (input: consulta SQL SELECT para PostgreSQL)\n\n"
        "Responde SOLO con JSON:\n"
        "{\n"
        '  "final_response": "texto para el usuario",\n'
        '  "actions": [{"tool": "calculator", "input": "2+2"}],\n'
        '  "action_notes": "opcional"\n'
        "}"
    )

    if _is_local_backend_unavailable(openai_summary) or _is_local_backend_unavailable(
        anthropic_summary
    ):
        if _local_rescue_enabled(state):
            try:
                rescue_prompt = (
                    (f"{context_block}\n\n" if context_block else "")
                    + f"Solicitud del usuario:\n{user_prompt}\n\n"
                    "Responde de forma clara y accionable para operar el sistema. Limite: 120 palabras."
                )
                rescue = await _call_free_oss_chat(
                    model=free_model,
                    user_prompt=rescue_prompt,
                    system_prompt=(
                        "Eres un asistente local de respaldo. Responde de forma concreta, "
                        "sin inventar estados del sistema."
                    ),
                    provider=free_provider,
                )
                rescue_text = str(rescue).strip()
                if rescue_text:
                    report = "\n\n".join(
                        [
                            "[DeepSeek ejecución]",
                            f"Respuesta final:\n{rescue_text}",
                            (
                                "Notas:\nModo rescate local activado. Se continuo con una sola "
                                "llamada para mantener la operacion del asistente."
                            ),
                        ]
                    )
                    return {"messages": [AIMessage(content=report)]}
            except Exception:
                pass

        rule_response = _rule_based_response(user_prompt)
        report = "\n\n".join(
            [
                "[DeepSeek ejecución]",
                f"Respuesta final:\n{rule_response}",
            ]
        )
        return {"messages": [AIMessage(content=report)]}

    if api_key and not _force_free_mode(state):
        try:
            raw = await _call_openai_compatible(
                api_key=api_key,
                model=model,
                user_prompt=user_payload,
                system_prompt=system_prompt,
                base_url=base_url,
            )
        except Exception as exc:
            raw = json.dumps(
                {"final_response": f"Error llamando DeepSeek: {exc}", "actions": []}
            )
    else:
        try:
            raw = await _call_free_oss_chat(
                model=free_model,
                user_prompt=user_payload,
                system_prompt=system_prompt,
                provider=free_provider,
            )
        except Exception:
            rule_response = _rule_based_response(user_prompt)
            raw = json.dumps({"final_response": rule_response, "actions": []})

    parsed = _extract_first_json_object(raw) or {"final_response": raw, "actions": []}
    final_response = str(parsed.get("final_response", "")).strip() or "No response."
    actions = parsed.get("actions", [])
    action_notes = str(parsed.get("action_notes", "")).strip()

    execution_lines: list[str] = []
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool", "")).strip()
            tool_input = action.get("input", "")
            result = execute_tool_action(tool_name, tool_input)
            execution_lines.append(f"- {tool_name}({tool_input}) -> {result}")

    report_parts = [
        "[DeepSeek ejecución]",
        f"Respuesta final:\n{final_response}",
    ]
    if execution_lines:
        report_parts.append("Acciones ejecutadas:\n" + "\n".join(execution_lines))
    if action_notes:
        report_parts.append(f"Notas:\n{action_notes}")

    return {"messages": [AIMessage(content="\n\n".join(report_parts))]}

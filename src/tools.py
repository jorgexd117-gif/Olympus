"""Tool definitions for the agent."""

import ast
import asyncio
import datetime
import os
import re
from typing import Any

import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

import httpx
from langchain_core.tools import tool

_URL_RE = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]{4,}')


def extract_urls(text: str) -> list[str]:
    """Extract all HTTP/HTTPS URLs from a text string."""
    return _URL_RE.findall(text)


async def fetch_url_content(url: str) -> dict:
    """Fetch and extract readable text content from a URL."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; OlympusAgent/1.0; +https://olympus.ai)"}
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, verify=False) as client:
            resp = await client.get(url, headers=headers)
            content_type = resp.headers.get("content-type", "")
            raw = resp.text

            title = ""
            if "html" in content_type.lower():
                raw = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
                raw = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
                raw = re.sub(r"<!--.*?-->", " ", raw, flags=re.DOTALL)
                title_m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
                if title_m:
                    title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()
                text_only = re.sub(r"<[^>]+>", " ", raw)
            else:
                text_only = raw

            text_only = re.sub(r"[ \t]{2,}", " ", text_only)
            text_only = re.sub(r"\n{3,}", "\n\n", text_only).strip()
            preview = text_only[:3000] + ("…" if len(text_only) > 3000 else "")

            return {
                "url": url,
                "title": title,
                "content": preview,
                "status_code": resp.status_code,
                "error": None,
            }
    except Exception as exc:
        return {
            "url": url,
            "title": "",
            "content": "",
            "status_code": 0,
            "error": str(exc),
        }


async def fetch_all_urls(urls: list[str]) -> list[dict]:
    """Fetch up to 3 URLs concurrently."""
    return list(await asyncio.gather(*[fetch_url_content(u) for u in urls[:3]]))


def _safe_eval_math(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression."""
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.FloorDiv,
        ast.Load,
    )
    parsed = ast.parse(expression, mode="eval")
    for node in ast.walk(parsed):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Unsupported expression")
    return eval(compile(parsed, "<calculator>", "eval"), {"__builtins__": {}}, {})


def run_calculator(expression: str) -> str:
    """Run calculator tool logic."""
    try:
        result = _safe_eval_math(expression)
        return str(result)
    except Exception as exc:
        return f"Error: {str(exc)}"


def get_current_info_text() -> str:
    """Return current system information."""
    return f"Current time: {datetime.datetime.now().isoformat()}"


_BLOCKED_SQL_PATTERNS = [
    r"(?i)\b(DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    r"(?i)\b(DELETE\s+FROM|UPDATE\s+\S+\s+SET)\b",
    r"(?i)\b(INSERT\s+INTO)\b",
    r"(?i);\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)",
    r"(?i)\bpg_shadow\b",
    r"(?i)\bpg_authid\b",
]

_ALLOWED_SQL_STARTS = [
    "select", "with", "explain", "show",
]


def _validate_sql(sql: str) -> tuple[bool, str]:
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False, "Empty query"
    first_word = stripped.split()[0].lower()
    if first_word not in _ALLOWED_SQL_STARTS:
        return False, f"Only SELECT/WITH/EXPLAIN/SHOW queries are allowed. Got: {first_word.upper()}"
    for pattern in _BLOCKED_SQL_PATTERNS:
        if re.search(pattern, stripped):
            return False, "Query contains blocked SQL operations"
    return True, ""


def run_query_db(sql: str) -> str:
    """Execute a read-only SQL query against the PostgreSQL database."""
    is_valid, error_msg = _validate_sql(sql)
    if not is_valid:
        return f"Error: {error_msg}"

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        return "Error: DATABASE_URL not configured"

    try:
        from sqlalchemy import create_engine, text as sa_text
        sync_url = db_url
        if "+asyncpg" in sync_url:
            sync_url = sync_url.replace("postgresql+asyncpg://", "postgresql://")

        engine = create_engine(sync_url, pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(sa_text(sql.strip()))
            columns = list(result.keys()) if result.returns_rows else []
            if not columns:
                return "Query executed (no rows returned)"
            rows = result.fetchmany(100)
            if not rows:
                return f"Columns: {', '.join(columns)}\nNo rows found"
            lines = [f"Columns: {', '.join(columns)}"]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            total = len(rows)
            if total == 100:
                lines.append("... (limited to 100 rows)")
            else:
                lines.append(f"({total} row{'s' if total != 1 else ''})")
            return "\n".join(lines)
        engine.dispose()
    except Exception as exc:
        return f"Error executing query: {exc}"


@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    
    Args:
        expression: A mathematical expression (e.g., "2 + 2")
        
    Returns:
        The result of the calculation
    """
    return run_calculator(expression)


@tool
def get_current_info() -> str:
    """
    Get current system information.
    
    Returns:
        Current system information
    """
    return get_current_info_text()


@tool
def query_db(sql: str) -> str:
    """
    Execute a read-only SQL query against the PostgreSQL database.
    Only SELECT, WITH, EXPLAIN, and SHOW queries are allowed.
    
    Args:
        sql: A SQL SELECT query
        
    Returns:
        Query results formatted as text
    """
    return run_query_db(sql)


tools = [calculator, get_current_info, query_db]


def execute_tool_action(tool_name: str, tool_input: Any) -> str:
    """Execute a named local tool action."""
    normalized = (tool_name or "").strip().lower()

    if normalized == "calculator":
        expression = str(tool_input or "")
        return run_calculator(expression)

    if normalized == "get_current_info":
        return get_current_info_text()

    if normalized == "query_db":
        sql = str(tool_input or "")
        return run_query_db(sql)

    if normalized == "fetch_url":
        url = str(tool_input or "").strip()
        if not url:
            return "Error: URL requerida"
        try:
            result = asyncio.run(fetch_url_content(url))
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, fetch_url_content(url))
                result = future.result(timeout=20)
        if result.get("error"):
            return f"Error al abrir {url}: {result['error']}"
        title = f"**{result['title']}**\n" if result.get("title") else ""
        return f"{title}URL: {url}\n\n{result['content']}"

    return f"Error: Unknown tool '{tool_name}'"

#!/bin/sh
set -e

PORT="${PORT:-3000}"

exec uvicorn src.api_server:app --host 0.0.0.0 --port "$PORT"

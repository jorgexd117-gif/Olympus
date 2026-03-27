# Thunder Client Setup

1. Install VS Code extension `rangav.vscode-thunder-client`.
2. Open Thunder Client inside VS Code.
3. Import:
   - `langgraph-app.postman_collection.json`
   - `langgraph-local.postman_environment.json`
4. Select environment `LangGraph Local (8010)`.
5. Run `Health` first to verify API connectivity.

## Variables

- `base_url`: API base URL (default `http://localhost:8010`)
- `project_id`: project identifier used by assistant/orchestrator requests
- `agent_key`: profile key for profile updates (default `action`)

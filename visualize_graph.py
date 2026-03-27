"""Generate visualizations for the LangGraph agent graph."""

from src.graph import compile_graph


def main() -> None:
    graph = compile_graph().get_graph()

    # Terminal visualization
    print(graph.draw_ascii())

    # Mermaid source visualization
    with open("Workspace/langgraph-app/graph.mmd", "w", encoding="utf-8") as f:
        f.write(graph.draw_mermaid())

    # PNG visualization (requires external Mermaid rendering API by default)
    try:
        graph.draw_mermaid_png(output_file_path="Workspace/langgraph-app/graph.png")
        print("Wrote Workspace/langgraph-app/graph.png")
    except Exception as exc:
        print(f"Skipping PNG export: {exc}")

    print("Wrote Workspace/langgraph-app/graph.mmd")


if __name__ == "__main__":
    main()

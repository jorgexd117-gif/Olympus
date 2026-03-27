"""LangGraph state and graph configuration."""

from typing import Annotated, Any
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
import operator
from src.nodes import anthropic_review_node, deepseek_action_node, openai_thought_node


class AgentState(TypedDict, total=False):
    """State for the agent."""

    messages: Annotated[list[BaseMessage], operator.add]
    agent_config: dict[str, Any]
    project_context: str
    memory_context: str


def create_graph() -> StateGraph:
    """
    Create the LangGraph state machine.
    
    Returns:
        Configured StateGraph
    """
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("openai_thought", openai_thought_node)
    graph.add_node("anthropic_review", anthropic_review_node)
    graph.add_node("deepseek_action", deepseek_action_node)
    
    # Add edges
    graph.set_entry_point("openai_thought")
    graph.add_edge("openai_thought", "anthropic_review")
    graph.add_edge("anthropic_review", "deepseek_action")
    graph.add_edge("deepseek_action", END)
    
    return graph


def compile_graph():
    """Compile and return the agent graph."""
    graph = create_graph()
    return graph.compile()

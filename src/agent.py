"""Main agent application."""

import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.graph import compile_graph
from src.thermal import get_thermal_regulator


async def main():
    """Run the agent application."""
    # Load environment variables
    load_dotenv()
    load_dotenv(".env.local", override=True)
    
    thermal = get_thermal_regulator()
    thermal.start()
    snapshot = thermal.current_snapshot()
    print(
        "Thermal status:"
        f" level={snapshot.level} temp={snapshot.cpu_temp_c if snapshot.cpu_temp_c is not None else 'N/A'}"
        f" cooldown={snapshot.recommended_cooldown_s:.1f}s"
    )

    # Compile the graph
    agent = compile_graph()
    
    # Example interaction
    initial_state = {
        "messages": [HumanMessage(content="whats the business with most profits online 2025 no initial inversion" \
        "")]
    }
    
    print("Starting agent...")
    result = await agent.ainvoke(initial_state)
    
    print("\nFinal state:")
    for message in result["messages"]:
        print(f"  {message.__class__.__name__}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())

from langgraph.graph import StateGraph, END
from scripts.state_graph import GraphState
from scripts.agents import generator_agent_node, retrieve_agent_node, supervisor_agent_node, route_logic
workflow = StateGraph(GraphState)

workflow.add_node("generator_agent", generator_agent_node)
workflow.add_node("retriever_agent", retrieve_agent_node)
workflow.add_node("supervisor_agent", supervisor_agent_node)

workflow.set_entry_point("supervisor_agent")

workflow.add_edge("generator_agent", "supervisor_agent")
workflow.add_edge("retriever_agent", "supervisor_agent")

workflow.add_conditional_edges(
    "supervisor_agent",
    route_logic,
    {
        "generator_agent": "generator_agent",
        "retriever_agent": "retriever_agent",
        "__end__": END
    }
)

graph = workflow.compile()
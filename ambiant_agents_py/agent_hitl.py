import operator
from typing import Literal, cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain.tools import tool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

_ = load_dotenv(override=True)


class ClassificationSchema(BaseModel):
    reasoning: str = Field(
        description="Step by step reasoning behing the classification"
    )
    classification: Literal["bug", "question", "unsure", "billing"] = Field(
        description="The classification of the request, 'bug': a bug report, 'question': a question, 'unsure': unsure, 'billing': billing related"
    )


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    request: str
    classification_decision: ClassificationSchema


@tool
def write_request_response(receiver: str, sender: str, content: str) -> str:
    """
    Write a request response message.
    receiver: who the message is sent to
    sender: who the message is sent from
    content: the content of the message
    """
    return f"From {sender} to {receiver}: {content}"


llm = init_chat_model("openai:gpt-5-nano")
llm_router = llm.with_structured_output(ClassificationSchema)
llm_with_tools = llm.bind_tools([write_request_response])


def triage_router(state: MessagesState) -> dict:
    """Classify the request and store the classification decision."""
    request = state["request"]

    classification = cast(
        ClassificationSchema,
        llm_router.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that classifies user requests. Classify the request as: bug, question, unsure, or billing.",
                },
                {"role": "user", "content": request},
            ]
        ),
    )

    return {
        "classification_decision": classification,
    }


def route_after_triage(state: MessagesState) -> str:
    """Route to agent subgraph if classification is 'bug' or 'unsure', otherwise end."""
    classification = state.get("classification_decision")
    if classification and classification.classification in [
        "bug",
        "unsure",
        "question",
    ]:
        return "response_agent"
    return "__end__"


def call_model(state: MessagesState) -> dict[str, list[AnyMessage]]:
    # Get the classification to understand what kind of response to provide
    classification_decision = state.get("classification_decision")
    classification_type = (
        classification_decision.classification if classification_decision else "unknown"
    )

    # Build a dynamic system prompt based on classification
    system_content = f"""You are a helpful assistant. The user's request has been classified as: '{classification_type} because {classification_decision.reasoning}'.

Based on this classification:
- If 'question': Provide a helpful answer to the user's question.
- If 'unsure': Ask the user clarifying questions to better understand their request.
- If 'bug': Report the bug using the write_request_response tool.
"""

    messages = [SystemMessage(content=system_content)] + state.get("messages", [])
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
    }


def run_tool(state: MessagesState):
    results = []
    last_message = state["messages"][-1]

    if (
        isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and last_message.tool_calls
    ):
        for tool_call in last_message.tool_calls:
            observation = write_request_response.invoke(tool_call["args"])
            results.append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )

    return {"messages": results}


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if (
        isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and last_message.tool_calls
    ):
        return "run_tool"
    return END


def human_verify(state: MessagesState):
    classification = state["classification_decision"].classification
    if classification == "bug":
        human_decision = interrupt(
            {
                "classification": state["classification_decision"].classification,
                "reason": state["classification_decision"].reasoning,
            }
        )
        if human_decision["feedback"]:
            feedback = human_decision["feedback"]
            feedback_message = f"Feedback: {feedback}"
            messages = state["messages"] + [HumanMessage(content=feedback_message)]
            return Command(update={"messages": messages}, goto="call_model")
        if human_decision["approved"]:
            return Command(update={}, goto="call_model")
        else:
            return Command(
                update={"messages": [AIMessage(content="verification denied")]},
                goto=END,
            )
    else:
        return Command(update={}, goto="call_model")


graph = StateGraph(MessagesState)
graph.add_node("call_model", call_model)
graph.add_node("run_tool", run_tool)
graph.add_edge(START, "call_model")
graph.add_conditional_edges(
    "call_model", should_continue, {"run_tool": "run_tool", END: END}
)
graph.add_edge("run_tool", "call_model")
graph.add_edge("call_model", END)

agent = graph.compile()

agent_hitl = (
    StateGraph(MessagesState)
    .add_node("triage_router", triage_router)
    .add_node("response_agent", agent)
    .add_node("human_decision", human_verify)
    .add_edge(START, "triage_router")
    .add_edge("triage_router", "human_decision")
    .add_conditional_edges(
        "human_decision",
        route_after_triage,
        {"response_agent": "response_agent", "__end__": "__end__"},
    )
).compile()

# response = llm_with_tools.invoke(
#     "Draft a request response to my boss saying ill be off in christmass"
# )
# print(response.content)
# args = response.tool_calls[0]["args"]
# tool_message = write_request_response.invoke(args)
# print(tool_message)
# llm_with_tools.invoke(tool_message)

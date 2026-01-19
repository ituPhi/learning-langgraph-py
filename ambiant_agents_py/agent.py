# import operator
import time
from typing import Literal, cast

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain.tools import tool
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.graph import MessagesState
from langgraph.graph.state import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field


# from typing_extensions import Annotated
#  define state as a BaseModel
class ClassificationSchema(BaseModel):
    """
    This is a schema for classification. Set up with Pydantic
    """

    reasoning: str = Field("the reason")
    classification: Literal[
        "positive",
        "negative",
        "neutral",
    ] = Field(description="the classification")


# define state
class State(MessagesState):
    """
    set up messages state with a reducer
    """

    # messages: Annotated[list[AnyMessage], operator.add]
    # add only extra fields to messageState istead of using the reducer
    request: str
    classification_descision: ClassificationSchema


# Define a tool


@tool
def send_response_request(sender: str, receiver: str, content: str) -> str:
    """
    send a report to the support center
    sender : the sender of the message, ask the user who he is if unsure
    receiver : Support center + sub department
    content: the content of the request
    """
    # mock api call
    try:
        print("Trying to send report")
        time.sleep(0.01)
        print("Report sent")
    except Exception as e:
        return f"Error sending report: {e}"

    return f"Sent report from {sender} to {receiver}: {content}"


llm = init_chat_model("openai:gpt-5-nano")
llm_router = llm.with_structured_output(ClassificationSchema)
llm_with_tools = llm.bind_tools([send_response_request])


# Build base agent loop


def call_model(state: State) -> State:
    classification = cast(dict, state["classification_descision"])
    type = classification["classification"]
    reason = classification["reasoning"]
    request = state["request"]

    system_message = SystemMessage(
        content=f"The following {request} has been classified as {type} because {reason}"
    )
    response = llm_with_tools.invoke([system_message])
    return {
        "messages": [response],
        "request": "",
        "classification_descision": state["classification_descision"],
    }


def run_tool(state: State):
    """
    run our tool and return a tool message
    """
    results = []
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for call in last_message.tool_calls:
            tool_result = send_response_request.invoke(call["args"])
            results.append(ToolMessage(content=tool_result, tool_call_id=call["id"]))

    return {"messages": results}


def should_continue(state: State) -> Literal["run_tool", "__end__"]:
    """
    determine if we should continue calling tools
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "run_tool"

    return "__end__"


# Build agent graph
agent_graph = (
    StateGraph(State)
    .add_node("call_model", call_model)
    .add_node("run_tool", run_tool)
    .add_edge(START, "call_model")
    .add_conditional_edges(
        "call_model", should_continue, {"run_tool": "run_tool", END: "__end__"}
    )
    .add_edge("run_tool", "call_model")
    .add_edge("call_model", END)
).compile()


def router(state: State) -> Command[Literal["agent", "__end__"]]:
    """
    Classifies and routes the request to the appropriate agent
    """
    request = state["request"]
    classification = cast(
        ClassificationSchema,
        llm_router.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a deterministic request classifier. Classify the following human request as : positive, negative or neutral. Provide a clear reason for your decision in the reasoning field.",
                },
                {"role": "user", "content": request},
            ]
        ),
    )
    if classification.classification == "negative":
        return Command(
            goto="agent",
            update={
                "messages": [
                    AIMessage(
                        content="The request is negative, lets respond creating a ticket with our support team"
                    )
                ]
            },
        )

    return Command(goto="__end__", update={})

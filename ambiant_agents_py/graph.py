import operator

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain.tools import tool
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

_ = load_dotenv(override=True)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    # request: str
    # request_response: str


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
llm_with_tools = llm.bind_tools([write_request_response])


# def write_request_node(state: MessagesState):
#     request = state.get("request")
#     response = llm_with_tools.invoke(request)
#     args = response.tool_calls[0]["args"]
#     tool_message = write_request_response.invoke(args)
#     request_response = llm_with_tools.invoke(tool_message)
#     return {"request_response": request_response}


def call_model(state: MessagesState) -> dict[str, list[AnyMessage] | int]:
    messages = [
        SystemMessage(
            content=" You are a helpful assistant, if the user makes a direct request,write a brief response to the users request using the write_request_response tool then show the user the response"
        )
    ] + state.get("messages")
    response = llm_with_tools.invoke(messages)

    # print(len(response.tool_calls))
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
    if last_message.tool_calls:
        return "run_tool"
    return END


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


# response = llm_with_tools.invoke(
#     "Draft a request response to my boss saying ill be off in christmass"
# )
# print(response.content)
# args = response.tool_calls[0]["args"]
# tool_message = write_request_response.invoke(args)
# print(tool_message)
# llm_with_tools.invoke(tool_message)

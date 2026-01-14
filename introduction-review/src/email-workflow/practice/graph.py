import os
import random
import sqlite3
from operator import add
from sqlite3.dbapi2 import sqlite_version
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import List

load_dotenv()
conn = sqlite3.connect(":memory:")
api_key = os.environ.get("OPENAI_API_KEY")


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of the two integers.
    """
    return a * b


if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


model = ChatOpenAI(model="gpt-5-nano")
model_with_tools = model.bind_tools([multiply])


def count_reducer(a: list[int], b: list[int]) -> list:
    return a + b


class MState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    count: Annotated[list[int], count_reducer]


class MessageState(MessagesState):
    count: Annotated[list[int], count_reducer]


def node_one(state: MessageState):
    print("calling node_one")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Respond in Spanish, always use your tools for math operations",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    formated = prompt.invoke({"messages": state["messages"]})

    # or messages = [system_message] + state["messages"]
    response = model_with_tools.invoke(formated)
    # print(state)
    return {"messages": [response], "count": [state["count"][-1] + 1]}


# def node_two(state: State):
#     print("calling node_two")
#     state = state.copy()
#     state["graph_state"] += " node_two"
#     # print(state)
#     return state


# def node_three(state: State):
#     print("calling node_three")
#     state = state.copy()
#     state["graph_state"] += " node_three"
#     # print(state)
#     return state


# def condition(state: State) -> Literal["node_two", "node_three"]:
#     return random.choice(["node_two", "node_three"])


graph_builder = StateGraph(MessageState)
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("tools", ToolNode([multiply]))
# graph_builder.add_node("node_two", node_two)
# graph_builder.add_node("node_three", node_three)

graph_builder.add_edge(START, "node_one")
graph_builder.add_conditional_edges("node_one", tools_condition)
graph_builder.add_edge("tools", "node_one")
graph_builder.add_edge("node_one", END)
# graph_builder.add_conditional_edges("node_one", condition)
# graph_builder.add_edge("node_three", END)

agent = graph_builder.compile(checkpointer=MemorySaver())


result = agent.stream(
    {"messages": [HumanMessage("Hi how are you today")], "count": [0]},
    {"configurable": {"thread_id": "1"}},
    stream_mode="values",
)
# current = agent.get_state({"configurable": {"thread_id": "1"}})
for chunk in result:
    message_chunk = chunk["messages"]
    for message in message_chunk:
        print(message.pretty_print())

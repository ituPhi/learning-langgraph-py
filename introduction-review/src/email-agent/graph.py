import os
import uuid
from imaplib import Commands
from typing import Literal, TypedDict, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.chat_agent_executor import Prompt
from langgraph.types import Command, Interrupt, interrupt
from pydantic import UUID1
from pydantic.types import UUID4
from requests.sessions import Request

load_dotenv()


class RequestClassificationSchema(TypedDict):
    intent: Literal["bug", "question", "unsure", "billing"]
    urgency: Literal["low", "medium", "high"]
    summary: str


class RequestAgentState(TypedDict):
    request_id: str
    sender_email: str
    content: str
    classification: RequestClassificationSchema | None
    ticked_id: uuid.UUID | None
    search_results: list[str] | None
    customer_history: dict | None
    draft_response: str | None


model = ChatOpenAI(model="gpt-5-nano")


def read_request(state: RequestAgentState):
    pass


def classify_request(
    state: RequestAgentState,
) -> dict[str, RequestClassificationSchema]:
    structured_output = model.with_structured_output(RequestClassificationSchema)

    classification_prompt = f"Classify the following email as a bug, question, unsure, or billing request. Also, determine the urgency level (low, medium, high) and provide a brief summary of the email content. Email content: {state['content']} from : {state['sender_email']}"

    response = structured_output.invoke(classification_prompt)
    response = cast(RequestClassificationSchema, response)

    return {"classification": response}


def search_documentation(
    state: RequestAgentState,
) -> dict[str, list[str]] | RequestAgentState:
    classification = state.get("classification", {})
    if classification is None:
        return state
    query = f"{classification.get('intent', '')} {classification.get('summary', '')}"
    search_results: list[str] = []
    try:
        mock_search_result = ["Document one", "Document two", "Document three"]
        search_results = mock_search_result
    except Exception as e:
        print(f"Error searching documentation: {e}")
        search_results = [f"search temporarily unavailable {str(e)}"]
    return {"search_results": search_results}


def bug_tracking(state: RequestAgentState) -> dict[str, UUID4]:
    ticked_id = uuid.uuid4()
    return {"ticked_id": ticked_id}


def write_response(
    state: RequestAgentState,
) -> Command[Literal["needs_review", "send_response"]]:
    classification = state.get("classification", {}) or {}
    search_results = state.get("search_results", [])

    context_section = []

    if search_results is not None:
        formated_docs = "\n".join(f"{doc}" for doc in search_results)
        context_section.append(formated_docs)
    customer_history = state.get("customer_history", "None")
    if customer_history is not None:
        context_section.append(f"Customer history: {customer_history}")

    prompt = f"""
Draft a response to this request :
{state["content"]} from {state["sender_email"]}
Metadata :
- Email Intent: {classification.get("intent", "")}
- Urgency: {classification.get("urgency", "")}
- Extra Context: {chr(10).join(context_section)}

Guidelines :
        - be helpful
        - be concise
        - be polite
    """

    response = model.invoke(prompt)

    needs_review = (
        classification.get("urgency") in ["medium", "high"]
        or classification.get("intent") == "bug"
    )

    if needs_review:
        goto = "needs_review"
    else:
        goto = "send_response"

    return Command(update={"draft_response": response}, goto=goto)


def needs_review(state: RequestAgentState) -> Command[str]:
    classification = state.get("classification", {})
    if classification is None:
        classification = {}

    human_decision = interrupt(
        {
            "request_id": state.get("request_id"),
            "request_content": state.get("content", ""),
            "request_sender": state.get("sender_email", ""),
            "urgency": classification.get("urgency", ""),
            "intent": classification.get("intent", ""),
        }
    )

    if human_decision["approved"]:
        return Command(
            update={"draft_response": state["draft_response"]}, goto="send_response"
        )
    else:
        return Command(update={}, goto=END)


def send_response(state: RequestAgentState):
    response = state.get("draft_response", "")
    print(f"Sending response to {state['sender_email']}: {response}")


builder = StateGraph(RequestAgentState)
builder.add_node("read_request", read_request)
builder.add_node("classify_request", classify_request)
builder.add_node("bug_tracking", bug_tracking)
builder.add_node("search_documentation", search_documentation)
builder.add_node("write_response", write_response)
builder.add_node("needs_review", needs_review)
builder.add_node("send_response", send_response)

# add edges
builder.add_edge(START, "read_request")
builder.add_edge("read_request", "classify_request")
builder.add_edge("classify_request", "bug_tracking")
builder.add_edge("classify_request", "search_documentation")
builder.add_edge("bug_tracking", "write_response")
builder.add_edge("search_documentation", "write_response")
builder.add_edge("send_response", END)
agent = builder.compile()

# initial_state = {
#     "request_id": "12345",
#     "content": "I'm experiencing a critical issue where the application crashes whenever I try to export data to CSV format. This happens consistently on every attempt. Please help me resolve this as soon as possible.",
#     "sender_email": "john.doe@example.com",
# }

# config = {"configurable":{"thread_id": "1"}}

# result = agent.invoke(initial_state, config)
# print(result)

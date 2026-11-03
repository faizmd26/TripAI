import os
import uuid
import operator
from typing import TypedDict, Annotated

import certifi
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from tools.tavilty_tool import tavily_search  # file name kept exactly as in your project
from tools.flight_tool import search_flights


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")

    if "sslmode" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0.7,
)


class TravelState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    final_plan: str
    llm_calls: int


def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content=f"Flight search results:\n{flight_data}"),
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def hotel_agent(state: TravelState):
    query = state["user_query"]
    hotel_data = tavily_search(query)

    return {
        "hotel_results": hotel_data,
        "messages": [
            AIMessage(content=f"Hotel search results:\n{hotel_data}"),
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def itinerary_agent(state: TravelState):
    prompt = f"""
User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt),
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def final_agent(state: TravelState):
    prompt = f"""
Generate a final travel plan based on the following information:
user_query: {state['user_query']}
Flight Results: {state['flight_results']}
Hotel Results: {state['hotel_results']}
itinerary: {state['itinerary']}

Format the output as a JSON object with the following keys:
1. Trip summary
2. Flight details
3. Hotel details
4. day-wise itinerary
5. estimated budget
6. final recommendations

Important:
- Be concise and clear in your responses.
- Ensure the JSON is valid and properly formatted.
- Mention the live flight API may not provide ticket prices if the prices are not available.
- Keep the response useful for real travel planning, including practical tips and advice.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt),
    ])

    return {
        "final_plan": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "final_plan": "",
            "llm_calls": 0,
        },
        config=config,
    )

    final_answer = result.get("final_plan") or result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
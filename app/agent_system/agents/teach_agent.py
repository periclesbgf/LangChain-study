import logfire
import asyncio

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CODE = os.getenv("CODE")
SECRET_EDUCATOR_CODE = os.getenv("SECRET_EDUCATOR_CODE")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_URI = os.getenv("MONGO_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
SECRET_KEY = os.getenv("SECRET_KEY")


agent = Agent('openai:gpt-4o-mini')


"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode


import operator
from typing import List, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

class OpenFilesSchema(BaseModel):
    """Schema for the open_files tool."""
    file_paths: List[str] = Field(description="List of file paths to open, relative to the codebase root.")

class ChatState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | ToolMessage], add_messages]
    all_files_opened: Annotated[List[str], operator.add]

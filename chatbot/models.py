import operator
from typing import List, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

class OpenFilesSchema(BaseModel):
    """Schema for the open_files tool."""
    file_paths: List[str] = Field(description="List of file paths to open, relative to the codebase root.")

class ChatState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | ToolMessage], add_messages]
    all_files_opened: Annotated[List[str], operator.add]
    kb_exploration_rounds: int = 0  # Track KB-specific exploration rounds
    generating_kb: bool = False     # Flag for KB generation mode
    knowledge_base: Optional[str] = None  # Store generated KB
    command: str
    summary: str = ""  # Add summary field
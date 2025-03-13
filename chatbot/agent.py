from typing import Dict, Any
import logging
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .models import ChatState
from .filesystem import FileSystem
from .tools import create_tools
from config import session, AWS_REGION, BEDROCK_MODEL

logger = logging.getLogger(__name__)

def route_tools(state: ChatState):
    if not state["messages"]:
        return END
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

class CodeExplorerChatbot:
    def __init__(self, codebase_path: str):
        self.fs = FileSystem(codebase_path)
        self.checkpointer = MemorySaver()
        self.llm = ChatBedrock(
            model_id=BEDROCK_MODEL,
            region_name=session.region_name or AWS_REGION,
            model_kwargs={"temperature": 0.7, "max_tokens": 200000}
        )
        self.tools, openai_tools = create_tools(self.fs)
        self.llm_with_tools = self.llm.bind_tools(openai_tools)
        self._initialize_workflow()

    def _initialize_workflow(self):
        async def agent(state: ChatState, config) -> ChatState:
            response = await self.llm_with_tools.ainvoke(state["messages"], config)
            return {
                "messages": [response]
            }
    
        def execute_tools(state: ChatState) -> ChatState:
            messages = []
            last_message = state["messages"][-1]
            all_files_opened = []
            
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_id = tool_call["id"]
                
                # Create a more structured response for better UI display
                if tool_name == "open_files":
                    file_paths = tool_call["args"].get("file_paths", [])
                    result = self.tools[1].func(file_paths)
                    all_files_opened.extend(file_paths)
                    # Add metadata for better UI rendering
                    tool_metadata = {
                        "tool_name": tool_name,
                        "files_count": len(file_paths),
                        "files": file_paths
                    }
                elif tool_name == "get_file_structure":
                    result = self.tools[0].func()
                    tool_metadata = {
                        "tool_name": tool_name,
                        "structure_size": len(result.split('\n'))
                    }
                else:
                    result = f"Unknown tool: {tool_name}"
                    tool_metadata = {"tool_name": tool_name}
                
                messages.append(ToolMessage(
                    content=result, 
                    tool_call_id=tool_id,
                    additional_kwargs={"metadata": tool_metadata}
                ))
            
            return {
                "messages": messages,
                "all_files_opened": all_files_opened
            }

        workflow = StateGraph(ChatState)
        workflow.add_node("agent", agent)
        workflow.add_node("tools", execute_tools)
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges(
            "agent",
            route_tools,
            {"tools": "tools", END: END}
        )
        
        workflow.add_edge("tools", "agent")
        self.app = workflow.compile(
            checkpointer=self.checkpointer
        )

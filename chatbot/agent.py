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
from langchain_core.messages import RemoveMessage

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

    # Node implementations as class methods
    def _preprocessor_node(self, state: ChatState) -> ChatState:
        if not state["messages"]:
            return state
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage) and isinstance(last_message.content, str):
            content = last_message.content.lower().strip()
            if "generate kb" in content or "generate knowledge base" in content:
                return {**state, "command": "generate_kb"}
        return {**state, "command": None}

    async def _agent_node(self, state: ChatState, config) -> ChatState:
        last_message = state["messages"][-1]
        generating_kb = state.get("generating_kb", False)

        response = await self.llm_with_tools.ainvoke(state["messages"], config)

        if not response.tool_calls and generating_kb:
            {"messages": [response, HumanMessage(content="Continue to explore")]}

        if isinstance(last_message, ToolMessage) and hasattr(last_message, 'metadata') and last_message.metadata["tool_name"] == "open_files":
            last_message.content = last_message.content[:300] + "..."
            self.app.update_state({"configurable": {"thread_id": "1"}}, {"messages": last_message})
        return {"messages": [response]}

    def _execute_tools_node(self, state: ChatState) -> ChatState:
        messages = []
        last_message = state["messages"][-1]
        all_files_opened = []
        generating_kb = state.get("generating_kb", False)
        kb_exploration_rounds = state.get("kb_exploration_rounds", 0)
        used_open_files = False
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_id = tool_call["id"]
            if tool_name == "open_files":
                file_paths = tool_call["args"].get("file_paths", [])
                result = self.tools[1].func(file_paths)
                all_files_opened.extend(file_paths)
                used_open_files = True
                tool_metadata = {"tool_name": tool_name, "files": file_paths}
            elif tool_name == "get_file_structure":
                result = self.tools[0].func()
                tool_metadata = {"tool_name": tool_name}
            else:
                result = f"Unknown tool: {tool_name}"
                tool_metadata = {"tool_name": tool_name}
            messages.append(ToolMessage(
                content=result, 
                tool_call_id=tool_id,
                metadata=tool_metadata
            ))
        
        if generating_kb and used_open_files:
            kb_exploration_rounds += 1
        
        return {
            "messages": messages,
            "all_files_opened": all_files_opened,
            "kb_exploration_rounds": kb_exploration_rounds,
            "generating_kb": generating_kb,
            "command": None
        }

    def _start_kb_exploration_node(self, state: ChatState) -> ChatState:
        kb_instruction = """
        Explore the tools to build a comprehensive understanding of the codebase through iterative file analysis, summarization, and strategic file selection.

        In each round:
        1. Request to open up to 5 files each time 
        2. Receive file contents (simulated through tools)
        3. Generate a structured key learnings and findings for opended files including Key classes, functions and their responsibilities. Ensure it is detailed enough because the raw file content will be removed later. 
        4. Propose next files to explore based on new findings and use tools to open it.
        """
        return {
            "messages": [HumanMessage(content=kb_instruction)],
            "generating_kb": True,
            "kb_exploration_rounds": 0,
            "command": None
        }

    async def _generate_knowledge_base_node(self, state: ChatState, config) -> ChatState:
        kb_prompt = HumanMessage(content=f"""
        Based on your exploration, generate a comprehensive knowledge base document that explains:
        
        1. Overall architecture and component relationships
        2. Key classes, functions and their responsibilities
        3. Main workflows and control flows
        4. Important APIs and integration points
        5. Design patterns and implementation details
        
        Structure your response as a well-organized technical document with clear sections.
        No tools to use at this time.
        """)
        
        kb_response = await self.llm_with_tools.ainvoke(
            state["messages"] + [kb_prompt],
            config
        )
        return {
            "messages": [kb_prompt, kb_response],
            "knowledge_base": kb_response.content,
            "generating_kb": False,
            "command": None
        }
    def _route_after_preprocessing(self, state: ChatState):
        command = state.get("command")
        if command == "generate_kb":
            return "start_kb_exploration"
        return "agent"
    
    def _route_tools_node(self, state: ChatState):
        generating_kb = state.get("generating_kb", False)

        if not state["messages"]:
            return END
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        if generating_kb:
            return "agent"
        return END
    
    def _route_after_tools(self, state: ChatState):
        if state.get("generating_kb", False) and state.get("kb_exploration_rounds", 0) > 2:
            return "generate_kb"
        return 'agent'
    
    def _initialize_workflow(self):
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("preprocessor", self._preprocessor_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._execute_tools_node)
        workflow.add_node("start_kb_exploration", self._start_kb_exploration_node)
        workflow.add_node("generate_kb", self._generate_knowledge_base_node)
        
        # Set entry point
        workflow.set_entry_point("preprocessor")
        
        # Connect preprocessor router
        workflow.add_conditional_edges(
            "preprocessor",
            self._route_after_preprocessing,
            {
                "agent": "agent",
                "start_kb_exploration": "start_kb_exploration"
            }
        )
        
        # Connect agent router
        workflow.add_conditional_edges(
            "agent",
            self._route_tools_node,
            {
                "tools": "tools",
                END: END
            }
        )
        
        # Standard edges
        workflow.add_conditional_edges("tools", self._route_after_tools, {
            "agent": "agent",
            "generate_kb": "generate_kb"
        })
        workflow.add_edge("start_kb_exploration", "agent")
        workflow.add_edge("generate_kb", END)
        
        self.app = workflow.compile(checkpointer=self.checkpointer)
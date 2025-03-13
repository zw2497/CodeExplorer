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
        # Preprocessor node checks for special commands before agent processes them
        def preprocessor(state: ChatState) -> ChatState:
            if not state["messages"]:
                return state
                
            last_message = state["messages"][-1]
            
            # Check if this is a KB generation request from user
            if isinstance(last_message, HumanMessage) and isinstance(last_message.content, str):
                content = last_message.content.lower().strip()
                if "generate kb" in content or "generate knowledge base" in content:
                    # Set special command flag
                    return {
                        **state,
                        "command": "generate_kb"
                    }
            
            # No special command
            return {
                **state,
                "command": None
            }
        
        # Router after preprocessing
        def route_after_preprocessing(state: ChatState):
            # Check for special commands first
            command = state.get("command")
            if command == "generate_kb":
                return "start_kb_exploration"
            
            # Check if KB exploration is complete (15 rounds)
            if state.get("generating_kb", False) and state.get("kb_exploration_rounds", 0) >= 15:
                return "generate_kb"
                
            # Normal flow
            return "agent"
        
        # Regular agent node (unchanged)
        async def agent(state: ChatState, config) -> ChatState:
            response = await self.llm_with_tools.ainvoke(state["messages"], config)
            return {"messages": [response]}
        
        # Tools execution (mostly unchanged)
        def execute_tools(state: ChatState) -> ChatState:
            messages = []
            last_message = state["messages"][-1]
            all_files_opened = []
            
            # Get KB-specific state
            generating_kb = state.get("generating_kb", False)
            kb_exploration_rounds = state.get("kb_exploration_rounds", 0)
            
            # Track if open_files was used
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
                    additional_kwargs={"metadata": tool_metadata}
                ))
            
            # Increment KB rounds counter if we're generating KB and opened files
            if generating_kb and used_open_files:
                kb_exploration_rounds += 1
            
            return {
                "messages": messages,
                "all_files_opened": all_files_opened,
                "kb_exploration_rounds": kb_exploration_rounds,
                "generating_kb": generating_kb,
                "command": None  # Clear command flag
            }
        
        # Node to initiate KB exploration
        def start_kb_exploration(state: ChatState) -> ChatState:
            kb_instruction = """
            I need you to perform 15 rounds of code exploration to generate a comprehensive knowledge base.
            For each round:
            1. Choose important files to explore based on what you've learned so far
            2. Use the open_files tool to examine their content
            3. Build up your understanding of the codebase structure and design
            
            Don't stop early - complete all 15 rounds before generating the knowledge base.
            After each tool use, briefly summarize what you've learned and what to explore next.
            """
            
            return {
                "messages": [HumanMessage(content=kb_instruction)],
                "generating_kb": True,
                "kb_exploration_rounds": 0,
                "command": None  # Clear command flag
            }
        
        # Knowledge base generation node
        async def generate_knowledge_base(state: ChatState, config) -> ChatState:
            files_opened = state.get("all_files_opened", [])
            unique_files = list(set(files_opened))
            
            kb_prompt = f"""
            Based on your exploration of {len(unique_files)} files across 15 rounds,
            generate a comprehensive knowledge base document that explains:
            
            1. Overall architecture and component relationships
            2. Key classes, functions and their responsibilities
            3. Main workflows and control flows
            4. Important APIs and integration points
            5. Design patterns and implementation details
            
            Structure your response as a well-organized technical document with clear sections.
            """
            
            kb_response = await self.llm.ainvoke(
                [HumanMessage(content="You are a code documentation expert."),
                HumanMessage(content=kb_prompt)],
                config
            )
            
            return {
                "messages": [AIMessage(content="✅ Knowledge base generated successfully after 15 rounds of exploration!")],
                "knowledge_base": kb_response.content,
                "generating_kb": False,  # Reset KB generation mode
                "command": None  # Clear command flag
            }
        
        # Standard tool routing
        def route_tools(state: ChatState):
            if not state["messages"]:
                return END
                
            last_message = state["messages"][-1]
            
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
                
            return END

        # Build graph with the new structure
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("preprocessor", preprocessor)
        workflow.add_node("agent", agent)
        workflow.add_node("tools", execute_tools)
        workflow.add_node("start_kb_exploration", start_kb_exploration)
        workflow.add_node("generate_kb", generate_knowledge_base)
        
        # Set preprocessor as entry point
        workflow.set_entry_point("preprocessor")
        
        # Connect preprocessor router
        workflow.add_conditional_edges(
            "preprocessor",
            route_after_preprocessing,
            {
                "agent": "agent",
                "start_kb_exploration": "start_kb_exploration",
                "generate_kb": "generate_kb"
            }
        )
        
        # Connect agent router
        workflow.add_conditional_edges(
            "agent",
            route_tools,
            {
                "tools": "tools",
                END: END
            }
        )
        
        # Standard edges
        workflow.add_edge("tools", "agent")
        workflow.add_edge("start_kb_exploration", "agent")
        workflow.add_edge("generate_kb", "agent")
        
        self.app = workflow.compile(
            checkpointer=self.checkpointer
        )

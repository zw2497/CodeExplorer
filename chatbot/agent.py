from typing import Dict, Any
import logging
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .models import ChatState
from .filesystem import FileSystem
from .tools import create_tools
from config import session, AWS_REGION, BEDROCK_MODEL

logger = logging.getLogger(__name__)

def route_tools(state: ChatState):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

class CodeExplorerChatbot:
    def __init__(self, codebase_path: str):
        logger.info(f"Initializing CodeExplorerChatbot with codebase path: {codebase_path}")
        self.fs = FileSystem(codebase_path)
        
        # Initialize Bedrock client
        self.llm = ChatBedrock(
            model_id=BEDROCK_MODEL,
            region_name=session.region_name or AWS_REGION,
            credentials_profile_name="default",
            model_kwargs={"temperature": 0.7, "max_tokens": 200000}
        )
        logger.info(f"Connected to Bedrock model: {BEDROCK_MODEL}")

        # Create tools
        self.tools, openai_tools = create_tools(self.fs)
        self.llm_with_tools = self.llm.bind_tools(openai_tools)

        # Initialize memory and workflow
        self._initialize_workflow()

    def _initialize_workflow(self):
        """Initialize the LangGraph workflow."""
        memory = MemorySaver()
        
        # Define agent node
        async def agent(state: ChatState, config) -> ChatState:
            response = await self.llm_with_tools.ainvoke(state["messages"], config)
            return {"messages": state["messages"] + [response], "all_files_opened": state["all_files_opened"]}
        
        # Define reflection node
        async def reflect(state: ChatState, config) -> ChatState:
            # Create a reflection prompt
            reflection_prompt = (
                "Based on the information gathered so far, please:\n"
                "1. Update your answer to my question\n"
                "2. Identify what information is still missing,\n"
                "3. Explain what files should be explored next and why. Keep it efficient and as less as possible because of context limit."
                "4. No tool usage in current message"
            )
            
            # Add reflection prompt
            reflection_request = HumanMessage(content=reflection_prompt)
            
            # Get reflection from LLM
            reflection = await self.llm_with_tools.ainvoke(state["messages"] + [reflection_request], config)
            
            # Add reflection to messages
            return {
                "messages": state["messages"] + [reflection_request, reflection, HumanMessage(content="Continue")], 
                "all_files_opened": state["all_files_opened"]
            }

        # Define tool execution node
        def execute_tools(state: ChatState) -> ChatState:
            messages = state["messages"]
            last_message = messages[-1]  # Last LLM response before placeholder
            if not last_message.tool_calls:
                return state
            all_files_opened = state["all_files_opened"]
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "open_files":
                    file_paths = tool_call["args"].get("file_paths", [])
                    logger.info(f"Opening files: {file_paths}")
                    result = self.tools[1].func(file_paths)
                    all_files_opened.extend(file_paths)
                elif tool_call["name"] == "get_file_structure":
                    result = self.tools[0].func()
                else:
                    result = f"Unknown tool: {tool_call['name']}"
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            return {"messages": messages, "all_files_opened": all_files_opened}

        # Define graph
        workflow = StateGraph(ChatState)
        workflow.add_node("agent", agent)
        workflow.add_node("tools", execute_tools)
        workflow.add_node("reflect", reflect)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            route_tools,
            {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "reflect")
        workflow.add_edge("reflect", "agent")
        self.app = workflow.compile(checkpointer=memory)

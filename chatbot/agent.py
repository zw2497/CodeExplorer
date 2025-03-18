import os
from typing import Dict, Any
import logging
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
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
        self.kb_path = os.path.join(codebase_path, 'knowledge_base.md')  # New
        self.checkpointer = MemorySaver()
        self.llm = ChatBedrock(
            model_id=BEDROCK_MODEL,
            region_name=session.region_name or AWS_REGION,
            model_kwargs={"temperature": 0.7, "max_tokens": 200000}
        )

        self.tools, openai_tools = create_tools(self.fs)
        self.llm_with_tools = self.llm.bind_tools(openai_tools)
        self._initialize_workflow()

    def _load_knowledge_base(self) -> str:        
        try:
            if os.path.exists(self.kb_path):
                with open(self.kb_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
        return ""

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
        if isinstance(last_message, ToolMessage) and hasattr(last_message, 'metadata') and last_message.metadata["tool_name"] == "open_files":
            response = await self.llm_with_tools.ainvoke(state["messages"] + [HumanMessage(content="\n\n Remind: [Create an immediate comprehensive summary for these files first, then continue the exploration or anwser directly if you are confident.]")], config)

        response = await self.llm_with_tools.ainvoke(state["messages"], config)

        if not response.tool_calls and generating_kb:
            return {"messages": [response, HumanMessage(content="\n\n Continue to self explore in order to generate the comprehensive knowledge base.")]}

        if isinstance(last_message, ToolMessage) and hasattr(last_message, 'metadata') and last_message.metadata["tool_name"] == "open_files":
            last_message.content = "..."
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

        Let's start build a comprehensive understanding of the codebase through iterative file analysis, summarization, and strategic file selection.
        if there is current knowledge base. You goal is to dive more deep to enhance and keep improving the knowledge base to gain more understanding of the code base.
        THe current knowledge base has provided examined file. You should prioritize file not examined before for new information.

        In each round:
        1. Request to open up to 5 files each time, ONLY open files that are exist in the file structure output
        2. Generate a structured key learnings and findings for opended files including Key classes, functions, relations with known knowledge and their responsibilities.
        3. Propose next files to explore based on new findings and use tools to open it.

        """
        return {
            "messages": [HumanMessage(content=kb_instruction)],
            "generating_kb": True,
            "kb_exploration_rounds": 0,
            "command": None
        }

    async def _generate_knowledge_base_node(self, state: ChatState, config) -> ChatState:
        existing_kb = state.get("knowledge_base", "")
        kb_prompt = HumanMessage(content=f"""
# Knowledge Base Update Protocol

## OBJECTIVE
Generate a comprehensive, structured technical document that captures the complete understanding of the codebase based on all explorations so far.

## INPUT SOURCES
- Previously established knowledge: {existing_kb}
- New insights from your recent file explorations

## DOCUMENT STRUCTURE
Create a complete technical document with the following sections:

### 1. Code Exploration Summary
- **Files Examined**: List all files you've analyzed (filenames without full paths)
- **Coverage Assessment**: Brief evaluation of what percentage of the codebase has been explored

### 2. System Architecture
- **Component Overview**: Major components and their relationships
- **Dependency Graph**: Key dependencies between components
- **Data Flow**: How information moves through the system

### 3. Key Components Reference
For each significant component:
- **Purpose**: Primary responsibility
- **API Surface**: Public methods/interfaces
- **Dependencies**: What it relies on
- **Usage Context**: Where and how it's used

### 4. Workflows and Processes
- **Main Execution Flows**: Step-by-step description of key processes
- **Control Flow Patterns**: How program execution is directed
- **Edge Cases**: Notable exception handling or special conditions

### 5. Technical Patterns and Implementation Details
- **Design Patterns**: Identified software patterns and their implementation
- **Architecture Decisions**: Notable architectural choices and their implications
- **Performance Considerations**: Any observed optimization strategies

### 6. Integration Points
- **External APIs**: Interfaces to external systems
- **Extension Points**: How the system can be extended
- **Configuration Options**: How the system can be configured

### 7. Knowledge Gaps
- **Unexplored Areas**: Components or aspects not yet fully understood
- **Open Questions**: Uncertainties that remain after exploration

## INTEGRATION GUIDELINES
1. **Reconcile Information**: When new findings contradict existing knowledge, evaluate which is more accurate based on direct code evidence
2. **Preserve Valid Insights**: Retain all accurate information from the existing knowledge base
3. **Expand, Don't Replace**: Add detail to existing sections rather than overwriting them when possible
4. **Evidence-Based Updates**: Only include information directly observed in the code

## OUTPUT FORMAT
Generate the COMPLETE knowledge base document incorporating both existing and new knowledge. Do not merely describe changes or additions - provide the entire updated knowledge base as a cohesive markdown document.
        """)
        
        kb_response = await self.llm_with_tools.ainvoke(
            state["messages"] + [kb_prompt],
            config
        )

        # Process content to ensure it's always a string
        if isinstance(kb_response.content, list):
            kb_content = "".join(
                chunk["text"] for chunk in kb_response.content 
                if isinstance(chunk, dict) and chunk.get("type") == "text"
            )
        else:
            kb_content = str(kb_response.content)


        # Persist updated KB  # New
        try:
            with open(self.kb_path, 'w', encoding='utf-8') as f:
                f.write(kb_content)
            logger.info(f"Updated knowledge base persisted to {self.kb_path}")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")


        return {
            "messages": [kb_prompt, kb_response],
            "knowledge_base": kb_content,
            "generating_kb": False,
            "command": None
        }

    async def _summarize_conversation(self, state: ChatState, config) -> ChatState:
        # Keep last 4 messages (adjust number as needed)
        keep_messages = 1
        messages_to_remove = state["messages"][:-keep_messages]

        if isinstance(messages_to_remove[0], SystemMessage):
            messages_to_remove.pop(0)
        
        # Generate summary including existing summary
        summary_prompt = f"""
        Current conversation summary: {state.get('summary', '')}
        Update this summary with key technical details from these new messages:
        {messages_to_remove}
        less than 500 words
        """
        
        response = await self.llm_with_tools.ainvoke(
            [HumanMessage(content=summary_prompt)],
            config
        )
        
        # Create RemoveMessage for old messages
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_remove]
        
        return {
            "summary": response.content,
            "messages": delete_messages  # This will remove old messages from state
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
        
        if len(state["messages"]) > 15:
            return "summarizer"
        
        return END
    
    def _route_after_tools(self, state: ChatState):
        if state.get("generating_kb", False) and state.get("kb_exploration_rounds", 0) > 8:
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
        workflow.add_node("summarizer", self._summarize_conversation)
        
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
                "summarizer": "summarizer",
                "agent": "agent",
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
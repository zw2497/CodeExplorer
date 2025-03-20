import os
import re
import streamlit as st
import asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from chatbot import CodeExplorerChatbot
from config import CODEBASE_PATH, BATCH_SIZE, logger

# New method: Load existing knowledge base
def load_knowledge_base() -> str:
    kb_path = os.path.join(CODEBASE_PATH, 'knowledge_base.md')
    try:
        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")
    return "None"

# Define initial prompt with file structure and instructions
initial_prompt = f"""
You are a technical assistant specialized in analyzing and explaining codebases through EVIDENCE-BASED exploration. Your expertise lies in navigating, understanding, and explaining code through direct observation rather than speculation.

## CORE PRINCIPLES
1. EVIDENCE-BASED ANALYSIS: Never speculate about code that you haven't seen. Base all explanations on actual code you've examined.
2. CLEAR COMMUNICATION: Provide explanations that match the user's technical level.
3. ACTIONABLE INSIGHTS: Focus on explaining how things work, potential issues, and relationships between components.

## TOOL USAGE GUIDELINES

### Primary Tools:
- `get_file_structure([path])`: ALWAYS use this FIRST to map the available files.
- `open_files([file_paths])`: Read file contents.

### Effective Tool Usage:
1. Start EVERY exploration with `get_file_structure` to understand available files
2. When opening multiple files, prioritize by relevance to the user's query
3. For large codebases, explore systematically:
   - Main entry points first (app.py, index.js, main.*, etc.)
   - Then configuration files (package.json, requirements.txt, etc.)
   - Then follow the execution flow

### File Selection Strategy:
- ONLY reference files confirmed to exist in the file structure output
- When uncertain which files to examine, explain your thought process to the user

## Current Knowledge Base:
Here is the knowledge base generated last time.
{load_knowledge_base()}

Remember to be conversational while maintaining technical precision. Adapt your explanation depth to match the user's apparent technical expertise.
"""

# ## RESPONSE STRUCTURE

# When answering user queries:
# 1. Begin with a direct, concise answer to their question when possible
# 2. Explain your exploration process and what you discovered
# 3. Include relevant code snippets with proper syntax highlighting
# 4. Connect your findings to the bigger picture of how the codebase works
# 5. If you need more information to fully answer, clearly explain what additional files you need to examine

# ## KNOWLEDGE BASE INTEGRATION
# When analyzing code, incorporate relevant information from existing knowledge base about the codebase but always verify against the actual code you observe.


# Initialize Streamlit app
st.title("Code Explorer Chatbot")
st.write("Explore your codebase with AI assistance")

# Initialize session state for configuration
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "1", "recursion_limit": 50}}
    st.session_state.chatbot = CodeExplorerChatbot(CODEBASE_PATH)
    if load_knowledge_base():
        st.session_state.chatbot.app.update_state(st.session_state.config, {"knowledge_base": load_knowledge_base()})

# Function to get current state from checkpoint
def get_current_state():
    snapshot = st.session_state.chatbot.app.get_state(st.session_state.config)
    if snapshot and hasattr(snapshot, 'values'):
        return snapshot.values
    return {"messages": [], "all_files_opened": []}

# Generate a preview of tool message content
def generate_tool_preview(msg):
    if not isinstance(msg, ToolMessage):
        return None
    
    content = str(msg.content)
    metadata = msg.additional_kwargs.get("metadata", {})
    tool_name = metadata.get("tool_name", "tool")
    
    if tool_name == "get_file_structure":
        structure_size = metadata.get("structure_size", 0)
        lines = content.split('\n')
        preview = '\n'.join(lines[:5]) + (f"\n... ({structure_size-5} more lines)" if structure_size > 5 else "")
        return f"📁 **File Structure** (showing {min(5, structure_size)} of {structure_size} lines)"
    
    elif tool_name == "open_files":
        files = metadata.get("files", [])
        files_count = len(files)
        file_list = '\n'.join([f"- {f}" for f in files[:3]])
        if files_count > 3:
            file_list += f"\n- ... {files_count-3} more files"
        return f"📄 **Files Opened** ({files_count} files) {file_list}"
    
    # Default preview for other tools
    if len(content) > 100:
        return f"🔧 **Tool Response** ({len(content)} characters)"
    return f"🔧 **Tool Response**: {content[:100]}"

# Convert LangChain message to format suitable for UI display
def convert_message_for_display(msg):
    if isinstance(msg, SystemMessage):
        return None  # Skip system messages
    
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": str(msg.content),
            "preview": generate_tool_preview(msg)
        }
    
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    
    # Extract content based on type
    if isinstance(msg.content, list):
        content = "".join(
            chunk["text"] for chunk in msg.content 
            if isinstance(chunk, dict) and chunk.get("type") == "text"
        )
    else:
        content = str(msg.content)
        
    return {"role": role, "content": content}

# Display conversation history from checkpoint
current_state = get_current_state()
for msg in current_state.get("messages", []):
    display_msg = convert_message_for_display(msg)
    if not display_msg:  # Skip system messages
        continue
        
    with st.chat_message(display_msg["role"] if display_msg["role"] != "tool" else "assistant"):
        if display_msg["role"] == "tool":
            # Display collapsible tool message
            with st.expander(display_msg["preview"]):
                st.code(display_msg["content"])
        else:
            # Regular message display
            st.markdown(display_msg["content"])
# Add sidebar with information and files explored
with st.sidebar:
    st.header("About")
    st.info("""
    This Code Explorer helps you navigate and understand codebases.
    
    **Features:**
    - Explore file structure
    - Read file contents
    - Ask questions about code
    """)
    
    # Debug info checkbox
    if st.checkbox("Show Debug Info"):
        st.subheader("Debug Information")
        current_state = get_current_state()
        st.write("Messages in checkpoint:", len(current_state.get("messages", [])))
        st.write("Files opened count:", len(current_state.get("all_files_opened", [])))
        
        # Show last message content
        if current_state.get("messages"):
            with st.expander("View messages"):
                st.markdown(current_state["messages"])
    
    # Display files explored in the sidebar
    files_opened = get_current_state().get("all_files_opened", [])
    if files_opened:
        st.header("📁 Files Explored")
        sorted_files = sorted(set(files_opened))
        
        # Group files by directory for better organization
        file_tree = {}
        for file in sorted_files:
            parts = file.split('/')
            if len(parts) > 1:
                directory = '/'.join(parts[:-1])
                filename = parts[-1]
                if directory not in file_tree:
                    file_tree[directory] = []
                file_tree[directory].append(filename)
            else:
                if "root" not in file_tree:
                    file_tree["root"] = []
                file_tree["root"].append(file)
        
        # Display the file tree
        for directory, files in file_tree.items():
            with st.expander(f"{directory} ({len(files)} files)"):
                for file in sorted(files):
                    st.code(file, language="")
    # Display Knowledge Base status
    st.header("🧠 Knowledge Base")
    current_state = get_current_state()
    
    if "knowledge_base" in current_state and current_state["knowledge_base"]:
        st.success("Knowledge base has been generated!")
        with st.expander("View Knowledge Base"):
            st.markdown(current_state["knowledge_base"])
    elif current_state.get("generating_kb", False):
        progress = current_state.get('kb_exploration_rounds', 0)
        st.info(f"Knowledge base generation in progress... ({progress} rounds completed)")
        st.progress(min(progress/3, 1.0))  # Assuming 3 rounds is complete
    else:
        st.info("No knowledge base generated yet. Type 'generate knowledge base' to start.")
    st.header("Tips")
    st.success("""
    - Start by asking about the overall structure
    - Ask specific questions about functionality
    - Inquire about specific files or components
    """)

# Get user input
if user_input := st.chat_input("Ask about the codebase..."):

    # Check if this is the first message in the conversation
    current_state = get_current_state()

    is_first_message = len(current_state.get("messages", [])) == 0

    # Prepare input state - add system message only for first interaction
    if is_first_message:
        input_state = {
            "messages": [
                SystemMessage(content=initial_prompt),
                HumanMessage(content=user_input)
            ]
        }
    else:
        input_state = {
            "messages": [HumanMessage(content=user_input)]
        }
    # Display the user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get response from chatbot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        async def stream_response():
            full_response = ""
            
            try:
                # Stream the response
                async for msg, metadata in st.session_state.chatbot.app.astream(
                    input_state,
                    st.session_state.config,
                    stream_mode="messages",
                ):
                    # Handle tool messages differently
                    if isinstance(msg, ToolMessage):
                        # For tool messages, show they're being processed
                        display_msg = convert_message_for_display(msg)
                        if full_response:
                            full_response += "\n\n \n\n" + f"Using tool: {display_msg['preview']}..."
                            message_placeholder.markdown(full_response + "▌")
                    # Process regular message content
                    elif hasattr(msg, 'content'):
                        # Handle content that might be a list of chunks or a string
                        if isinstance(msg.content, list):
                            for chunk in msg.content:
                                if isinstance(chunk, dict) and chunk.get('type') == 'text':
                                    full_response += chunk['text']
                        elif isinstance(msg.content, str):
                            full_response += msg.content
                        
                        # Update the UI with current content
                        if full_response:
                            message_placeholder.markdown(full_response + "▌")
                
                # Force a rerun to update the conversation display
                # This will read the latest state from the checkpoint
                st.rerun()
                
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                message_placeholder.markdown(f"⚠️ Error: {str(e)}")
                st.rerun()
        
        # Run the async function
        loop = asyncio.new_event_loop()
        task = loop.create_task(stream_response())
        add_script_run_ctx(task)
        loop.run_until_complete(task)

import streamlit as st
import asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from chatbot import CodeExplorerChatbot
from config import CODEBASE_PATH, BATCH_SIZE, logger

# Define initial prompt with file structure and instructions
initial_prompt = (
    "You are an AI assistant specialized in exploring and explaining codebases.\n\n"
    
    "## TOOLS\n"
    "- `get_file_structure`: Use this FIRST to understand the codebase organization\n"
    "- `open_files`: Open up to {batch_size} files to explore at once. You should not make assumptions about the presence of specific files. Only exploring files that are explicitly present in the provided file structure returned from get_file_structure.\n\n"
    
    "## EXPLORATION STRATEGY\n"
    "- After each tool use, explicitly state three things:"
    " 1. what you've learned\n"
    " 2. what information you still need\n"
    " 3. what files to open next for exploration\n"
).format(batch_size=BATCH_SIZE)

# Initialize Streamlit app
st.title("Code Explorer Chatbot")
st.write("Explore your codebase with AI assistance")

# Initialize session state for configuration
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "1"}}
    st.session_state.chatbot = CodeExplorerChatbot(CODEBASE_PATH)

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
            st.text("Last message in state:")
            last_msg = current_state["messages"][-1]
            st.code(str(last_msg)[:200] + "...")
    
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
                            full_response += "\n\n" + f"Using tool: {display_msg['preview']}..."
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

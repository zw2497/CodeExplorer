import streamlit as st
import asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from chatbot import CodeExplorerChatbot
from config import CODEBASE_PATH, BATCH_SIZE, logger

# Define initial prompt with file structure and instructions
initial_prompt = (
    "You're a helpful AI chatbot to help user exploring a new codebase.\n"
    "You should be able to anwser general conversation. If necessary, you can use following tools."
    "After each tool use, reflect what to explore next to anwser user's query. if you think it's confident to answer, give final result without tool use."
    "Tool available:\n"
    "1. Use 'get_file_structure' to understand the codebase file structure.\n"
    "2. Use 'open_files' to inspect up to {batch_size} files each time from the file structure.\n"
    "**Important formatting instructions:**\n"
    "- Format your responses using Markdown syntax\n"
    "- Use code blocks with language specification for code: ```python\n"
    "- Use bold for important concepts: **important**\n"
    "- Use headers for sections: ## Section Title\n"
    "- Use lists and tables when appropriate\n"
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

# Convert LangChain message to format suitable for UI display
def convert_message_for_display(msg):
    if isinstance(msg, SystemMessage):
        return None  # Skip system messages
    
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
    print(display_msg)
    if display_msg:  # Skip system messages
        with st.chat_message(display_msg["role"]):
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
    
    # # Add option to reset conversation
    # if st.button("Reset Conversation"):
    #     initial_state = {
    #         "messages": [SystemMessage(content=initial_prompt)],
    #         "all_files_opened": []
    #     }
    #     st.session_state.chatbot.app.update_state(initial_state, st.session_state.config)
    #     st.rerun()

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
            # Prepare input for the chatbot
            
            try:
                # Stream the response
                async for msg, metadata in st.session_state.chatbot.app.astream(
                    input_state,
                    st.session_state.config,
                    stream_mode="messages",
                ):
                    # Process message content based on its type
                    if hasattr(msg, 'content'):
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
                
                # # Final update without cursor
                # message_placeholder.markdown(full_response)
                
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

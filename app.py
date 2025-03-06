import streamlit as st
import asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx
from langchain_core.messages import HumanMessage
from chatbot import CodeExplorerChatbot
from config import CODEBASE_PATH, BATCH_SIZE, logger

# Initialize the chatbot
chatbot = CodeExplorerChatbot(CODEBASE_PATH)

# Define initial prompt with file structure and instructions
initial_prompt = (
    "You're a helpful AI chatbot to help user exploring a new codebase.\n"
    "You should be able to anwser general conversation. If necessary, you can use following tools."
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

# Initialize session state for conversation history
if "messages" not in st.session_state:
    initial_system_msg = HumanMessage(content=initial_prompt)
    st.session_state.messages = [{"role": "system", "content": initial_prompt}]
    st.session_state.all_files_opened = []
    st.session_state.input_state = {
        "messages": [initial_system_msg],
        "all_files_opened": []
    }
    st.session_state.config = {"configurable": {"thread_id": "1"}}



# Display conversation history
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
    # Add to your Streamlit app
    if st.checkbox("Show Debug Info"):
        st.subheader("Debug Information")
        st.write("Messages in input_state:", len(st.session_state.input_state["messages"]))
        st.write("Messages in UI history:", len(st.session_state.messages))
        st.write("Files opened count:", len(st.session_state.all_files_opened))
        
        # Show last message content
        if st.session_state.input_state["messages"]:
            st.text("Last message in state:")
            st.code(str(st.session_state.input_state["messages"][-1])[:200] + "...")
    # Display files explored in the sidebar
    if st.session_state.all_files_opened:
        st.header("📁 Files Explored")
        sorted_files = sorted(set(st.session_state.all_files_opened))
        
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
    
    # # Add a debug section to help troubleshoot
    # if st.checkbox("Show Debug Info"):
    #     st.subheader("Debug Information")
    #     st.write("Messages in input_state:", len(st.session_state.input_state["messages"]))
    #     st.write("Messages in UI history:", len(st.session_state.messages))
        
    #     # Add a button to reset the conversation
    #     if st.button("Reset Conversation"):
    #         st.session_state.messages = [{"role": "system", "content": initial_prompt}]
    #         st.session_state.all_files_opened = []
    #         st.session_state.input_state = {
    #             "messages": [HumanMessage(content=initial_prompt)],
    #             "all_files_opened": []
    #         }
    #         st.rerun()


# Get user input
if user_input := st.chat_input("Ask about the codebase..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add to LangGraph input state
    st.session_state.input_state["messages"].append(HumanMessage(content=user_input))
    
    # Get response from chatbot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        async def stream_response():
            full_response = ""
            updated_state = None
            
            # Create a copy of the current state to send to LangGraph
            current_state = {
                "messages": st.session_state.input_state["messages"].copy(),
                "all_files_opened": st.session_state.all_files_opened.copy()
            }
            
            async for msg, metadata in chatbot.app.astream(
                current_state,
                st.session_state.config,
                stream_mode="messages",
            ):
                # Save the updated state from metadata
                if metadata and "state" in metadata:
                    updated_state = metadata["state"]
                
                if msg.content:
                    # Handle case where content is a list of chunks
                    if isinstance(msg.content, list):
                        for chunk in msg.content:
                            if isinstance(chunk, dict) and chunk.get('type') == 'text':
                                full_response += chunk['text']
                                message_placeholder.markdown(full_response + "▌")
                    # Handle string content
                    elif isinstance(msg.content, str):
                        full_response += msg.content
                        message_placeholder.markdown(full_response + "▌")
            
            # Final update without the cursor
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # CRITICAL: Update the session state with the complete state from LangGraph
            if updated_state:
                # Replace the entire input state with the latest from LangGraph
                st.session_state.input_state = updated_state
                # Also update the all_files_opened list for sidebar display
                st.session_state.all_files_opened = updated_state.get("all_files_opened", [])
            else:
                logger.warning("No updated state received from LangGraph!")

        
        # Run the async function
        loop = asyncio.new_event_loop()
        task = loop.create_task(stream_response())
        add_script_run_ctx(task)
        loop.run_until_complete(task)

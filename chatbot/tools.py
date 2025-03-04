from langchain.tools import Tool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from typing import List
from .models import OpenFilesSchema

def create_tools(filesystem):
    """Create and return the tools used by the chatbot."""
    
    def open_files(file_paths: List[str]) -> str:
        """Open and return the contents of the specified files (up to 30000 chars each)."""
        return filesystem.read_files(file_paths, max_chars=30000)
    
    tools = [
        Tool(
            name="get_file_structure",
            func=filesystem.get_file_structure,
            description="Retrieve the file structure of the codebase."
        ),
        StructuredTool.from_function(
            func=open_files,
            name="open_files",
            description="Open and retrieve contents of files from the codebase.",
            args_schema=OpenFilesSchema
        )
    ]
    
    return tools, [convert_to_openai_tool(t) for t in tools]

import os
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

class FileSystem:
    def __init__(self, codebase_path: str):
        self.path = Path(codebase_path)
        self.files = self._list_files()

    def _list_files(self) -> List[str]:
        """List all text files in the codebase."""
        text_extensions = {'.java', '.py', '.js', '.cpp', '.h', '.yml', '.yaml', '.properties'}
        return [str(f.relative_to(self.path)) for f in self.path.rglob("*") 
                if f.is_file() and f.suffix.lower() in text_extensions]

    def get_file_structure(self) -> str:
        """Generate a nested file structure with total size, ignoring test files."""
        # Filter out test files (case insensitive)
        non_test_files = [file for file in self.files if not Path(file).stem.lower().endswith('test')]
        
        # Calculate total size of non-test files
        total_size = sum((self.path / file).stat().st_size for file in non_test_files) / 1024
        structure = f"{self.path.name} ({total_size:.1f}KB, {len(non_test_files)} files)\n"
        
        packages = {}
        for file in non_test_files:
            parts = file.split(os.sep)
            pkg = "/".join(parts[:-1]) if len(parts) > 1 else ""
            if pkg not in packages:
                packages[pkg] = []
            packages[pkg].append(parts[-1])
        
        for pkg, files in sorted(packages.items()):
            if pkg:
                structure += f"├── {pkg} ({len(files)} files)\n"
            for file in sorted(files):
                structure += f"│   ├── {file}\n" if pkg else f"├── {file}\n"
        
        return structure.strip()


    def read_files(self, file_paths: List[str], max_chars: int = 30000) -> str:
        """Read content of selected files, trimmed to max_chars, with path cleaning."""
        contents = {}
        cleaned_paths = []

        if isinstance(file_paths, str):
            fp_cleaned = file_paths.replace('\n', '').strip()
            if fp_cleaned.startswith('[') and fp_cleaned.endswith(']'):
                try:
                    import ast
                    cleaned_paths = ast.literal_eval(fp_cleaned)
                    logger.debug("Converted stringified files list: %s", cleaned_paths)
                except (ValueError, SyntaxError) as e:
                    logger.error("Failed to parse stringified file list: %s", str(e))
                    return "No valid file contents retrieved."
            else:
                cleaned_paths = [fp_cleaned]
        elif isinstance(file_paths, list):
            cleaned_paths = file_paths

        for fp in cleaned_paths:
            fp_clean = fp.strip().strip("'\"[]").strip()
            fp_clean = ''.join(c for c in fp_clean if not c.isdigit()).strip().lstrip('. ').strip()
            if not fp_clean or fp_clean == '/':
                logger.warning(f"Skipping invalid file path: {fp}")
                continue
            full_path = self.path / fp_clean
            if full_path.exists() and full_path.is_file():
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        contents[fp_clean] = content[:max_chars] + ("..." if len(content) > max_chars else "")
                except Exception as e:
                    logger.error(f"Error reading file {fp_clean}: {str(e)}")
            else:
                logger.warning(f"File not found or not a file: {fp_clean}")
        return "\n\n".join([f"{fp}:\n{cont}" for fp, cont in contents.items()]) if contents else "No valid file contents retrieved."

# Code Explorer Chatbot

An AI-powered chatbot that helps developers explore and understand codebases quickly.

## Features

- Browse file structure of any codebase
- Read file contents with AI assistance
- Ask questions about code functionality
- Explore code relationships and dependencies

## Setup
1. install pyenv
For zsh:
```
brew install pyenv xz
grep 'pyenv init' ~/.zshrc || echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
source ~/.zshrc
```

If you receive a warning `pyenv init -` no longer sets PATH.' and the installation is not working, you may instead need to manually edit the ~/.bash_profile or ~/.zshrc file and replace 'eval "$(pyenv init -)"' with
```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
```
The list of extra brew install packages comes from the python-build wiki.

Now you can install whatever version of Python you want. For example to install Python 3.8.11 and use it globally run:
```
pyenv install 3.8.11
pyenv global 3.8.11
```
If you are experiencing global python version not changed, and still showing `Python 2.7.18` when running python -V, try go to the previous step and make sure eval "$(pyenv init --path)" is in your ~/.bash_profile or ~/.zshrc file. 

To see all available versions run:
```
pyenv install --list
```

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/code-explorer-chatbot.git
   cd code-explorer-chatbot
   streamlit run app.py
   ```

# aelf-repo-chat

A conversational AI agent that allows users to query and retrieve information from any aelf GitHub repository. This tool ingests repository content into a vector database and uses OpenAI's GPT-4 to generate accurate, context-aware responses to questions about the codebase.

## Features

- Recursively fetches and processes GitHub repository content
- Excludes binary and media files to focus on code and documentation
- Stores content in a ChromaDB vector database for semantic search
- Provides natural language interface to query repository information
- Generates concise, accurate responses using GPT-4

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- GitHub access (for repository fetching)
- OpenAI API key (for GPT-4 access)

### Installation

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run the application
python3 main.py
```

## How It Works

1. **Ingest Mode**: Provide a GitHub repository URL to fetch and process its contents
2. **Query Mode**: Ask questions about the repository in natural language
3. **Response**: Get detailed answers based on the repository's content

## Configuration

The application uses Azure OpenAI by default. You can configure API keys and endpoints through environment variables in the `.env` file:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`

### Setting up the .env file

Create a `.env` file in the root directory with the following variables:

```
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_VERSION=your_api_version_here
```

Note: The `.env` file is included in `.gitignore` to prevent sensitive information from being committed to the repository.

# aelf-repo-chat

A conversational AI agent that allows users to query and retrieve information from any aelf GitHub repository. This tool ingests repository content into a vector database and uses OpenAI's GPT-4 to generate accurate, context-aware responses to questions about the codebase.

## Features

- Recursively fetches and processes GitHub repository content
- Excludes binary and media files to focus on code and documentation
- Stores content in a ChromaDB vector database for semantic search
- Provides natural language interface to query repository information through REST API
- Generates concise, accurate responses using GPT-4
- Searches across multiple repositories simultaneously

## Architecture

The application is split into two main components:

1. **Indexing Script**: Processes GitHub repositories and stores their content in a vector database
2. **API Server**: Provides a REST endpoint for querying the indexed repositories

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

### Configuration

1. Configure repositories in `config.json`:

```json
{
  "repositories": [
    {
      "url": "https://github.com/AElfProject/aelf",
      "name": "aelf-main",
      "branch": "dev"
    }
  ]
}
```

2. Set up a GitHub Personal Access Token:

   - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Click "Generate new token" → "Generate new token (classic)"
   - Give it a name (e.g., "aelf-repo-chat")
   - Select the `repo` scope (to access public repositories)
   - Click "Generate token" and copy the token value
   - Add the token to your `.env` file as `GITHUB_TOKEN=your_token_here`

   **Note**: This step is **required** to avoid GitHub API rate limits, especially when indexing large repositories.

3. Set up environment variables in `.env` file (see below)

### Usage

#### Step 1: Index repositories

```bash
# Run the indexing script to process repositories defined in config.json
python3 index_repos.py
```

#### Step 2: Start the API server

```bash
# Start the API server
python3 main.py
```

### API Endpoints

#### Chat Endpoint

```
POST /api/chat
```

Request body:
```json
{
  "query": "How does the AELF consensus mechanism work?",
  "top_k": 10
}
```

Response:
```json
{
  "answer": "Detailed answer about AELF consensus...",
  "source_documents": [
    {
      "content": "Excerpt from the document...",
      "source": "path/to/file.md",
      "collection": "aelf-main_collection"
    }
  ]
}
```

## Environment Variables

The application uses environment variables for configuration:

```
# Azure OpenAI API Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_VERSION=your_api_version_here

# GitHub API Token (optional)
GITHUB_TOKEN=your_github_token_here

# Server Configuration
HOST=0.0.0.0
PORT=5000
```

Note: The `.env` file is included in `.gitignore` to prevent sensitive information from being committed to the repository.

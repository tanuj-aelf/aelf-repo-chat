# aelf-repo-chat

A conversational AI agent that allows users to query and retrieve information from any aelf GitHub repository. This tool ingests repository content into a vector database and uses advanced language models (like GPT-4, Claude, or Gemini) to generate accurate, context-aware responses to questions about the codebase.

## Features

- Recursively fetches and processes GitHub repository content
- Excludes binary and media files to focus on code and documentation
- Stores content in a ChromaDB vector database for semantic search
- Provides natural language interface to query repository information through REST API
- Generates concise, accurate responses using AI models from multiple providers
- Searches across multiple repositories simultaneously
- Integrates with Lark (Feishu) bot for chat-based interaction
- Supports multiple AI providers (Azure OpenAI, OpenAI, Google, Anthropic, and OpenRouter)

## Architecture

The application is split into three main components:

1. **Indexing Script**: Processes GitHub repositories and stores their content in a vector database
2. **API Server**: Provides a REST endpoint for querying the indexed repositories
3. **Lark Bot Handler**: Processes messages from Lark and integrates with the API server

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- GitHub access (for repository fetching)
- API key for at least one supported AI provider:
  - Azure OpenAI
  - OpenAI
  - Google AI (Gemini)
  - Anthropic (Claude)
  - OpenRouter

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
python3 -m services.index_repos
```

#### Step 2: Start the API server

```bash
# Start the API server
python3 -m services.main
```

You can also generate or update repository summaries without re-indexing:

```bash
# Generate/update repository summaries
python3 -m services.generate_summaries

# View generated summaries
python3 -m services.generate_summaries --print
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
# Model Provider Configuration
MODEL_PROVIDER=azure_openai  # Options: azure_openai, openai, anthropic, google, openrouter

# Azure OpenAI API Configuration (if using azure_openai)
AZURE_OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_MODEL=your_deployment_name

# OpenAI Configuration (if using openai)
OPENAI_API_KEY=your_openai_api_key  
OPENAI_MODEL=gpt-4o  # Or other OpenAI models

# Google Gemini Configuration (if using google)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-1.5-pro  # Or other Gemini models

# Anthropic Configuration (if using anthropic)
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-3-sonnet-20240229  # Or other Claude models

# OpenRouter Configuration (if using openrouter)
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=deepseek/deepseek-chat:free  # Or another model available on OpenRouter
OPENROUTER_SITE_URL=your_site_url  # Optional. Site URL for rankings on openrouter.ai
OPENROUTER_SITE_NAME=your_site_name  # Optional. Site name for rankings on openrouter.ai

# GitHub API Token
GITHUB_TOKEN=your_github_token
GITHUB_DEFAULT_OWNER=AElfProject
GITHUB_DEFAULT_BRANCH=main

# Server Configuration
HOST=0.0.0.0
PORT=5000

# Lark Bot Configuration
LARK_APP_ID=your_app_id
LARK_APP_SECRET=your_app_secret
LARK_VERIFICATION_TOKEN=your_verification_token
LARK_ENCRYPT_KEY=your_encrypt_key
LARK_HOST=https://open.feishu.cn
```

Note: The `.env` file is included in `.gitignore` to prevent sensitive information from being committed to the repository.

## Model Configuration

The application uses a flexible model configuration system that supports multiple AI providers:

1. **Azure OpenAI** - Microsoft's hosted version of OpenAI models with additional enterprise features
2. **OpenAI** - Direct integration with OpenAI's API (GPT-4, GPT-3.5, etc.)
3. **Google** - Integration with Google's Gemini models
4. **Anthropic** - Integration with Anthropic's Claude models
5. **OpenRouter** - Access to various open-source and commercial models through a unified API

To switch between providers, simply change the `MODEL_PROVIDER` environment variable in your `.env` file and ensure the corresponding API keys and model configurations are set.

## Lark Bot Integration

The application includes integration with Lark (Feishu) bots, allowing users to interact with the repository knowledge base directly through Lark chat.

### Setting Up the Lark Bot

1. Create a Custom Bot in the [Lark Developer Console](https://open.feishu.cn/app)
2. Configure the following permissions for your bot:
   - `im:message.receive_v1` - For receiving messages
   - `im:message:send_as_bot` - For sending messages

3. Set up the event subscription URL to point to your server's endpoint:
   ```
   https://your-server-domain/lark/event
   ```

4. Get your Lark bot credentials and update your `.env` file:
   - `LARK_APP_ID` - From the bot's app credentials
   - `LARK_APP_SECRET` - From the bot's app credentials
   - `LARK_VERIFICATION_TOKEN` - Token for verifying event subscriptions
   - `LARK_ENCRYPT_KEY` - Key for encrypting/decrypting messages (if encryption is enabled)

5. Start your server, and the bot will be able to receive messages and respond with information from your aelf repositories.

### Using the Lark Bot

Once set up, users can interact with the bot by:

1. Finding the bot in Lark and starting a conversation
2. Asking questions about aelf repositories
3. The bot will process the query, search across indexed repositories, and respond with relevant information

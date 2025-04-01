import os
import json
import logging
import time
import chromadb
from chromadb.utils import embedding_functions
from tiktoken import encoding_for_model
from dotenv import load_dotenv
from flask import Flask, request, jsonify
# Import Lark handler
from lark_handler import init_lark_bot
# Import repository summarizer
from repo_summarizer import get_formatted_repo_summaries
# Import model configuration
from model_config import generate_completion, start_request_flow, end_request_flow, suppress_duplicate_logs
import re

# Determine the project root directory (one level up from services/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ensure logs directory exists
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "server.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("server")

# Load environment variables from .env file
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=os.path.join(PROJECT_ROOT, "chroma_db"))
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Initialize Flask app
app = Flask(__name__)

# Dictionary to store conversation history for each session
conversation_histories = {}

# Maximum conversation history length to maintain (increased from 10 to 5*2)
MAX_CONVERSATION_HISTORY = 10  # 5 exchanges = 10 messages (user+assistant)

# Conversation history cleanup interval (in seconds)
HISTORY_CLEANUP_INTERVAL = 86400  # 24 hours (was 3600 = 1 hour)

# Last cleanup timestamp
last_cleanup_time = time.time()

def get_active_collections():
    """
    Get a list of collection names for active repositories from config.json
    """
    try:
        config_path = os.path.join(PROJECT_ROOT, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        repositories = config.get("repositories", [])
        active_collections = [f"{repo['name']}_collection" for repo in repositories]
        
        logger.info(f"Active collections from config: {active_collections}")
        return active_collections
    except Exception as e:
        logger.error(f"Error reading config.json: {str(e)}")
        # Fall back to getting all collections if config can't be read
        return get_all_collections()

def get_all_collections():
    """
    Get a list of all collection names in the database.
    """
    collections = chroma_client.list_collections()
    collection_names = [collection.name for collection in collections]
    logger.info(f"All available collections in DB: {collection_names}")
    return collection_names

def load_structured_content(repo_name):
    """
    Load the structured content for a repository.
    """
    structured_content_path = os.path.join(
        PROJECT_ROOT, "structured_content", f"{repo_name}_structured_content.json"
    )
    
    if os.path.exists(structured_content_path):
        try:
            with open(structured_content_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading structured content for {repo_name}: {str(e)}")
    
    return None

def determine_relevant_repos(query, request_id=None):
    """
    Use an LLM to determine which repositories are most relevant to the query.
    Returns a list of repository names based on repository summaries.
    """
    try:
        logger.info(f"Determining relevant repositories for query: {query}")
        
        summary_path = os.path.join(PROJECT_ROOT, "repo_summaries", "all_repo_summaries.json")
        
        try:
            with open(summary_path, "r") as f:
                repo_summaries = json.load(f)
            
            repo_info = []
            for repo_name, repo_data in repo_summaries.items():
                modules = repo_data.get("modules", [])
                description = repo_data.get("metadata", {}).get("description", "")
                
                repo_info.append({
                    "name": repo_name,
                    "description": description,
                    "modules": modules
                })
            
            repo_info = sorted(repo_info, key=lambda x: x["name"])
            
            repo_descriptions = "\n\n".join([
                f"Repository: {repo['name']}\nDescription: {repo['description']}\nModules: {', '.join(repo['modules'])}"
                for repo in repo_info
            ])
            
            prompt = f"""Based on the following query and repository descriptions, determine which repositories are most relevant for finding information to answer the query.

Query: {query}

Available repositories:
{repo_descriptions}

Instructions:
1. Analyze the query and understand its information needs
2. Review each repository's purpose, description, and modules
3. Select ONLY repositories that are likely to contain information relevant to the query
4. Return ONLY a JSON array of repository names in the format: ["repository_name"]
5. If unsure, select all repositories
6. Do not include any explanations, only return the JSON array

JSON array of relevant repository names:"""

            sub_request_id = f"determine_repos_{hash(query)}"
            
            response = generate_completion(
                prompt,
                system_message="You are a repository selection assistant. Your task is to determine which repositories are most relevant to a given query. Return only a valid JSON array of repository names.",
                config={
                    "max_tokens": 200,
                    "temperature": 0.1 
                },
                request_id=sub_request_id,
                parent_request_id=request_id
            )
            
            logger.info(f"Raw LLM response for repo selection: '{response}'")
            
            try:
                response_cleaned = response.strip()
                
                if response_cleaned.startswith("```") and response_cleaned.endswith("```"):
                    json_str = response_cleaned.strip("```").strip()
                    if json_str.startswith("json"):
                        json_str = json_str[4:].strip()
                    response_cleaned = json_str
                elif not (response_cleaned.startswith("[") and response_cleaned.endswith("]")):
                    start_idx = response_cleaned.find("[")
                    end_idx = response_cleaned.rfind("]")
                    if start_idx >= 0 and end_idx >= 0:
                        json_str = response_cleaned[start_idx:end_idx+1]
                        response_cleaned = json_str
                    else:
                        logger.error(f"Could not find JSON array in response: '{response_cleaned}'")
                        raise ValueError("Could not find JSON array in response")
                
                relevant_repos = json.loads(response_cleaned)
                logger.info(f"Parsed repositories: {relevant_repos}")
                
                # Get list of active repositories from config
                config_path = os.path.join(PROJECT_ROOT, "config.json")
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                available_repos = [repo["name"] for repo in config.get("repositories", [])]
                
                # Ensure all repos are valid
                valid_repos = [repo for repo in relevant_repos if repo in available_repos]
                
                if not valid_repos:
                    logger.warning(f"No valid repositories found in LLM response. Falling back to all repositories.")
                    return available_repos
                
                logger.info(f"LLM determined relevant repositories: {valid_repos}")
                return valid_repos
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM response as JSON: {str(e)}")
                # Fall back to getting all active repositories
                config_path = os.path.join(PROJECT_ROOT, "config.json")
                with open(config_path, "r") as f:
                    config = json.load(f)
                return [repo["name"] for repo in config.get("repositories", [])]
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading repository summaries: {str(e)}")
            # Fall back to getting all active repositories
            config_path = os.path.join(PROJECT_ROOT, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            return [repo["name"] for repo in config.get("repositories", [])]
            
    except Exception as e:
        logger.error(f"Error determining relevant repositories: {str(e)}")
        # Fall back to getting all active repositories
        config_path = os.path.join(PROJECT_ROOT, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        return [repo["name"] for repo in config.get("repositories", [])]

def parse_json_array_safely(response_text):
    """
    Safely parse a JSON array from the LLM response text, handling truncated or malformed JSON.
    Returns a list of strings or empty list if parsing fails.
    """
    response_text = response_text.strip()
    
    # Try direct JSON parsing first
    try:
        if response_text.startswith("[") and response_text.endswith("]"):
            return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Handle code block format
    if response_text.startswith("```") and "```" in response_text:
        # Extract content between code block markers
        parts = response_text.split("```")
        if len(parts) >= 3:
            code_content = parts[1]
            # Remove language identifier if present
            if code_content.startswith("json"):
                code_content = code_content[4:].strip()
            elif "\n" in code_content and code_content.split("\n", 1)[0].strip() in ["json", "JSON"]:
                code_content = code_content.split("\n", 1)[1].strip()
            
            try:
                return json.loads(code_content)
            except json.JSONDecodeError:
                pass
    
    # Try to find array content using regex
    import re
    array_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
    if array_match:
        try:
            array_content = array_match.group(0)
            return json.loads(array_content)
        except json.JSONDecodeError:
            pass
    
    # If all else fails, try to extract file paths using regex
    file_paths = []
    # Look for quoted strings that might be file paths
    path_matches = re.findall(r'"([^"]+\.[^"]+)"', response_text)
    if path_matches:
        for match in path_matches:
            if '/' in match and '.' in match:  # Basic check if it looks like a file path
                file_paths.append(match)
    
    return file_paths

def select_relevant_files(query, structured_content, repo_name, max_files=50, request_id=None):
    """
    Use an LLM to select the most relevant files from the structured content based on the query.
    This replaces the RAG approach with a more intelligent file selection process.
    """
    if not structured_content:
        logger.warning(f"No structured content available for repository {repo_name}")
        return []
    
    logger.info(f"Selecting relevant files from {repo_name} for query: {query}")
    
    # Create a map of file paths for easier access later
    file_map = {}
    
    # Create a map of file paths to their content previews
    file_previews = {}
    
    def extract_files(items, current_path=""):
        if not items or not isinstance(items, list):
            return
        
        for item in items:
            if item.get("type") == "file":
                file_path = item.get("path", "")
                # Skip files that are too large for context window
                content = item.get("content", "")
                # Estimate tokens (rough approximation: 4 chars per token)
                estimated_tokens = len(content) // 4
                if estimated_tokens < 6000:  # Safe limit for context
                    file_map[file_path] = item
                    
                    # Create a preview of the file content (first few lines)
                    content_lines = content.split('\n')
                    preview_lines = content_lines[:min(5, len(content_lines))]
                    preview = '\n'.join(preview_lines)
                    if len(content_lines) > 5:
                        preview += "\n..."
                    
                    # Add classification for the file type
                    file_type = "Unknown"
                    if file_path.endswith(".cs"):
                        file_type = "C# Code"
                    elif file_path.endswith(".csproj"):
                        file_type = "C# Project File"
                    elif file_path.endswith(".json"):
                        file_type = "JSON Configuration"
                    elif file_path.endswith(".py"):
                        file_type = "Python Code"
                    elif file_path.endswith(".md"):
                        file_type = "Markdown Documentation"
                    
                    file_previews[file_path] = {
                        "type": file_type,
                        "preview": preview,
                        "line_count": len(content_lines),
                        "size": len(content)
                    }
            elif item.get("type") == "dir" and "contents" in item:
                extract_files(item.get("contents", []))
    
    extract_files(structured_content)
    
    # Create a file directory listing with additional information
    file_list = []
    for file_path in sorted(file_map.keys()):
        preview_info = file_previews.get(file_path, {})
        file_type = preview_info.get("type", "Unknown")
        line_count = preview_info.get("line_count", 0)
        size = preview_info.get("size", 0)
        file_info = f"{file_path} ({file_type}, {line_count} lines, {size} bytes)"
        file_list.append(file_info)
    
    file_directory = "\n".join(file_list)
    
    # For very large repos, we might need to summarize the directory structure
    # to fit within context limits
    if len(file_directory) > 20000:
        # Truncate the file list to a reasonable size
        file_list = file_list[:500]  # Take top 500 files
        file_directory = "\n".join(file_list)
        logger.warning(f"Repository {repo_name} has too many files. Truncated to 500 entries.")
    
    # Create a sample of file previews to help the LLM understand the content better
    file_preview_samples = []
    sample_count = min(10, len(file_previews))
    sample_files = list(file_previews.keys())[:sample_count]
    
    for file_path in sample_files:
        preview_info = file_previews.get(file_path, {})
        preview = preview_info.get("preview", "")
        file_type = preview_info.get("type", "Unknown")
        file_preview_samples.append(f"File: {file_path} ({file_type})\nPreview:\n{preview}\n")
    
    file_preview_text = "\n".join(file_preview_samples)
    
    sub_request_id = f"select_files_{repo_name}_{hash(query)}"
    
    # Create prompt for file selection
    prompt = f"""You are tasked with selecting the most relevant files from a code repository to answer a specific query about building a feature in the Aevatar framework.

Query: {query}

Repository: {repo_name}

Below is a list of available files in the repository with their type and size:
{file_directory}

Here are previews of some representative files to help you understand the codebase better:
{file_preview_text}

Your task:
1. Analyze the query carefully to understand what implementation is needed
2. Examine the file structure to identify potentially relevant files
3. Select files that would be most useful as examples or templates for implementing the requested functionality
4. IMPORTANT: Include files that contain:
   - Interfaces and base classes that define the architecture
   - Implementation examples similar to what the user is trying to build
   - Files with import/using statements and dependent methods that the primary files reference
   - Configuration and utility code needed for the implementation
5. Look for dependency chains - if file A is relevant and imports code from file B, include BOTH files
6. Include interface definitions for any implementations that are selected
7. Prioritize completeness of context over brevity - include supporting files that help understand the code
8. Limit your selection to at most {max_files} files, prioritizing the most relevant ones

Return ONLY a JSON array of file paths in this format:
["file/path1.ext", "file/path2.ext", ...]

Do not include any explanation or reasoning, only the JSON array.
"""

    try:
        response = generate_completion(
            prompt,
            system_message="You are a code file selection assistant specialized in the Aevatar framework. Your task is to select the most relevant files from a repository to implement a specific feature, including all necessary dependent and imported files for complete context. Return only a valid JSON array of file paths.",
            config={
                "max_tokens": 500,
                "temperature": 0.2
            },
            request_id=sub_request_id,
            parent_request_id=request_id
        )
        
        logger.info(f"LLM response for file selection from {repo_name}: '{response}'")
        
        # Use the robust JSON parser to handle possible formatting issues
        selected_files = parse_json_array_safely(response)
        
        if not selected_files:
            logger.warning(f"Could not parse file list from LLM response. Using fallback selection.")
            # Fallback to selecting a reasonable default set of files
            selected_files = list(file_map.keys())[:max_files]
        
        logger.info(f"Selected {len(selected_files)} files from {repo_name}: {selected_files}")
        
        # Filter to keep only files that exist in our map
        valid_files = [path for path in selected_files if path in file_map]
        logger.info(f"Valid selected files: {valid_files}")
        
        # Convert to document format
        documents = []
        for file_path in valid_files:
            file_item = file_map[file_path]
            content = file_item.get("content", "")
            
            # Skip README.md files (as a safety check) and empty files
            if file_path.lower().endswith('readme.md') or not content.strip():
                logger.debug(f"Skipping file: {file_path}")
                continue
            
            documents.append({
                'document': content,
                'metadata': {'path': file_path},
                'collection': repo_name,
                'score': 1.0  # All LLM-selected files have equal score initially
            })
        
        return documents
        
    except Exception as e:
        logger.error(f"Error selecting files from {repo_name}: {str(e)}")
        return []

def get_relevant_documents(query, max_files_per_repo=50, request_id=None):
    """
    Get the most relevant documents for a query by using LLM to:
    1. Determine which repositories are relevant to the query
    2. Select the most relevant files from each repository's structured content
    
    This replaces the RAG approach with direct file selection by the LLM.
    """
    try:
        # Determine relevant repositories
        relevant_repos = determine_relevant_repos(query, request_id)
        logger.info(f"Selected relevant repositories: {relevant_repos}")
        
        all_documents = []
        
        # Process each repository
        for repo_name in relevant_repos:
            # Load structured content
            structured_content = load_structured_content(repo_name)
            if not structured_content:
                logger.warning(f"Could not load structured content for {repo_name}")
                continue
            
            # Select relevant files using LLM
            repo_documents = select_relevant_files(
                query, 
                structured_content,
                repo_name,
                max_files=max_files_per_repo,
                request_id=request_id
            )
            
            all_documents.extend(repo_documents)
            logger.info(f"Added {len(repo_documents)} documents from {repo_name}")
        
        # Log summary of selected documents
        logger.info(f"Total documents selected: {len(all_documents)}")
        repo_counts = {}
        for doc in all_documents:
            repo = doc.get('collection', 'unknown')
            repo_counts[repo] = repo_counts.get(repo, 0) + 1
        
        logger.info(f"Documents selected by repository: {repo_counts}")
        
        return all_documents
        
    except Exception as e:
        logger.error(f"Error getting relevant documents: {str(e)}")
        return []

def truncate_to_token_limit(text, max_tokens, encoding='gpt-4'):
    """
    Truncate the text to fit within the specified token limit.
    """
    tokenizer = encoding_for_model(encoding)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)

def summarize_text(text, max_tokens=200, parent_request_id=None):
    """
    Summarize the text to fit within a limited number of tokens.
    For very long texts, it first truncates before attempting summarization.
    """
    # Generate a request ID based on the input text (but tie it to the parent request)
    request_id = f"summarize_{hash(text[:100])}"
    
    # If text is very long, truncate it first to avoid context length errors
    # GPT-4o has a context length of ~128K tokens, but we use a much smaller limit for safety
    max_input_tokens = 12000
    truncated_text = truncate_to_token_limit(text, max_input_tokens)
    
    # If the text is still very large, use a simpler truncation
    if len(truncated_text) > 15000:  # ~3-4K tokens
        logger.info(f"Text too long for summarization ({len(truncated_text)} chars), using simple truncation")
        simple_summary = truncated_text[:5000] + "..."
        return simple_summary
    
    try:
        prompt = f"Summarize the following text to {max_tokens} tokens. Focus on extracting the key technical information, concepts, and code examples:\n\n{truncated_text}\n\nSummary:"
        
        system_message = "You are a technical assistant that summarizes documentation and code. Focus on preserving key technical details and code examples."
        
        return generate_completion(
            prompt,
            system_message=system_message,
            config={
                "max_tokens": max_tokens,
                "temperature": 0.3
            },
            request_id=request_id,
            parent_request_id=parent_request_id
        )
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        # Return a truncated version as fallback
        logger.info("Using fallback truncation for summarization")
        return truncated_text[:1000] + "..."

def generate_gpt4_response(query, context_docs, max_context_tokens=8000, max_response_tokens=4000, request_id=None):
    """
    Generate a response using the configured model based on the query and retrieved context documents.
    """
    # Combine documents and their metadata
    logger.info(f"Starting to generate LLM response for query: {query[:50]}...")
    
    # Get repository summaries
    repo_summaries = get_formatted_repo_summaries()
    logger.info(f"Retrieved repository summaries ({len(repo_summaries)} characters)")
    
    enhanced_docs = []
    for doc in context_docs:
        try:
            metadata = doc.get('metadata', {})
            file_path = metadata.get('path', 'unknown')
            
            # Skip README.md files (as a safety check)
            if file_path.lower().endswith('readme.md'):
                logger.debug(f"Skipping README file from context: {file_path}")
                continue
                
            collection = doc.get('collection', 'unknown')
            content = doc.get('document', '')
            
            # Add a more descriptive header for each document to make it clear to the LLM
            enhanced_doc = f"FILE: {collection}/{file_path}\n\n{content}"
            enhanced_docs.append({
                'content': enhanced_doc,
                'file_path': file_path,
                'collection': collection
            })
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
    
    logger.info(f"Processed {len(enhanced_docs)} documents for context")
    
    # Order documents by estimated relevance (currently all have equal weight)
    # In future could implement additional relevance scoring here
    
    # Group documents by repository for better organization in the context
    docs_by_repo = {}
    for doc in enhanced_docs:
        collection = doc['collection']
        if collection not in docs_by_repo:
            docs_by_repo[collection] = []
        docs_by_repo[collection].append(doc)
    
    # Create more structured context
    structured_contexts = []
    for repo, docs in docs_by_repo.items():
        repo_context = f"### {repo.upper()} FILES:\n\n"
        for doc in docs:
            # For each document, include the full path and content
            content = doc['content']
            # Determine if we need to summarize based on length
            if len(content) > 1500:  # Only summarize long documents
                summarized = summarize_text(content, max_tokens=800, parent_request_id=request_id)
                repo_context += summarized + "\n\n---\n\n"
            else:
                repo_context += content + "\n\n---\n\n"
        structured_contexts.append(repo_context)
    
    # Combine into a single string
    combined_context = "\n\n".join(structured_contexts)
    
    # Set aside tokens for the repository summaries and calculate remaining tokens for the context
    summary_token_estimate = len(repo_summaries) // 4  # Rough estimate: 4 characters per token
    remaining_context_tokens = max(1000, max_context_tokens - min(2000, summary_token_estimate))
    
    # Truncate context to fit within token limit
    truncated_context = truncate_to_token_limit(combined_context, remaining_context_tokens)
    
    # Construct file list for reference
    file_list = "\n".join([f"- {doc['collection']}/{doc['file_path']}" for doc in enhanced_docs])
    
    # Combine repository summaries with context
    full_context = f"""REPOSITORY OVERVIEW:
{repo_summaries}

SELECTED FILES FOR CONTEXT:
{file_list}

DETAILED FILE CONTENTS:
{truncated_context}"""
    
    # Final check to ensure we're within the limit
    context = truncate_to_token_limit(full_context, max_context_tokens)
    logger.info(f"Prepared context with {len(context)} characters (including repository summaries)")

    system_message = """You are an AI assistant specialized in the aelf blockchain ecosystem and Aevatar framework. 
You provide accurate, helpful information based on the context provided.
When answering:
1. First, review the repository overview to understand the broader context and purpose of each repository
2. Then review the list of selected files to understand what information is available
3. Most importantly, use the DETAILED FILE CONTENTS to provide a comprehensive and detailed answer
4. When answering coding questions, provide clear, working code examples based on the patterns in the file contents
5. Cite the specific files you're referencing (e.g., "Based on src/Example.cs...")
6. If you're unsure about something, acknowledge that but still try to provide helpful guidance based on what you do know
7. DO NOT say "the context does not contain information" if relevant files have been selected - instead use those files to provide the best possible answer
8. Format code blocks with proper syntax highlighting
9. Use markdown for better readability
10. Provide a complete, detailed, and helpful answer that would allow the user to implement the requested functionality"""

    prompt = f"""Context:
{context}

Question:
{query}

Answer the question based on the provided context. The files have been specifically selected as relevant to this question, so use their contents to provide a detailed and accurate response. 
Include code examples that demonstrate how to implement the requested functionality using patterns from the selected files.
Do not say that information is missing if files have been provided - use what's in the files to construct the best possible answer."""

    # Create final response request ID
    final_request_id = f"final_{request_id}" if request_id else None

    # Retry logic for API calls
    max_retries = 3
    retry_count = 0
    backoff_time = 2  # Starting backoff time in seconds
    
    while retry_count < max_retries:
        try:
            logger.info(f"Sending request to LLM API (attempt {retry_count + 1}/{max_retries})")
            
            # Set a timeout for the API call
            start_time = time.time()
            
            answer = generate_completion(
                prompt,
                system_message=system_message,
                config={
                    "max_tokens": max_response_tokens,
                    "temperature": 0.2,
                    "timeout": 150  # 150 second timeout for the API call
                },
                request_id=final_request_id,
                parent_request_id=request_id
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"LLM API response received in {elapsed_time:.2f} seconds")
            
            logger.info(f"Generated response of {len(answer)} characters")
            return answer
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error generating LLM response (attempt {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count >= max_retries:
                # If we've exhausted all retries, return a fallback response
                logger.warning("All retries exhausted. Returning fallback response.")
                return "I apologize, but I'm having trouble generating a response at the moment. The system might be experiencing high load or technical issues. Please try again in a few minutes."
            
            # Exponential backoff before retry
            logger.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff

def cleanup_old_conversations():
    """
    Clean up conversation histories that haven't been used for a while.
    """
    global last_cleanup_time
    current_time = time.time()
    
    # Only run cleanup periodically
    if current_time - last_cleanup_time < HISTORY_CLEANUP_INTERVAL:
        return
        
    logger.info("Performing conversation history cleanup")
    
    # Sessions to remove
    to_remove = []
    
    for session_id, history_data in conversation_histories.items():
        last_updated = history_data.get('last_updated', 0)
        if current_time - last_updated > HISTORY_CLEANUP_INTERVAL:  # 24 hours of inactivity
            to_remove.append(session_id)
    
    # Remove old sessions
    for session_id in to_remove:
        conversation_histories.pop(session_id, None)
    
    logger.info(f"Cleaned up {len(to_remove)} inactive conversation histories")
    last_cleanup_time = current_time

def is_followup_question(query, conversation_history, request_id=None, retrieved_docs=None):
    """
    Use an LLM to determine if the current query is a follow-up question based on conversation history.
    If it is a follow-up, generate and return a response directly.
    
    Parameters:
    - query: The user's question
    - conversation_history: Past conversation messages
    - request_id: Tracking ID for the request
    - retrieved_docs: Previously retrieved documents (for follow-up responses)
    
    Returns a tuple:
    - is_followup: Boolean indicating if it's a follow-up question
    - confidence: Confidence score (0-1) of the follow-up classification
    - answer: Response string if it's a follow-up, None otherwise
    """
    # Perform quick check for obvious follow-up indicators before making an LLM call
    query_lower = query.lower()
    obvious_indicators = [
        "above step", "previous answer", "you provided", "from previous", 
        "the code you showed", "implement above", "previous step",
        "previous response", "earlier response", "last step"
    ]
    
    is_followup = False
    confidence = 0.0
    
    for indicator in obvious_indicators:
        if indicator in query_lower:
            logger.info(f"Detected obvious follow-up indicator: '{indicator}'")
            is_followup = True
            confidence = 0.95
            break
    
    # If there's no history or just one query, it can't be a follow-up
    if not conversation_history or len(conversation_history) < 3:
        logger.info("Not enough conversation history for follow-up detection")
        return False, 0.0, None
    
    # Log the conversation history being analyzed
    logger.info(f"Analyzing conversation history for follow-up detection ({len(conversation_history)} messages)")
    for i, entry in enumerate(conversation_history[-6:]):  # Log last 3 exchanges at most
        role = entry.get('role', 'unknown')
        content = entry.get('content', '')
        truncated_content = content[:100] + "..." if len(content) > 100 else content
        logger.info(f"  History[{i}] - {role}: {truncated_content}")
    
    # If not already detected as follow-up, use LLM to determine
    if not is_followup:
        # Create a simplified view of the conversation for the LLM
        conversation_summary = []
        for entry in conversation_history[:-1]:  # Exclude current query
            prefix = "User: " if entry['role'] == 'user' else "Assistant: "
            # Truncate very long entries to keep the prompt manageable
            content = entry['content']
            if len(content) > 500:
                content = content[:300] + "..." + content[-200:]
            conversation_summary.append(f"{prefix}{content}")
        
        conversation_text = "\n\n".join(conversation_summary)
        
        # Create a sub-request ID for tracking
        sub_request_id = f"followup_detection_{hash(query)}"
        
        prompt = f"""Given the following conversation history and a new user query, determine if the new query is a follow-up question that builds upon, refers to, or seeks clarification about information in the previous conversation.

Conversation History:
{conversation_text}

New Query:
{query}

A follow-up question might:
1. Explicitly refer to previous answers
2. Ask for clarification or elaboration on something mentioned before
3. Request implementation of a feature discussed previously
4. Request modifications to code or solutions provided earlier
5. Ask about specific details mentioned in previous responses
6. Reference steps, numbers, or sections from a previous answer
7. Ask how to implement specific parts from a prior response

Analyze the new query in relation to the conversation history and determine if it's a follow-up question.
Output your decision as "true" or "false" followed by a confidence score between 0 and 1.
Example: "true,0.95" or "false,0.75"
"""

        system_message = """You are an AI that analyzes conversations to determine if a new query is a follow-up question to previous exchanges. Your task is to identify when a user is referencing, building upon, or asking for more details about something mentioned in earlier messages. Be especially attentive to cases where the user is asking about specific steps, sections, or code mentioned in previous responses. Output only your decision as true/false with a confidence score."""

        try:
            response = generate_completion(
                prompt,
                system_message=system_message,
                config={
                    "max_tokens": 50,
                    "temperature": 0.1
                },
                request_id=sub_request_id,
                parent_request_id=request_id
            )
            
            # Parse the response to get the decision and confidence
            response = response.strip().lower()
            logger.info(f"LLM follow-up detection response: {response}")
            
            if "true" in response:
                # Extract confidence score if present
                confidence = 0.9  # Default high confidence
                try:
                    # Try to extract confidence from patterns like "true,0.85"
                    if "," in response:
                        confidence_str = response.split(",")[1].strip()
                        confidence = float(confidence_str)
                except:
                    logger.warning("Could not parse confidence score from LLM response")
                
                is_followup = True
            else:
                # Extract confidence score if present
                confidence = 0.9  # Default high confidence
                try:
                    # Try to extract confidence from patterns like "false,0.85"
                    if "," in response:
                        confidence_str = response.split(",")[1].strip()
                        confidence = float(confidence_str)
                except:
                    logger.warning("Could not parse confidence score from LLM response")
                
                is_followup = False
                
        except Exception as e:
            logger.error(f"Error in LLM follow-up detection: {str(e)}")
            
            # Fall back to a smarter heuristic detection in case of API error
            query_lower = query.lower()
            followup_indicators = [
                "previous", "above", "earlier", "you said", "your answer",
                "step", "implement", "provided", "mentioned", "showed",
                "code", "example", "modify", "change", "update", "fix"
            ]
            
            # Count how many indicators are present
            indicator_count = sum(1 for indicator in followup_indicators if indicator in query_lower)
            
            # If multiple indicators are found, very likely a follow-up
            if indicator_count >= 2:
                logger.info(f"Fallback detected {indicator_count} follow-up indicators in query")
                is_followup = True
                confidence = 0.8
            
            # If even one indicator is found with a recent conversation, probably a follow-up
            elif indicator_count >= 1 and len(conversation_history) >= 3:
                logger.info(f"Fallback detected 1 follow-up indicator with recent conversation")
                is_followup = True
                confidence = 0.7
                
            # If the conversation has multiple exchanges within the last 5 minutes, lean toward classifying as follow-up
            elif len(conversation_history) >= 5:
                logger.info("Fallback classified as follow-up based on active conversation")
                is_followup = True
                confidence = 0.6
            else:
                is_followup = False
                confidence = 0.5  # Low confidence when falling back to heuristics
    
    # If this is a follow-up question and we have retrieved documents, generate a response
    if is_followup and retrieved_docs:
        logger.info("Generating response for follow-up question")
        answer = generate_followup_response(query, retrieved_docs, conversation_history, request_id)
        return is_followup, confidence, answer
    
    # If it's a follow-up but we don't have retrieved_docs, just return the classification
    return is_followup, confidence, None

def generate_followup_response(query, context_docs, conversation_history, request_id=None):
    """
    Generate a response for a follow-up question, focusing on conversation history.
    This function is separate from the main response generation flow to keep concerns separated.
    """
    logger.info("Using dedicated follow-up question processing")
    
    # Log conversation history being used in generation
    if conversation_history:
        logger.info(f"Using conversation history with {len(conversation_history)} messages for follow-up response")
        for i, entry in enumerate(conversation_history[-6:]):  # Log last 3 exchanges (6 messages)
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            content_preview = content[:100] + "..." if len(content) > 100 else content
            logger.info(f"  ConvHistory[{i}] - {role}: {content_preview}")
    
    # Process context documents
    enhanced_docs = []
    for doc in context_docs:
        try:
            metadata = doc.get('metadata', {})
            file_path = metadata.get('path', 'unknown')
            
            # Skip README.md files (as a safety check)
            if file_path.lower().endswith('readme.md'):
                logger.debug(f"Skipping README file from context: {file_path}")
                continue
                
            collection = doc.get('collection', 'unknown')
            content = doc.get('document', '')
            
            # Add a more descriptive header for each document to make it clear to the LLM
            enhanced_doc = f"FILE: {collection}/{file_path}\n\n{content}"
            enhanced_docs.append({
                'content': enhanced_doc,
                'file_path': file_path,
                'collection': collection
            })
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
    
    logger.info(f"Processed {len(enhanced_docs)} documents for follow-up context")
    
    # Create file list for reference
    file_list = "\n".join([f"- {doc['collection']}/{doc['file_path']}" for doc in enhanced_docs])
    
    # Prepare the conversation history context with higher prominence
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        # For follow-ups, include more detailed conversation history
        history_entries = []
        for entry in conversation_history:
            if entry['role'] == 'user':
                history_entries.append(f"USER: {entry['content']}")
            else:  # 'assistant'
                history_entries.append(f"ASSISTANT: {entry['content']}")
        
        conversation_context = "CONVERSATION HISTORY:\n" + "\n\n".join(history_entries) + "\n\n"
        logger.info(f"Including detailed conversation history with {len(conversation_history)} exchanges for follow-up")
    
    # Build a focused context for follow-up questions that prioritizes conversation history
    context = f"""CONVERSATION HISTORY:
{conversation_context}

AVAILABLE FILES FOR REFERENCE:
{file_list}

CODE CONTEXT SUMMARY:
The codebase includes files from repositories that contain classes, interfaces, and methods
relevant to your question. The complete file list is provided above, and specific code details
can be referenced when needed."""

    # Use a system message focused on iterative improvement
    system_message = """You are an AI assistant specialized in the aelf blockchain ecosystem and Aevatar framework.
You are responding to a follow-up question where the user wants you to improve, modify, or build upon your previous response.

When answering follow-up questions:
1. Carefully analyze the conversation history to understand what the user is asking you to improve or implement
2. If they're asking you to implement TODOs or suggestions you made previously, use your earlier guidance to provide concrete implementations
3. If they're asking for improvements to code you provided, revise the code with the requested enhancements
4. Be precise and direct in addressing what the user is asking for
5. Prefer to build upon your previous answers rather than starting from scratch
6. If you're improving code, provide complete implementations, not just snippets or outlines
7. Ensure your response is immediately useful and actionable"""

    # Special prompt for follow-up questions
    prompt = f"""CONVERSATION HISTORY AND CONTEXT:
{context}

CURRENT QUERY:
{query}

Please respond to this follow-up question by directly addressing what the user is asking for.
If they want you to implement TODOs or suggestions you mentioned earlier, provide a complete implementation.
If they want you to improve code you provided earlier, make those improvements and explain what changed.
Your response should be specific, actionable, and build upon the conversation history."""

    # Create final response request ID
    final_request_id = f"followup_{request_id}" if request_id else None

    # Retry logic for API calls
    max_retries = 3
    retry_count = 0
    backoff_time = 2  # Starting backoff time in seconds
    
    while retry_count < max_retries:
        try:
            logger.info(f"Sending follow-up request to LLM API (attempt {retry_count + 1}/{max_retries})")
            
            # Set a timeout for the API call
            start_time = time.time()
            
            answer = generate_completion(
                prompt,
                system_message=system_message,
                config={
                    "max_tokens": 4000,
                    "temperature": 0.2,
                    "timeout": 150  # 150 second timeout for the API call
                },
                request_id=final_request_id,
                parent_request_id=request_id
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Follow-up LLM API response received in {elapsed_time:.2f} seconds")
            
            logger.info(f"Generated follow-up response of {len(answer)} characters")
            return answer
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error generating follow-up LLM response (attempt {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count >= max_retries:
                # If we've exhausted all retries, return a fallback response
                logger.warning("All retries exhausted. Returning fallback response for follow-up.")
                return "I apologize, but I'm having trouble generating a follow-up response at the moment. The system might be experiencing high load or technical issues. Please try again in a few minutes."
            
            # Exponential backoff before retry
            logger.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """
    API endpoint for chat functionality.
    Expects JSON with format: {"query": "your question here", "top_k": 20, "session_id": "unique_session_id"}
    Now supports conversation history for iterative improvements.
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' parameter"}), 400
        
        query = data['query']
        max_files_per_repo = data.get('max_files_per_repo', 50)  # Default to 50 if not specified
        
        # Get session ID for conversation tracking (defaults to a new random ID if not provided)
        # For Lark bot messages, create a consistent session ID based on the user ID
        if data.get('source') == 'lark_bot':
            lark_user_id = data.get('lark_user_id', 'unknown_lark_user')
            # Create a stable ID for each Lark user rather than basing it on message content
            session_id = f"lark_user_{lark_user_id}"
            logger.info(f"Using Lark-specific user session ID: {session_id}")
        else:
            # For other clients, use the provided session ID or generate a stable one
            session_id = data.get('session_id')
            if not session_id:
                # Generate a session ID based on client IP for stability
                client_ip = request.remote_addr
                session_id = f"session_{hash(client_ip)}"
                logger.info(f"Generated session ID based on client IP: {session_id}")
        
        logger.info(f"Received query in session {session_id}: {query}")
        
        # Initialize or get conversation history for this session
        if session_id not in conversation_histories:
            logger.info(f"Creating new conversation history for session {session_id}")
            conversation_histories[session_id] = {
                'history': [],
                'last_updated': time.time(),
                'last_documents': [],  # Store the last retrieved documents
                'last_timestamp': time.time()
            }
        else:
            logger.info(f"Using existing conversation history for session {session_id} with {len(conversation_histories[session_id]['history'])} messages")
        
        # Update the last access time
        conversation_histories[session_id]['last_updated'] = time.time()
        
        # Get the conversation history
        conversation_history = conversation_histories[session_id]['history']
        
        # Add the user query to history
        conversation_history.append({
            'role': 'user',
            'content': query
        })
        
        # Log conversation state after adding the new query
        logger.info(f"Conversation history now has {len(conversation_history)} messages")
        
        # Trim history if it gets too long
        if len(conversation_history) > MAX_CONVERSATION_HISTORY * 2:  # *2 because each exchange has user and assistant
            logger.info(f"Trimming conversation history from {len(conversation_history)} to {MAX_CONVERSATION_HISTORY*2} messages")
            conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY*2:]
            conversation_histories[session_id]['history'] = conversation_history
        
        # Start a request flow for this chat query
        request_id = start_request_flow(f"chat_{session_id}_{hash(query)}")
        
        # Step 1: First check if we have documents from a previous query for this session
        # This will be used for follow-up detection and handling
        previous_docs = conversation_histories[session_id].get('last_documents', [])
        
        # Step 2: Determine if this is a follow-up question using pattern detection first, then LLM
        is_followup = False
        followup_confidence = 0.0
        followup_answer = None
        
        # Check for explicit follow-up flag in the request
        if data.get('is_followup', False):
            is_followup = True
            followup_confidence = 1.0
            logger.info("Follow-up explicitly indicated in API request")
            
            # If we have documents, generate follow-up response
            if previous_docs:
                followup_answer = generate_followup_response(query, previous_docs, conversation_history, request_id)
        else:
            # Use specialized follow-up handler to determine if it's a follow-up and generate a response if needed
            is_followup, followup_confidence, followup_answer = is_followup_question(
                query, 
                conversation_history, 
                request_id=request_id,
                retrieved_docs=previous_docs
            )
            logger.info(f"Follow-up detection: {is_followup} (confidence: {followup_confidence})")
        
        # Update timestamp for this interaction
        conversation_histories[session_id]['last_timestamp'] = time.time()
        
        # If we recognized this as a follow-up and already generated a response, use it
        if is_followup and followup_answer:
            logger.info("Using pre-generated follow-up response")
            answer = followup_answer
            
            # Add the assistant's response to history
            conversation_history.append({
                'role': 'assistant',
                'content': answer
            })
            
            # Format source documents for the response (using previous documents)
            source_documents = []
            for doc in previous_docs:
                source_documents.append({
                    "content": doc.get('document', '')[:200] + "...",  # Truncated preview
                    "source": doc.get('metadata', {}).get('path', 'unknown'),
                    "collection": doc.get('collection', 'unknown'),
                    "relevance_score": 1.0  # All LLM-selected files have equal base relevance
                })
            
            # Clean up old conversation histories periodically
            cleanup_old_conversations()
            
            logger.info(f"Returning follow-up response with {len(source_documents)} source documents, history length: {len(conversation_history)}")
            
            return jsonify({
                "answer": answer,
                "source_documents": source_documents,
                "session_id": session_id,
                "is_followup": is_followup,
                "followup_confidence": followup_confidence,
                "history_length": len(conversation_history)
            })
        
        # If not a follow-up or couldn't generate a follow-up response, proceed with normal flow
        try:
            # Retrieve fresh documents for a new query
            retrieved_docs = get_relevant_documents(
                query, 
                max_files_per_repo=max_files_per_repo, 
                request_id=request_id
            )
            
            # Store these documents for potential future follow-up questions
            conversation_histories[session_id]['last_documents'] = retrieved_docs
            
            if not retrieved_docs:
                answer = "I couldn't find any relevant information in the repository."
                
                # Add the assistant's response to history
                conversation_history.append({
                    'role': 'assistant',
                    'content': answer
                })
                
                end_request_flow(request_id)  # Clean up tracking
                return jsonify({
                    "answer": answer,
                    "source_documents": [],
                    "session_id": session_id,
                    "is_followup": is_followup,
                    "history_length": len(conversation_history)
                })
            
            # Generate response using GPT-4 for a new query (no conversation history needed)
            answer = generate_gpt4_response(
                query, 
                retrieved_docs, 
                request_id=request_id
            )
            
            # Log the response before adding to history
            logger.info(f"Response generated, adding to conversation history. History will have {len(conversation_history) + 1} messages")
            
            # Add the assistant's response to history
            conversation_history.append({
                'role': 'assistant',
                'content': answer
            })
            
            # Format source documents for the response
            source_documents = []
            for doc in retrieved_docs:
                source_documents.append({
                    "content": doc.get('document', '')[:200] + "...",  # Truncated preview
                    "source": doc.get('metadata', {}).get('path', 'unknown'),
                    "collection": doc.get('collection', 'unknown'),
                    "relevance_score": 1.0  # All LLM-selected files have equal base relevance
                })
            
            # Log the full response for debugging
            logger.info(f"Returning answer with {len(source_documents)} source documents (is_followup: {is_followup}, history_length: {len(conversation_history)})")
            
            # Clean up old conversation histories periodically
            cleanup_old_conversations()
            
            return jsonify({
                "answer": answer,
                "source_documents": source_documents,
                "session_id": session_id,
                "is_followup": is_followup,
                "followup_confidence": followup_confidence,
                "history_length": len(conversation_history)
            })
        finally:
            # End the request flow to clean up tracking
            end_request_flow(request_id)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

def main():
    # Load configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5001))
    
    # Enable duplicate log suppression
    suppress_duplicate_logs()
    
    # Initialize Lark bot handler
    init_lark_bot(app)
    
    # Start the Flask app
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()

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

def determine_relevant_collections(query, request_id=None):
    """
    Use an LLM to determine which collections are most relevant to the query.
    Returns a list of collection names based on the repository summaries.
    """
    try:
        logger.info(f"Determining relevant collections for query: {query}")
        
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
                    "collection_name": f"{repo_name}_collection",
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
4. Return ONLY a JSON array of collection names in the format: ["repository_name_collection"]
5. If unsure, select all repositories
6. Do not include any explanations, only return the JSON array

JSON array of relevant collection names:"""

            sub_request_id = f"determine_collections_{hash(query)}"
            
            response = generate_completion(
                prompt,
                system_message="You are a repository selection assistant. Your task is to determine which repositories are most relevant to a given query. Return only a valid JSON array of collection names.",
                config={
                    "max_tokens": 200,
                    "temperature": 0.1 
                },
                request_id=sub_request_id,
                parent_request_id=request_id
            )
            
            logger.info(f"Raw LLM response: '{response}'")
            
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
                
                relevant_collections = json.loads(response_cleaned)
                logger.info(f"Parsed collections: {relevant_collections}")
                
                all_collections = get_active_collections()
                
                # Create mapping of repo names to collection names
                repo_to_collection_map = {}
                for collection_name in all_collections:
                    if collection_name.endswith("_collection"):
                        repo_name = collection_name[:-11]  # Remove "_collection" suffix
                        repo_to_collection_map[repo_name] = collection_name
                        repo_to_collection_map[collection_name] = collection_name
                
                # Transform LLM results to valid collection names
                transformed_collections = []
                for coll in relevant_collections:
                    if coll in all_collections:
                        transformed_collections.append(coll)
                    elif coll in repo_to_collection_map:
                        transformed_collections.append(repo_to_collection_map[coll])
                    elif f"{coll}_collection" in all_collections:
                        transformed_collections.append(f"{coll}_collection")
                
                logger.info(f"Transformed collections: {transformed_collections}")
                
                valid_collections = [coll for coll in transformed_collections if coll in all_collections]
                
                if not valid_collections:
                    logger.warning(f"No valid collections found in LLM response. Falling back to all collections.")
                    return all_collections
                
                logger.info(f"LLM determined relevant collections: {valid_collections}")
                return valid_collections
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM response as JSON: {str(e)}")
                return get_active_collections()
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading repository summaries: {str(e)}")
            return get_active_collections()
            
    except Exception as e:
        logger.error(f"Error determining relevant collections: {str(e)}")
        return get_active_collections()

def query_vector_db(query, top_k=20, request_id=None):
    """
    Query collections in the vector database to retrieve the most relevant documents.
    First uses LLM to determine which collections are most relevant to the query,
    then only queries those collections.
    """
    collection_names = determine_relevant_collections(query, request_id=request_id)
    logger.info(f"Querying across {len(collection_names)} selected collections: {collection_names}")
    
    all_docs = []
    
    for collection_name in collection_names:
        try:
            collection = chroma_client.get_collection(name=collection_name, embedding_function=default_ef)
            
            collection_top_k = min(top_k * 2, 20)  # Get more candidates but cap at 20
            
            results = collection.query(
                query_texts=[query],
                n_results=collection_top_k
            )
            
            for i, doc_list in enumerate(results['documents']):
                for j, doc in enumerate(doc_list):
                    if not doc or len(doc.strip()) == 0:
                        continue
                        
                    metadata = results['metadatas'][i][j] if i < len(results['metadatas']) and j < len(results['metadatas'][i]) else {}
                    file_path = metadata.get('path', '')
                    
                    if file_path.lower().endswith('readme.md'):
                        logger.debug(f"Skipping README file: {file_path}")
                        continue
                    
                    metadata['collection'] = collection_name
                    
                    # Calculate position-based score
                    position_score = 1.0 - (j / collection_top_k if collection_top_k > 0 else 0)
                    
                    # Calculate keyword match score
                    keyword_score = 0.0
                    query_terms = [term.lower() for term in query.split() if len(term) > 3]
                    doc_lower = doc.lower()
                    
                    for term in query_terms:
                        if term in doc_lower:
                            keyword_score += 0.2
                    
                    combined_score = position_score * 0.7 + min(keyword_score, 0.3)
                    
                    all_docs.append({
                        'document': doc,
                        'metadata': metadata,
                        'collection': collection_name,
                        'score': combined_score
                    })
            
            logger.info(f"Retrieved {len(doc_list) if doc_list else 0} documents from {collection_name}")
        except Exception as e:
            logger.error(f"Error querying collection {collection_name}: {str(e)}")
    
    # Sort by combined score (higher is better)
    all_docs = sorted(all_docs, key=lambda x: x.get('score', 0.0), reverse=True)
    
    # Take the top_k overall
    all_docs = all_docs[:top_k]
    
    # Log document sources
    collection_counts = {}
    for doc in all_docs:
        collection = doc.get('collection', 'unknown')
        collection_counts[collection] = collection_counts.get(collection, 0) + 1
        
    sources = [f"{doc.get('collection', 'unknown')}:{doc.get('metadata', {}).get('path', 'unknown')}" for doc in all_docs]
    logger.info(f"Selected documents: {sources}")
    logger.info(f"Documents selected by collection: {collection_counts}")
    logger.info(f"Total documents retrieved across all collections: {len(all_docs)}")
    return all_docs

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
            
            enhanced_doc = f"Source: {collection} - {file_path}\n\n{content}"
            enhanced_docs.append(enhanced_doc)
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
    
    logger.info(f"Processed {len(enhanced_docs)} documents for context")
    
    # Summarize long documents
    summarized_contexts = []
    for doc in enhanced_docs:
        if len(doc) > 1500:  # Only summarize long documents
            summarized = summarize_text(doc, max_tokens=500, parent_request_id=request_id)
            summarized_contexts.append(summarized)
        else:
            summarized_contexts.append(doc)

    # Combine into a single string
    combined_context = "\n\n---\n\n".join(summarized_contexts)
    
    # Set aside tokens for the repository summaries and calculate remaining tokens for the context
    summary_token_estimate = len(repo_summaries) // 4  # Rough estimate: 4 characters per token
    remaining_context_tokens = max(1000, max_context_tokens - min(2000, summary_token_estimate))
    
    # Truncate context to fit within token limit
    truncated_context = truncate_to_token_limit(combined_context, remaining_context_tokens)
    
    # Combine repository summaries with context
    full_context = f"""REPOSITORY OVERVIEW:
{repo_summaries}

RELEVANT DOCUMENT CHUNKS:
{truncated_context}"""
    
    # Final check to ensure we're within the limit
    context = truncate_to_token_limit(full_context, max_context_tokens)
    logger.info(f"Prepared context with {len(context)} characters (including repository summaries)")

    system_message = """You are an AI assistant specialized in the aelf blockchain ecosystem. 
You provide accurate, helpful information based on the context provided.
When answering:
1. First, review the repository overview to understand the broader context and purpose of each repository
2. Then use the specific document chunks to answer the question in detail
3. Cite the source files when referring to specific information
4. If you don't know or the context doesn't contain the information, admit it
5. Be concise but thorough
6. Format code blocks with proper syntax highlighting
7. Use markdown for better readability"""

    prompt = f"""Context:
{context}

Question:
{query}

Answer the question based only on the provided context."""

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

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """
    API endpoint for chat functionality.
    Expects JSON with format: {"query": "your question here", "top_k": 20}
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' parameter"}), 400
        
        query = data['query']
        top_k = data.get('top_k', 20)  # Default to 20 if not specified
        
        logger.info(f"Received query: {query}")
        
        # Start a request flow for this chat query
        request_id = start_request_flow(f"chat_{hash(query)}")
        
        try:
            # Retrieve documents from all collections
            retrieved_docs = query_vector_db(query, top_k=top_k, request_id=request_id)
            
            if not retrieved_docs:
                end_request_flow(request_id)  # Clean up tracking
                return jsonify({
                    "answer": "I couldn't find any relevant information in the repository.",
                    "source_documents": []
                })
            
            # Generate response using GPT-4
            answer = generate_gpt4_response(query, retrieved_docs, request_id=request_id)
            
            # Format source documents for the response
            source_documents = []
            for doc in retrieved_docs:
                source_documents.append({
                    "content": doc.get('document', '')[:200] + "...",  # Truncated preview
                    "source": doc.get('metadata', {}).get('path', 'unknown'),
                    "collection": doc.get('collection', 'unknown'),
                    "relevance_score": round(doc.get('score', 0.0), 4)  # Use the score directly (already 0-1 scale)
                })
            
            # Log the full response for debugging
            logger.info(f"Returning answer with {len(source_documents)} source documents")
            
            return jsonify({
                "answer": answer,
                "source_documents": source_documents
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
    port = int(os.getenv("PORT", 5000))
    
    # Enable duplicate log suppression
    suppress_duplicate_logs()
    
    # Initialize Lark bot handler
    init_lark_bot(app)
    
    # Start the Flask app
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()

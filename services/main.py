import os
import json
import logging
import chromadb
from chromadb.utils import embedding_functions
from openai import AzureOpenAI
from tiktoken import encoding_for_model
from dotenv import load_dotenv
from flask import Flask, request, jsonify
# Import Lark handler
from lark_handler import init_lark_bot

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

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

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

def query_vector_db(query, top_k=10):
    """
    Query collections in the vector database to retrieve the most relevant documents.
    Only queries collections from active repositories defined in config.json.
    """
    collection_names = get_active_collections()
    logger.info(f"Querying across {len(collection_names)} collections: {collection_names}")
    
    all_docs = []
    
    # First pass: get documents from each collection
    for collection_name in collection_names:
        try:
            collection = chroma_client.get_collection(name=collection_name, embedding_function=default_ef)
            
            # Query for more documents initially to ensure we get enough candidates
            collection_top_k = min(top_k * 2, 10)  # Get more candidates but cap at 10
            
            results = collection.query(
                query_texts=[query],
                n_results=collection_top_k
            )
            
            # Process results and add collection name to metadata
            for i, doc_list in enumerate(results['documents']):
                for j, doc in enumerate(doc_list):
                    # Skip empty documents
                    if not doc or len(doc.strip()) == 0:
                        continue
                        
                    metadata = results['metadatas'][i][j] if i < len(results['metadatas']) and j < len(results['metadatas'][i]) else {}
                    file_path = metadata.get('path', '')
                    
                    # Skip README.md files
                    if file_path.lower().endswith('readme.md'):
                        logger.debug(f"Skipping README file: {file_path}")
                        continue
                    
                    metadata['collection'] = collection_name
                    
                    # Assign a score based on position (ChromaDB returns in order of relevance)
                    # First result gets highest score
                    position_score = 1.0 - (j / collection_top_k if collection_top_k > 0 else 0)
                    
                    # Assign a higher base score to documents containing exact keyword matches
                    keyword_score = 0.0
                    query_terms = [term.lower() for term in query.split() if len(term) > 3]
                    doc_lower = doc.lower()
                    
                    for term in query_terms:
                        if term in doc_lower:
                            keyword_score += 0.2  # Boost for each keyword match
                    
                    # Combine scores (position is more important, but keywords can boost)
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
    
    # Second pass: rank all documents and select the best ones
    # Sort by combined score (higher is better)
    all_docs = sorted(all_docs, key=lambda x: x.get('score', 0.0), reverse=True)
    
    # Take the top_k overall
    all_docs = all_docs[:top_k]
    
    # Log document sources to help with debugging
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

def summarize_text(text, max_tokens=200):
    """
    Summarize the text to fit within a limited number of tokens.
    For very long texts, it first truncates before attempting summarization.
    """
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
        
        response = client.chat.completions.create(
            model="dapp-factory-gpt-4o-westus",
            messages=[
                {"role": "system", "content": "You are a technical assistant that summarizes documentation and code. Focus on preserving key technical details and code examples."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        # Return a truncated version as fallback
        logger.info("Using fallback truncation for summarization")
        return truncated_text[:1000] + "..."

def generate_gpt4_response(query, context_docs, max_context_tokens=8000, max_response_tokens=4000):
    """
    Generate a response using OpenAI's GPT-4 based on the query and retrieved context documents.
    """
    # Combine documents and their metadata
    logger.info(f"Starting to generate GPT-4 response for query: {query[:50]}...")
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
            summarized = summarize_text(doc, max_tokens=500)
            summarized_contexts.append(summarized)
        else:
            summarized_contexts.append(doc)

    # Combine into a single string
    combined_context = "\n\n---\n\n".join(summarized_contexts)
    
    # Truncate to fit context window
    context = truncate_to_token_limit(combined_context, max_context_tokens)
    logger.info(f"Prepared context with {len(context)} characters")

    system_message = """You are an AI assistant specialized in the aelf blockchain ecosystem. 
You provide accurate, helpful information based on the context provided.
When answering:
1. Cite the source files when referring to specific information
2. If you don't know or the context doesn't contain the information, admit it
3. Be concise but thorough
4. Format code blocks with proper syntax highlighting
5. Use markdown for better readability"""

    prompt = f"""Context:
{context}

Question:
{query}

Answer the question based only on the provided context."""

    # Retry logic for API calls
    max_retries = 3
    retry_count = 0
    backoff_time = 2  # Starting backoff time in seconds
    
    while retry_count < max_retries:
        try:
            logger.info(f"Sending request to GPT-4 API (attempt {retry_count + 1}/{max_retries})")
            
            # Set a timeout for the API call
            import time
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="dapp-factory-gpt-4o-westus",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_response_tokens,
                temperature=0.2,
                timeout=60  # 60 second timeout for the API call
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"GPT-4 API response received in {elapsed_time:.2f} seconds")
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated response of {len(answer)} characters")
            return answer
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error generating GPT-4 response (attempt {retry_count}/{max_retries}): {str(e)}")
            
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
    Expects JSON with format: {"query": "your question here", "top_k": 10}
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' parameter"}), 400
        
        query = data['query']
        top_k = data.get('top_k', 10)  # Default to 10 if not specified
        
        logger.info(f"Received query: {query}")
        
        # Retrieve documents from all collections
        retrieved_docs = query_vector_db(query, top_k=top_k)
        
        if not retrieved_docs:
            return jsonify({
                "answer": "I couldn't find any relevant information in the repository.",
                "source_documents": []
            })
        
        # Generate response using GPT-4
        answer = generate_gpt4_response(query, retrieved_docs)
        
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
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

def main():
    # Load configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    
    # Initialize Lark bot handler
    init_lark_bot(app)
    
    # Start the Flask app
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()

import os
import json
import logging
import sys
import time
from urllib.parse import urlparse
import requests
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from repo_summarizer import generate_repo_summary, generate_all_repo_summaries

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "structured_content"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "indexing.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("indexer")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

chroma_client = chromadb.PersistentClient(path=os.path.join(PROJECT_ROOT, "chroma_db"))
default_ef = embedding_functions.DefaultEmbeddingFunction()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    logger.warning("GITHUB_TOKEN not found in environment variables. You may encounter rate limiting issues.")
    logger.warning("Set GITHUB_TOKEN in your .env file to increase rate limits.")

def get_github_headers():
    """Get headers for GitHub API requests with auth token if available"""
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers

def get_repo_info(repo_url):
    """Extract owner and repository name from GitHub URL"""
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub repository URL.")
    owner = path_parts[0]
    repo = path_parts[1].replace('.git', '')
    return owner, repo

def handle_rate_limit(response):
    """Handle GitHub API rate limiting by sleeping until reset time"""
    if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        
        if remaining == 0:
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
            sleep_time = max(1, reset_time - time.time() + 1)
            
            limit = response.headers.get('X-RateLimit-Limit', 'unknown')
            reset_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reset_time))
            
            logger.warning(f"GitHub API rate limit reached (Limit: {limit}). Waiting until {reset_datetime} ({sleep_time:.2f} seconds)")
            
            if sleep_time > 300:  # If more than 5 minutes
                end_time = time.time() + sleep_time
                while time.time() < end_time:
                    remaining_time = end_time - time.time()
                    if remaining_time > 60:
                        logger.info(f"Rate limit: Still waiting... {int(remaining_time / 60)} minutes remaining.")
                    time.sleep(min(60, remaining_time))
            else:
                time.sleep(sleep_time)
                
            logger.info("Rate limit wait completed, resuming operations.")
            return True
    
    return False

def get_contents(owner, repo, branch="master", path=""):
    """Recursively retrieve repository contents, excluding media files"""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}
    headers = get_github_headers()
    
    logger.info(f"Fetching contents from {api_url} with branch {branch}")
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(api_url, params=params, headers=headers)
            
            if handle_rate_limit(response):
                continue
                
            response.raise_for_status()
            items = response.json()
            break
        
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to fetch {path} after {max_retries} attempts: {str(e)}")
                raise
            
            logger.warning(f"Error fetching {path}, retrying ({retry_count}/{max_retries}): {str(e)}")
            time.sleep(5)
    
    contents = []
    
    if not isinstance(items, list):
        items = [items]

    excluded_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.mp4', '.mp3', '.wav', '.avi', '.mov']
    excluded_files = ['README.md', 'readme.md', 'Readme.md']

    for item in items:
        try:
            if item['type'] == 'file' and (
                any(item['path'].endswith(ext) for ext in excluded_extensions) or
                any(item['name'] == excluded_file for excluded_file in excluded_files)
            ):
                logger.debug(f"Skipping excluded file: {item['path']}")
                continue

            if item['type'] == 'file':
                file_content = get_file_content(item['download_url'])
                contents.append({
                    'type': 'file',
                    'path': item['path'],
                    'content': file_content
                })
                logger.debug(f"Added file: {item['path']}")
            elif item['type'] == 'dir':
                dir_contents = get_contents(owner, repo, branch, item['path'])
                contents.append({
                    'type': 'dir',
                    'path': item['path'],
                    'contents': dir_contents
                })
                logger.debug(f"Added directory: {item['path']}")
        except Exception as e:
            logger.error(f"Error processing item {item.get('path', 'unknown')}: {str(e)}")
    
    return contents

def get_file_content(download_url):
    """Retrieve raw content of a file"""
    headers = get_github_headers()
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(download_url, headers=headers)
            
            if handle_rate_limit(response):
                continue
                
            response.raise_for_status()
            return response.text
        
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to fetch file content after {max_retries} attempts: {str(e)}")
                raise
            
            logger.warning(f"Error fetching file content, retrying ({retry_count}/{max_retries}): {str(e)}")
            time.sleep(5)

def flatten_contents(contents):
    """Flatten nested contents into a list of files"""
    flat_files = []

    def _flatten(items):
        for item in items:
            if item['type'] == 'file':
                flat_files.append({
                    'path': item['path'],
                    'content': item['content']
                })
            elif item['type'] == 'dir' and 'contents' in item:
                _flatten(item['contents'])

    _flatten(contents)
    return flat_files

def ingest_into_vector_db(files, collection_name):
    """Ingest files into the vector database"""
    try:
        chroma_client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        logger.info(f"No existing collection to delete: {collection_name}")
    
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=default_ef)

    batch_size = 100
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    
    total_ingested = 0
    for batch_index, batch in enumerate(batches):
        texts = [file['content'] for file in batch]
        metadatas = [{'path': file['path']} for file in batch]
        ids = [f"{file['path']}_{batch_index}_{i}" for i, file in enumerate(batch)]
        
        try:
            collection.add(documents=texts, metadatas=metadatas, ids=ids)
            total_ingested += len(batch)
            logger.info(f"Ingested batch {batch_index + 1}/{len(batches)} into {collection_name}")
        except Exception as e:
            logger.error(f"Error ingesting batch {batch_index + 1} into {collection_name}: {str(e)}")
    
    logger.info(f"Total ingested: {total_ingested} documents into {collection_name}")
    return total_ingested

def index_repository(repo_config):
    """Index a repository based on its configuration"""
    url = repo_config["url"]
    name = repo_config["name"]
    branch = repo_config.get("branch", "master")
    
    logger.info(f"Starting indexing for repository: {name} ({url}) branch: {branch}")
    
    try:
        owner, repo = get_repo_info(url)
        logger.info(f"Extracted owner: {owner}, repo: {repo}")
        
        repo_contents = get_contents(owner, repo, branch)
        
        output_filename = os.path.join(PROJECT_ROOT, "structured_content", f"{name}_structured_content.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(repo_contents, f, indent=2)
        logger.info(f"Repository contents saved to {output_filename}")
        
        flat_files = flatten_contents(repo_contents)
        logger.info(f"Flattened to {len(flat_files)} files")
        
        collection_name = f"{name}_collection"
        docs_ingested = ingest_into_vector_db(flat_files, collection_name)
        
        try:
            logger.info(f"Generating summary for repository: {name}")
            summary = generate_repo_summary(repo_config)
            if summary:
                logger.info(f"Successfully generated summary for {name}")
            else:
                logger.warning(f"Failed to generate summary for {name}")
        except Exception as e:
            logger.error(f"Error generating summary for {name}: {str(e)}")
        
        logger.info(f"Successfully indexed {docs_ingested} documents from {name} into {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to index repository {name}: {str(e)}")
        return False

def cleanup_old_collections():
    """Remove collections no longer in config.json"""
    try:
        config_path = os.path.join(PROJECT_ROOT, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        repositories = config.get("repositories", [])
        active_collection_names = [f"{repo['name']}_collection" for repo in repositories]
        
        all_collections = chroma_client.list_collections()
        all_collection_names = [collection.name for collection in all_collections]
        
        collections_to_remove = [name for name in all_collection_names if name not in active_collection_names]
        
        if collections_to_remove:
            logger.info(f"Found {len(collections_to_remove)} old collections to remove: {collections_to_remove}")
            
            for collection_name in collections_to_remove:
                try:
                    chroma_client.delete_collection(name=collection_name)
                    logger.info(f"Successfully deleted old collection: {collection_name}")
                except Exception as e:
                    logger.error(f"Failed to delete collection {collection_name}: {str(e)}")
        else:
            logger.info("No old collections to clean up")
            
    except Exception as e:
        logger.error(f"Error during cleanup of old collections: {str(e)}")

def main():
    """Index all repositories from the config file"""
    logger.info("Starting repository indexing process")
    
    try:
        config_path = os.path.join(PROJECT_ROOT, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config.json: {str(e)}")
        return
    
    cleanup_old_collections()
    
    repositories = config.get("repositories", [])
    logger.info(f"Found {len(repositories)} repositories in config")
    
    if not GITHUB_TOKEN:
        logger.warning("GitHub API Token not set. This will significantly limit API request rate.")
        logger.warning("Set GITHUB_TOKEN in your .env file to increase rate limits.")
        proceed = input("Continue without GitHub token? (y/n): ")
        if proceed.lower() != 'y':
            logger.info("Aborting indexing process.")
            return
    
    success_count = 0
    for repo_config in repositories:
        success = index_repository(repo_config)
        if success:
            success_count += 1
    
    try:
        logger.info("Generating combined summary file for all repositories")
        generate_all_repo_summaries()
        logger.info("Successfully generated combined summary file")
    except Exception as e:
        logger.error(f"Error generating combined summary file: {str(e)}")
    
    logger.info(f"Indexing complete. Successfully indexed {success_count} out of {len(repositories)} repositories.")

if __name__ == "__main__":
    main() 
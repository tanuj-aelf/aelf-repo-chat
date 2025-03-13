import os
import json
import logging
import requests
from urllib.parse import urlparse
from openai import AzureOpenAI
from dotenv import load_dotenv

# Determine the project root directory (one level up from services/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configure logging
logger = logging.getLogger("repo_summarizer")

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Directory for storing repository summaries
SUMMARIES_DIR = os.path.join(PROJECT_ROOT, "repo_summaries")
os.makedirs(SUMMARIES_DIR, exist_ok=True)

def get_github_headers():
    """
    Get headers for GitHub API requests including authentication token if available
    """
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    return headers

def get_repo_info(repo_url):
    """
    Extract the owner and repository name from the GitHub URL.
    """
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub repository URL.")
    owner = path_parts[0]
    repo = path_parts[1].replace('.git', '')
    return owner, repo

def fetch_repo_metadata(owner, repo):
    """
    Fetch repository metadata from GitHub API.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = get_github_headers()
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching repo metadata: {str(e)}")
        return {}

def fetch_repo_readme(owner, repo):
    """
    Fetch repository README from GitHub API.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = get_github_headers()
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        readme_data = response.json()
        
        # Get the raw content
        if 'download_url' in readme_data:
            content_response = requests.get(readme_data['download_url'])
            content_response.raise_for_status()
            return content_response.text
        return ""
    except Exception as e:
        logger.error(f"Error fetching repo README: {str(e)}")
        return ""

def generate_repo_summary(repo_config):
    """
    Generate a comprehensive summary for a repository.
    """
    repo_url = repo_config["url"]
    name = repo_config["name"]
    branch = repo_config.get("branch", "master")
    
    logger.info(f"Generating summary for repository: {name} ({repo_url})")
    
    try:
        owner, repo = get_repo_info(repo_url)
        
        # Fetch repository metadata
        metadata = fetch_repo_metadata(owner, repo)
        
        # Fetch README content
        readme = fetch_repo_readme(owner, repo)
        
        # Create a structured summary with all available information
        summary = {
            "name": name,
            "url": repo_url,
            "branch": branch,
            "metadata": {
                "full_name": metadata.get("full_name", f"{owner}/{repo}"),
                "description": metadata.get("description", ""),
                "stars": metadata.get("stargazers_count", 0),
                "forks": metadata.get("forks_count", 0),
                "language": metadata.get("language", ""),
                "topics": metadata.get("topics", []),
                "created_at": metadata.get("created_at", ""),
                "updated_at": metadata.get("updated_at", ""),
                "license": metadata.get("license", {}).get("name", "Unknown")
            },
            "readme_content": readme
        }
        
        # Use GPT-4 to generate a concise, technical summary
        if readme:
            prompt = f"""
Please create a concise technical summary of this repository based on its README file and metadata.
Focus on:
1. The purpose and primary functionality of the project
2. Key features and components
3. Technical architecture (if mentioned)
4. Main APIs or interfaces
5. Important technical details and dependencies

Repository Information:
- Name: {name}
- Description: {metadata.get("description", "No description")}
- Primary Language: {metadata.get("language", "Unknown")}
- Topics: {', '.join(metadata.get("topics", []))}

README Content:
{readme[:8000]}  # Limit README content to prevent exceeding context limits

Generate a structured technical summary in markdown format, maximum 500 words.
"""
            try:
                response = client.chat.completions.create(
                    model="dapp-factory-gpt-4o-westus",
                    messages=[
                        {"role": "system", "content": "You are a technical analyst who specializes in creating concise, accurate summaries of software repositories. Focus on technical details and architecture."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                generated_summary = response.choices[0].message.content.strip()
                summary["generated_summary"] = generated_summary
            except Exception as e:
                logger.error(f"Error generating GPT-4 summary: {str(e)}")
                summary["generated_summary"] = "Error generating summary"
        
        # Save the summary
        summary_file = os.path.join(SUMMARIES_DIR, f"{name}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary for {name} saved to {summary_file}")
        return summary
    
    except Exception as e:
        logger.error(f"Error generating summary for {name}: {str(e)}")
        return None

def generate_all_repo_summaries():
    """
    Generate summaries for all repositories in the config file.
    """
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        repositories = config.get("repositories", [])
        logger.info(f"Found {len(repositories)} repositories to summarize")
        
        summaries = {}
        for repo_config in repositories:
            name = repo_config["name"]
            summary = generate_repo_summary(repo_config)
            if summary:
                summaries[name] = summary
        
        # Create a consolidated file with all summaries
        all_summaries_file = os.path.join(SUMMARIES_DIR, "all_repo_summaries.json")
        with open(all_summaries_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2)
        
        logger.info(f"All summaries saved to {all_summaries_file}")
        return summaries
    
    except Exception as e:
        logger.error(f"Error generating repository summaries: {str(e)}")
        return {}

def get_repo_summaries():
    """
    Get all repository summaries. Generate them if they don't exist.
    """
    all_summaries_file = os.path.join(SUMMARIES_DIR, "all_repo_summaries.json")
    
    if os.path.exists(all_summaries_file):
        try:
            with open(all_summaries_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading summaries file: {str(e)}")
    
    # Generate summaries if the file doesn't exist or couldn't be read
    return generate_all_repo_summaries()

def get_formatted_repo_summaries():
    """
    Get repository summaries formatted as a markdown text for use in prompts.
    """
    summaries = get_repo_summaries()
    
    if not summaries:
        return "No repository summaries available."
    
    formatted_text = "# Repository Summaries\n\n"
    
    for name, summary in summaries.items():
        formatted_text += f"## {name}\n\n"
        
        if "generated_summary" in summary and summary["generated_summary"]:
            formatted_text += summary["generated_summary"] + "\n\n"
        else:
            metadata = summary.get("metadata", {})
            formatted_text += f"- **Description**: {metadata.get('description', 'No description')}\n"
            formatted_text += f"- **Primary Language**: {metadata.get('language', 'Unknown')}\n"
            topics = metadata.get('topics', [])
            if topics:
                formatted_text += f"- **Topics**: {', '.join(topics)}\n"
            formatted_text += "\n"
    
    return formatted_text

if __name__ == "__main__":
    # For testing/standalone generation
    generate_all_repo_summaries() 
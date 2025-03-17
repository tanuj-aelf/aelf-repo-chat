import os
import json
import logging
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
import re
# Import model configuration
from model_config import generate_completion

# Determine the project root directory (one level up from services/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configure logging
logger = logging.getLogger("repo_summarizer")

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Directory for storing repository summaries
SUMMARIES_DIR = os.path.join(PROJECT_ROOT, "repo_summaries")
os.makedirs(SUMMARIES_DIR, exist_ok=True)

# Directory containing structured content files
STRUCTURED_CONTENT_DIR = os.path.join(PROJECT_ROOT, "structured_content")

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

def load_structured_content(repo_name):
    """
    Load the structured content file for a repository if it exists.
    """
    structured_content_path = os.path.join(STRUCTURED_CONTENT_DIR, f"{repo_name}_structured_content.json")
    
    if os.path.exists(structured_content_path):
        try:
            with open(structured_content_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading structured content for {repo_name}: {str(e)}")
    
    return None

def extract_key_files_from_structured_content(structured_content, important_file_patterns=None):
    """
    Extract important files and their contents from the structured content.
    
    Args:
        structured_content: The structured content data
        important_file_patterns: List of regex patterns to identify important files
                                (e.g., [".*README\\.md", ".*\\.csproj", "src/.*Agent\\.cs"])
    
    Returns:
        A list of dictionaries with file path and content
    """
    if not structured_content:
        return []
    
    if important_file_patterns is None:
        # Default patterns for important files that help understand a repo
        important_file_patterns = [
            r".*README\.md$",
            r".*\.csproj$",
            r".*/Program\.cs$",
            r".*/Startup\.cs$",
            r".*Agent\.cs$",
            r".*Service\.cs$",
            r".*Controller\.cs$",
            r".*Interface\.cs$",
            r".*Base\.cs$",   # Base classes are often important for understanding inheritance patterns
            r".*Factory\.cs$", # Factory patterns are important for understanding object creation
            r".*Repository\.cs$", # Repository patterns for data access
            r".*Manager\.cs$",  # Manager classes often contain key business logic
            r".*Handler\.cs$",  # Handler classes often contain important event/command handling
            r".*Client\.cs$",   # Client classes for external communication
            r".*Provider\.cs$", # Provider classes often implement key functionality
            r".*\.props$",
            r".*\.sln$"
        ]
    
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in important_file_patterns]
    
    key_files = []
    
    def extract_files(items):
        result = []
        if not items or not isinstance(items, list):
            return result
            
        for item in items:
            if item.get("type") == "file":
                file_path = item.get("path", "")
                if any(pattern.match(file_path) for pattern in patterns):
                    result.append({
                        "path": file_path,
                        "content": item.get("content", "")
                    })
            elif item.get("type") == "dir" and "contents" in item:
                result.extend(extract_files(item.get("contents", [])))
        return result
    
    return extract_files(structured_content)

def extract_modules_from_structured_content(structured_content):
    """
    Extract potential modules or components from the structured content.
    
    Returns:
        A list of dictionaries with module information
    """
    if not structured_content:
        return []
    
    modules_info = []
    
    # Look for src directory
    for item in structured_content:
        if item.get("type") == "dir" and item.get("path", "").lower() in ["src", "source", "lib"]:
            # Found src directory, extract its subdirectories as potential modules
            if "contents" in item:
                for subitem in item.get("contents", []):
                    if subitem.get("type") == "dir":
                        module_name = subitem.get("path", "").split("/")[-1]
                        
                        # Find key files within the module to gather more information
                        module_description = ""
                        key_classes = []
                        interfaces = []
                        
                        # Extract key files from the module directory
                        if "contents" in subitem:
                            for file_item in subitem.get("contents", []):
                                if file_item.get("type") == "file" and file_item.get("path", "").endswith(".cs"):
                                    file_path = file_item.get("path", "")
                                    file_name = file_path.split("/")[-1]
                                    
                                    # Look for key indicator files that can help understand the module's purpose
                                    file_content = file_item.get("content", "")
                                    if file_name.endswith("Base.cs") or "Interface.cs" in file_name:
                                        # Extract class definitions
                                        class_match = re.search(r'class\s+(\w+)', file_content)
                                        if class_match:
                                            key_classes.append(class_match.group(1))
                                            
                                        # Extract interface definitions
                                        interface_match = re.search(r'interface\s+(\w+)', file_content)
                                        if interface_match:
                                            interfaces.append(interface_match.group(1))
                                    
                                    # Extract summary comments that might describe the purpose
                                    summary_match = re.search(r'///\s*<summary>(.*?)</summary>', file_content, re.DOTALL)
                                    if summary_match and not module_description:
                                        module_description = summary_match.group(1).strip()
                        
                        # Create module info dictionary
                        module_info = {
                            "name": module_name,
                            "description": module_description,
                            "key_classes": key_classes,
                            "interfaces": interfaces
                        }
                        modules_info.append(module_info)
            break
    
    return modules_info

def fetch_repo_structure(owner, repo, path="", max_depth=2, current_depth=0):
    """
    Fetch repository structure (files and directories) up to a certain depth.
    """
    if current_depth > max_depth:
        return []
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = get_github_headers()
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        contents = response.json()
        
        structure = []
        for item in contents:
            if item["type"] == "dir" and current_depth < max_depth:
                # For directories, include name and recursively get substructure
                subpath = f"{path}/{item['name']}" if path else item['name']
                substructure = fetch_repo_structure(owner, repo, subpath, max_depth, current_depth + 1)
                structure.append({
                    "type": "dir",
                    "name": item["name"],
                    "path": subpath,
                    "contents": substructure
                })
            elif item["type"] == "file":
                # For files, just include name and path
                structure.append({
                    "type": "file",
                    "name": item["name"],
                    "path": f"{path}/{item['name']}" if path else item['name']
                })
        
        return structure
    except Exception as e:
        logger.error(f"Error fetching repo structure for {path}: {str(e)}")
        return []

def extract_class_hierarchies(structured_content):
    """
    Extract class inheritance hierarchies to understand extension points
    
    Returns:
        A dictionary mapping base classes to their known subclasses
    """
    if not structured_content:
        return {}
    
    # Simple regex to find class declarations and inheritance
    class_regex = re.compile(r'class\s+(\w+)(?:\s*:\s*(\w+))?')
    
    class_hierarchy = {}
    
    def process_file_content(content):
        if not content:
            return
            
        matches = class_regex.findall(content)
        for match in matches:
            class_name, base_class = match
            if base_class and base_class.strip():
                if base_class not in class_hierarchy:
                    class_hierarchy[base_class] = []
                if class_name not in class_hierarchy[base_class]:
                    class_hierarchy[base_class].append(class_name)
    
    def extract_from_files(items):
        if not items or not isinstance(items, list):
            return
            
        for item in items:
            if item.get("type") == "file":
                file_path = item.get("path", "")
                if file_path.endswith(".cs"):
                    process_file_content(item.get("content", ""))
            elif item.get("type") == "dir" and "contents" in item:
                extract_from_files(item.get("contents", []))
    
    extract_from_files(structured_content)
    return class_hierarchy

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
        
        # Try to load structured content first
        structured_content = load_structured_content(name)
        
        # If structured content exists, use it to extract key information
        if structured_content:
            # Extract modules/components from structured content
            modules = extract_modules_from_structured_content(structured_content)
            
            # Extract important files and their contents
            key_files = extract_key_files_from_structured_content(structured_content)
            
            # Extract class hierarchy information
            class_hierarchies = extract_class_hierarchies(structured_content)
            
            # No need to fetch repository structure if we have structured content
            repo_structure = []
        else:
            # If structured content doesn't exist, fallback to fetching repo structure from GitHub API
            repo_structure = fetch_repo_structure(owner, repo, max_depth=1)
            
            # Extract src directories to identify potential modules/components from API
            module_names = []
            for item in repo_structure:
                if item.get("type") == "dir" and item.get("name", "").lower() in ["src", "source", "lib"]:
                    # Try to get subdirectories of src to identify modules
                    src_content = fetch_repo_structure(owner, repo, item.get("path", ""), max_depth=0)
                    module_names = [sub.get("name", "") for sub in src_content if sub.get("type") == "dir"]
                    break
            
            # Create simple module info objects since we don't have detailed information
            modules = []
            for name in module_names:
                if name:
                    modules.append({
                        "name": name,
                        "description": "",
                        "key_classes": [],
                        "interfaces": []
                    })
            
            key_files = []
            class_hierarchies = {}
        
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
            "readme_content": readme,
            "modules": [module_info["name"] for module_info in modules]
        }
        
        # Add structure if we fetched it from API
        if repo_structure:
            summary["structure"] = repo_structure
        
        # Use the model to generate a concise, technical summary
        if readme or key_files:
            # Prepare key file contents for the prompt
            file_contents_for_prompt = ""
            if key_files:
                # Limit to a reasonable number of files to prevent exceeding context limits
                for i, file in enumerate(key_files[:5]):  # Limit to first 5 files
                    file_path = file.get("path", "")
                    content = file.get("content", "")
                    # Truncate long file contents
                    content_preview = content[:2000] + "..." if len(content) > 2000 else content
                    file_contents_for_prompt += f"\n\n## File: {file_path}\n```\n{content_preview}\n```"
            
            # Prepare class hierarchy information for the prompt
            class_hierarchy_for_prompt = ""
            if class_hierarchies:
                class_hierarchy_for_prompt = "\n\nClass Hierarchies and Extension Points:\n"
                for base_class, subclasses in class_hierarchies.items():
                    if subclasses:
                        class_hierarchy_for_prompt += f"- {base_class}: Extended by {', '.join(subclasses)}\n"
            
            # Prepare module information for the prompt
            modules_for_prompt = ""
            if modules:
                modules_for_prompt = "\nModules/Components:\n"
                for module_info in modules:
                    module_name = module_info["name"]
                    description = module_info["description"] if module_info["description"] else "No description available"
                    
                    # Add key classes and interfaces if available
                    additional_info = ""
                    if module_info.get("key_classes", []):
                        additional_info += f"\n  - Key Classes: {', '.join(module_info['key_classes'])}"
                    if module_info.get("interfaces", []):
                        additional_info += f"\n  - Interfaces: {', '.join(module_info['interfaces'])}"
                    
                    modules_for_prompt += f"- {module_name}: {description}{additional_info}\n"
            else:
                modules_for_prompt = "\nModules/Components: No clear modules identified"
            
            prompt = f"""
Please create a detailed technical summary of this repository based on its important source code files and metadata.
Focus on extracting:

1. The purpose and primary functionality of the project
2. Key components and major features with brief descriptions (identify distinct modules/services)
3. Technical architecture and design patterns
4. Authentication mechanisms or security features (if any)
5. Database technologies or data storage methods (if any)
6. Core APIs or interfaces
7. Integration points with other systems
8. Important dependencies and technologies used

Additionally, provide detailed information on:
9. Implementation patterns and how to use key modules/classes
10. Extension points and customization options
11. Common usage examples or code patterns
12. How to implement or extend major components

Repository Information:
- Name: {name}
- Description: {metadata.get("description", "No description")}
- Primary Language: {metadata.get("language", "Unknown")}
- Topics: {', '.join(metadata.get("topics", []))}

{modules_for_prompt}
{class_hierarchy_for_prompt}

README Content:
{readme[:3000] if readme else 'No README available'}

Important Source Files:
{file_contents_for_prompt}

Generate a structured technical summary in markdown format with clean sections, maximum 800 words.
For the "Key Components and Features" section:
1. First, identify 2-3 core/foundational modules and provide detailed information for each:
   - Primary purpose and responsibility
   - Key features and capabilities
   - Implementation details and usage patterns (how developers would use or extend this module)
   - Examples of common extension or integration points

2. Then, include an "Additional Modules" subsection that covers ALL remaining modules. For each module, include:
   - What the module does (1-2 sentences)
   - How it's typically used or implemented
   - Any special integration patterns or dependencies

IMPORTANT: Ensure that EVERY module listed in the Modules/Components section is mentioned in your summary. 
None should be omitted, even if they seem less important.
"""
            try:
                system_message = "You are a technical analyst specializing in creating detailed, accurate summaries of software repositories. Focus on extracting key components, features, architectural elements, and implementation patterns. When describing modules or services, be specific about their functionality and how developers would use or extend them. Include information about inheritance hierarchies, interfaces, and common usage patterns."
                
                generated_summary = generate_completion(
                    prompt,
                    system_message=system_message,
                    config={
                        "max_tokens": 2500,
                        "temperature": 0.3
                    }
                )
                
                summary["generated_summary"] = generated_summary
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
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
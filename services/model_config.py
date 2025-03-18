"""
Model configuration for the AELF repository chat service.
This module provides a unified interface for selecting and initializing different language models
based on environment variables.
"""

import os
import logging
import time
import threading
import uuid
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
logger = logging.getLogger("model_config")

# Client cache to avoid duplicate initializations
_CLIENT_CACHE = {}

# Track completion calls in a single request flow
_ACTIVE_CALLS = {}
_ACTIVITY_LOCK = threading.Lock()
_SUPPRESSED_LOGS = False

def suppress_duplicate_logs():
    """Enable suppression of duplicate logs globally"""
    global _SUPPRESSED_LOGS
    _SUPPRESSED_LOGS = True
    logger.info("Duplicate log suppression enabled")

def enable_all_logs():
    """Disable suppression of duplicate logs globally"""
    global _SUPPRESSED_LOGS
    _SUPPRESSED_LOGS = False
    logger.info("Full logging enabled")

def get_azure_openai_client(temperature: float = 0.2, timeout: int = 60) -> Any:
    """Initialize and return an Azure OpenAI client."""
    cache_key = f"azure_openai_{temperature}_{timeout}"
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]
        
    try:
        from openai import AzureOpenAI
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        if not all([api_key, endpoint, api_version]):
            logger.error("Missing required Azure OpenAI environment variables")
            raise ValueError("Azure OpenAI configuration is incomplete. Check AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_VERSION.")
        
        logger.info("Initializing Azure OpenAI client")
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        
        client._default_temperature = temperature
        client._default_timeout = timeout
        client._default_model = os.getenv("AZURE_OPENAI_MODEL", "dapp-factory-gpt-4o-westus")
        
        _CLIENT_CACHE[cache_key] = client
        return client
    
    except ImportError:
        logger.error("Failed to import AzureOpenAI. Make sure the OpenAI package is installed.")
        raise ImportError("OpenAI package is required for Azure OpenAI integration.")

def get_openai_client(temperature: float = 0.2, timeout: int = 60) -> Any:
    """Initialize and return an OpenAI client."""
    cache_key = f"openai_{temperature}_{timeout}"
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]
        
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.error("Missing required OpenAI API key")
            raise ValueError("OpenAI configuration is incomplete. Check OPENAI_API_KEY.")
        
        logger.info("Initializing OpenAI client")
        client = OpenAI(api_key=api_key)
        
        client._default_temperature = temperature
        client._default_timeout = timeout
        client._default_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        _CLIENT_CACHE[cache_key] = client
        return client
    
    except ImportError:
        logger.error("Failed to import OpenAI. Make sure the OpenAI package is installed.")
        raise ImportError("OpenAI package is required for OpenAI integration.")

def get_anthropic_client(temperature: float = 0.2, timeout: int = 60) -> Any:
    """Initialize and return an Anthropic client."""
    cache_key = f"anthropic_{temperature}_{timeout}"
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]
        
    try:
        from anthropic import Anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            logger.error("Missing required Anthropic API key")
            raise ValueError("Anthropic configuration is incomplete. Check ANTHROPIC_API_KEY.")
        
        logger.info("Initializing Anthropic client")
        client = Anthropic(api_key=api_key)
        
        client._default_temperature = temperature
        client._default_timeout = timeout
        client._default_model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        
        _CLIENT_CACHE[cache_key] = client
        return client
    
    except ImportError:
        logger.error("Failed to import Anthropic. Make sure the Anthropic package is installed.")
        raise ImportError("Anthropic package is required for Anthropic integration.")

def get_google_client(temperature: float = 0.2, timeout: int = 60) -> Any:
    """Initialize and return a Google Generative AI client."""
    cache_key = f"google_{temperature}_{timeout}"
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]
        
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            logger.error("Missing required Google API key")
            raise ValueError("Google configuration is incomplete. Check GOOGLE_API_KEY.")
        
        logger.info("Initializing Google Generative AI client")
        genai.configure(api_key=api_key)
        
        class GoogleGenAIWrapper:
            def __init__(self, temperature=0.2, timeout=60, model="gemini-2.0-flash"):
                self._default_temperature = temperature
                self._default_timeout = timeout
                self._default_model = model
                
            def chat_completions(self):
                return GoogleChatCompletions(self)
                
        class GoogleChatCompletions:
            def __init__(self, client):
                self.client = client
                
            def create(self, model=None, messages=None, temperature=None, max_tokens=None, timeout=None):
                model = model or self.client._default_model
                temperature = temperature or self.client._default_temperature
                timeout = timeout or self.client._default_timeout
                
                prompt_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    
                    if role == "system":
                        prompt_parts.append(content)
                    else:
                        prompt_parts.append(content)
                
                try:
                    generation_config = {
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "top_p": 0.95,
                    }
                    
                    model_obj = genai.GenerativeModel(model_name=model, generation_config=generation_config)
                    response = model_obj.generate_content(prompt_parts)
                    
                    class GoogleResponse:
                        def __init__(self, response):
                            self.choices = [GoogleChoice(response)]
                            
                    class GoogleChoice:
                        def __init__(self, response):
                            self.message = GoogleMessage(response)
                            
                    class GoogleMessage:
                        def __init__(self, response):
                            self.content = response.text
                            
                    return GoogleResponse(response)
                
                except Exception as e:
                    logger.error(f"Error in Google API call: {str(e)}")
                    raise
        
        client = GoogleGenAIWrapper(temperature=temperature, timeout=timeout, model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))
        _CLIENT_CACHE[cache_key] = client
        return client
    
    except ImportError:
        logger.error("Failed to import Google Generative AI. Make sure the package is installed.")
        raise ImportError("Google Generative AI package is required for Google integration.")

def get_openrouter_client(temperature: float = 0.2, timeout: int = 60) -> Any:
    """Initialize and return an OpenRouter client (using OpenAI client with custom base_url)."""
    cache_key = f"openrouter_{temperature}_{timeout}"
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]
        
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        site_url = os.getenv("OPENROUTER_SITE_URL", "")
        site_name = os.getenv("OPENROUTER_SITE_NAME", "")
        
        if not api_key:
            logger.error("Missing required OpenRouter API key")
            raise ValueError("OpenRouter configuration is incomplete. Check OPENROUTER_API_KEY.")
        
        logger.info("Initializing OpenRouter client")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        client._default_temperature = temperature
        client._default_timeout = timeout
        client._default_model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat:free")
        client._site_url = site_url
        client._site_name = site_name
        
        _CLIENT_CACHE[cache_key] = client
        return client
    
    except ImportError:
        logger.error("Failed to import OpenAI. Make sure the OpenAI package is installed.")
        raise ImportError("OpenAI package is required for OpenRouter integration.")

def get_model_client(config: Optional[Dict[str, Any]] = None) -> Any:
    """Get the appropriate model client based on environment variables or config."""
    config = config or {}
    temperature = config.get("temperature", 0.2)
    timeout = config.get("timeout", 60)
    
    model_provider = os.getenv("MODEL_PROVIDER", "azure_openai").lower()
    cache_key = f"{model_provider}_{temperature}_{timeout}"
    
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]
    
    try:
        client = None
        if model_provider == "azure_openai":
            client = get_azure_openai_client(temperature, timeout)
        elif model_provider == "openai":
            client = get_openai_client(temperature, timeout)
        elif model_provider == "anthropic":
            client = get_anthropic_client(temperature, timeout)
        elif model_provider in ["google", "google_genai"]:
            client = get_google_client(temperature, timeout)
        elif model_provider == "openrouter":
            client = get_openrouter_client(temperature, timeout)
        else:
            logger.error(f"Unsupported model provider: {model_provider}")
            logger.info("Falling back to Azure OpenAI")
            client = get_azure_openai_client(temperature, timeout)
            
        return client
    
    except Exception as e:
        logger.error(f"Error initializing model client: {str(e)}")
        logger.info("Falling back to Azure OpenAI due to error")
        return get_azure_openai_client(temperature, timeout)

def start_request_flow(request_id=None):
    """Start tracking a new request flow, returns a request ID that can be used for 
    all related generate_completion calls."""
    if request_id is None:
        request_id = f"req_{uuid.uuid4()}"
    
    with _ACTIVITY_LOCK:
        _ACTIVE_CALLS[request_id] = {
            'started': time.time(),
            'is_primary': True,
            'children': [],
            'logged': False
        }
    
    logger.debug(f"Started new request flow: {request_id}")
    return request_id

def end_request_flow(request_id):
    """End tracking for a request flow and all its children"""
    if not request_id:
        return
    
    with _ACTIVITY_LOCK:
        if request_id in _ACTIVE_CALLS:
            # Clean up this request and all children
            for child_id in _ACTIVE_CALLS[request_id].get('children', []):
                if child_id in _ACTIVE_CALLS:
                    del _ACTIVE_CALLS[child_id]
            
            del _ACTIVE_CALLS[request_id]
            logger.debug(f"Ended request flow: {request_id}")

def _is_part_of_active_flow(request_id):
    """Check if a request ID is part of an active flow and if logging should be suppressed"""
    if not request_id:
        return False, False
    
    with _ACTIVITY_LOCK:
        # If this is a directly tracked request
        if request_id in _ACTIVE_CALLS:
            record = _ACTIVE_CALLS[request_id]
            # If it's already been logged, suppress
            if record.get('logged', False):
                return True, True
            else:
                # Mark as logged and don't suppress
                record['logged'] = True
                return True, False
        
        # Check if it's a child of any active request
        for parent_id, record in _ACTIVE_CALLS.items():
            if request_id in record.get('children', []):
                if record.get('logged', False):
                    return True, True
                else:
                    record['logged'] = True
                    return True, False
    
    return False, False

def generate_completion(prompt: Union[str, list], system_message: Optional[str] = None, 
                        config: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None,
                        parent_request_id: Optional[str] = None) -> str:
    """Generate text completion using the configured model."""
    config = config or {}
    client = get_model_client(config)
    
    max_tokens = config.get("max_tokens", 4000)
    temperature = config.get("temperature", client._default_temperature)
    timeout = config.get("timeout", client._default_timeout)
    model = config.get("model", client._default_model)
    
    # Create a unique identifier for this request if not provided
    if not request_id:
        request_id = f"gen_{uuid.uuid4()}"
    
    # Register this request in the flow if it has a parent
    if parent_request_id:
        with _ACTIVITY_LOCK:
            if parent_request_id in _ACTIVE_CALLS:
                if request_id not in _ACTIVE_CALLS[parent_request_id]['children']:
                    _ACTIVE_CALLS[parent_request_id]['children'].append(request_id)
                    _ACTIVE_CALLS[request_id] = {
                        'started': time.time(),
                        'is_primary': False,
                        'parent': parent_request_id,
                        'children': [],
                        'logged': _ACTIVE_CALLS[parent_request_id].get('logged', False)
                    }
    
    # Determine if we should log this request
    is_tracked, should_suppress = _is_part_of_active_flow(request_id)
    
    # Force suppression if global setting is enabled
    should_suppress = should_suppress or _SUPPRESSED_LOGS
    
    if isinstance(prompt, str):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
    else:
        messages = prompt
    
    if not should_suppress:
        logger.info(f"Generating completion with {type(client).__name__} (model: {model})")
    
    try:
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            if os.getenv("MODEL_PROVIDER", "").lower() == "openrouter":
                extra_headers = {}
                if hasattr(client, "_site_url") and client._site_url:
                    extra_headers["HTTP-Referer"] = client._site_url
                if hasattr(client, "_site_name") and client._site_name:
                    extra_headers["X-Title"] = client._site_name
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    extra_headers=extra_headers,
                    extra_body={}
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout
                )
            return response.choices[0].message.content.strip()
        
        elif hasattr(client, "chat_completions"):
            response = client.chat_completions().create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            return response.choices[0].message.content.strip()
        
        elif hasattr(client, "messages"):
            system = None
            user_content = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                elif msg["role"] == "user":
                    user_content += msg["content"] + "\n"
            
            response = client.messages.create(
                model=model,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": user_content}]
            )
            return response.content[0].text
        
        else:
            logger.error(f"Unsupported client type: {type(client).__name__}")
            raise ValueError(f"Unsupported client type: {type(client).__name__}")
    
    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}")
        return f"Error generating response: {str(e)}"
    finally:
        # If this is a standalone request (not part of a flow), clean it up
        if not is_tracked and not parent_request_id:
            with _ACTIVITY_LOCK:
                if request_id in _ACTIVE_CALLS:
                    del _ACTIVE_CALLS[request_id]

def clear_client_cache():
    """Clear the client cache for testing or memory management purposes."""
    global _CLIENT_CACHE
    _CLIENT_CACHE.clear()
    logger.info("Client cache cleared")

def clear_request_tracking():
    """Clear all request tracking data."""
    with _ACTIVITY_LOCK:
        global _ACTIVE_CALLS
        _ACTIVE_CALLS.clear()
    logger.info("Request tracking cleared")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        response = generate_completion(
            "Hello, what can you tell me about aelf blockchain?",
            system_message="You are a helpful assistant specializing in blockchain technology.",
            config={"temperature": 0.7, "max_tokens": 100}
        )
        print("Response:", response)
    except Exception as e:
        print(f"Test failed: {str(e)}") 
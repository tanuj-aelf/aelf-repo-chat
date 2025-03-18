"""
Model configuration for the AELF repository chat service.
This module provides a unified interface for selecting and initializing different language models
based on environment variables.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv

# Determine the project root directory (one level up from services/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Configure logging
logger = logging.getLogger("model_config")

def get_azure_openai_client(temperature: float = 0.2, timeout: int = 60) -> Any:
    """
    Initialize and return an Azure OpenAI client.
    
    Args:
        temperature: The temperature for model inference
        timeout: Request timeout in seconds
        
    Returns:
        An initialized Azure OpenAI client
    """
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
        
        # Store client defaults
        client._default_temperature = temperature
        client._default_timeout = timeout
        client._default_model = os.getenv("AZURE_OPENAI_MODEL", "dapp-factory-gpt-4o-westus")
        
        return client
    
    except ImportError:
        logger.error("Failed to import AzureOpenAI. Make sure the OpenAI package is installed.")
        raise ImportError("OpenAI package is required for Azure OpenAI integration.")

def get_openai_client(temperature: float = 0.2, timeout: int = 60) -> Any:
    """
    Initialize and return an OpenAI client.
    
    Args:
        temperature: The temperature for model inference
        timeout: Request timeout in seconds
        
    Returns:
        An initialized OpenAI client
    """
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.error("Missing required OpenAI API key")
            raise ValueError("OpenAI configuration is incomplete. Check OPENAI_API_KEY.")
        
        logger.info("Initializing OpenAI client")
        client = OpenAI(api_key=api_key)
        
        # Store client defaults
        client._default_temperature = temperature
        client._default_timeout = timeout
        client._default_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        return client
    
    except ImportError:
        logger.error("Failed to import OpenAI. Make sure the OpenAI package is installed.")
        raise ImportError("OpenAI package is required for OpenAI integration.")

def get_anthropic_client(temperature: float = 0.2, timeout: int = 60) -> Any:
    """
    Initialize and return an Anthropic client.
    
    Args:
        temperature: The temperature for model inference
        timeout: Request timeout in seconds
        
    Returns:
        An initialized Anthropic client
    """
    try:
        from anthropic import Anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            logger.error("Missing required Anthropic API key")
            raise ValueError("Anthropic configuration is incomplete. Check ANTHROPIC_API_KEY.")
        
        logger.info("Initializing Anthropic client")
        client = Anthropic(api_key=api_key)
        
        # Store client defaults
        client._default_temperature = temperature
        client._default_timeout = timeout
        client._default_model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        
        return client
    
    except ImportError:
        logger.error("Failed to import Anthropic. Make sure the Anthropic package is installed.")
        raise ImportError("Anthropic package is required for Anthropic integration.")

def get_google_client(temperature: float = 0.2, timeout: int = 60) -> Any:
    """
    Initialize and return a Google Generative AI client.
    
    Args:
        temperature: The temperature for model inference
        timeout: Request timeout in seconds
        
    Returns:
        An initialized Google Generative AI client
    """
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            logger.error("Missing required Google API key")
            raise ValueError("Google configuration is incomplete. Check GOOGLE_API_KEY.")
        
        logger.info("Initializing Google Generative AI client")
        genai.configure(api_key=api_key)
        
        # Create a wrapped client object with similar interface to other clients
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
                
                # Convert OpenAI-style messages to Google format
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
                    
                    # Create a response object similar to OpenAI's
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
        
        return GoogleGenAIWrapper(temperature=temperature, timeout=timeout, model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))
    
    except ImportError:
        logger.error("Failed to import Google Generative AI. Make sure the package is installed.")
        raise ImportError("Google Generative AI package is required for Google integration.")

def get_openrouter_client(temperature: float = 0.2, timeout: int = 60) -> Any:
    """
    Initialize and return an OpenRouter client.
    
    Args:
        temperature: The temperature for model inference
        timeout: Request timeout in seconds
        
    Returns:
        An initialized OpenRouter client (using OpenAI client with custom base_url)
    """
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
        
        # Store client defaults
        client._default_temperature = temperature
        client._default_timeout = timeout
        client._default_model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat:free")
        client._site_url = site_url
        client._site_name = site_name
        
        return client
    
    except ImportError:
        logger.error("Failed to import OpenAI. Make sure the OpenAI package is installed.")
        raise ImportError("OpenAI package is required for OpenRouter integration.")

def get_model_client(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get the appropriate model client based on environment variables or config.
    
    Args:
        config: Optional configuration dictionary with parameters like temperature
        
    Returns:
        An initialized client for the specified model provider
    """
    config = config or {}
    temperature = config.get("temperature", 0.2)
    timeout = config.get("timeout", 60)
    
    model_provider = os.getenv("MODEL_PROVIDER", "azure_openai").lower()
    
    try:
        if model_provider == "azure_openai":
            return get_azure_openai_client(temperature, timeout)
        elif model_provider == "openai":
            return get_openai_client(temperature, timeout)
        elif model_provider == "anthropic":
            return get_anthropic_client(temperature, timeout)
        elif model_provider in ["google", "google_genai"]:
            return get_google_client(temperature, timeout)
        elif model_provider == "openrouter":
            return get_openrouter_client(temperature, timeout)
        else:
            logger.error(f"Unsupported model provider: {model_provider}")
            # Default to Azure OpenAI as fallback
            logger.info("Falling back to Azure OpenAI")
            return get_azure_openai_client(temperature, timeout)
    
    except Exception as e:
        logger.error(f"Error initializing model client: {str(e)}")
        logger.info("Falling back to Azure OpenAI due to error")
        return get_azure_openai_client(temperature, timeout)

def generate_completion(prompt: Union[str, list], system_message: Optional[str] = None, 
                       config: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate text completion using the configured model.
    
    Args:
        prompt: The prompt text or list of message dictionaries
        system_message: Optional system message to include (if prompt is a string)
        config: Optional configuration parameters (temperature, max_tokens, etc.)
        
    Returns:
        The generated text completion
    """
    config = config or {}
    client = get_model_client(config)
    
    max_tokens = config.get("max_tokens", 4000)
    temperature = config.get("temperature", client._default_temperature)
    timeout = config.get("timeout", client._default_timeout)
    model = config.get("model", client._default_model)
    
    # Determine if we're using a message-based or a string-based prompt
    if isinstance(prompt, str):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
    else:
        # Assume it's already in the message format
        messages = prompt
    
    logger.info(f"Generating completion with {type(client).__name__} (model: {model})")
    
    try:
        # Different clients have different APIs
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            # OpenAI/Azure OpenAI style
            if os.getenv("MODEL_PROVIDER", "").lower() == "openrouter":
                # OpenRouter specific parameters
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
                # Standard OpenAI/Azure OpenAI
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout
                )
            return response.choices[0].message.content.strip()
        
        elif hasattr(client, "chat_completions"):
            # Google wrapper style
            response = client.chat_completions().create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            return response.choices[0].message.content.strip()
        
        elif hasattr(client, "messages"):
            # Anthropic style
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

if __name__ == "__main__":
    # Simple test
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
#!/usr/bin/env python3
"""
Test script for model configuration.
Run this script to test the model configuration and ensure it's working properly.
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# Add the parent directory to path to allow importing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from services
from services.model_config import get_model_client, generate_completion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_model_config")

def test_model_config(provider=None, verbose=False):
    """
    Test the model configuration by generating a simple completion.
    
    Args:
        provider: Optional provider to test (overrides .env configuration)
        verbose: Whether to print verbose information
    """
    # If provider is specified, temporarily override the environment variable
    original_provider = None
    if provider:
        original_provider = os.environ.get("MODEL_PROVIDER")
        os.environ["MODEL_PROVIDER"] = provider
        logger.info(f"Testing with provider: {provider}")
    
    try:
        # Test client initialization
        client = get_model_client({
            "temperature": 0.7,
            "timeout": 30
        })
        
        model_type = type(client).__name__
        logger.info(f"Successfully initialized client: {model_type}")
        
        if verbose:
            # Print detailed client info
            if hasattr(client, "_default_model"):
                logger.info(f"Default model: {client._default_model}")
            if hasattr(client, "_default_temperature"):
                logger.info(f"Default temperature: {client._default_temperature}")
        
        # Test completion generation
        prompt = "Generate a one-sentence description of the aelf blockchain project."
        system_message = "You are a helpful AI assistant specializing in blockchain technology."
        
        logger.info("Sending test completion request...")
        response = generate_completion(
            prompt,
            system_message=system_message,
            config={
                "max_tokens": 100,
                "temperature": 0.7
            }
        )
        
        logger.info("Response received!")
        print("\n" + "-" * 80)
        print("TEST RESPONSE:")
        print("-" * 80)
        print(response)
        print("-" * 80 + "\n")
        
        logger.info("Model configuration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing model configuration: {str(e)}")
        return False
    
    finally:
        # Restore original provider if we changed it
        if original_provider is not None:
            os.environ["MODEL_PROVIDER"] = original_provider
            logger.info(f"Restored original provider: {original_provider or 'None'}")

def main():
    parser = argparse.ArgumentParser(description="Test model configuration")
    parser.add_argument("--provider", choices=["azure_openai", "openai", "anthropic", "google"], 
                        help="Provider to test (overrides .env configuration)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose information")
    
    args = parser.parse_args()
    
    # Configure logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting model configuration test")
    
    success = test_model_config(args.provider, args.verbose)
    
    if success:
        logger.info("Test completed successfully")
        sys.exit(0)
    else:
        logger.error("Test failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 
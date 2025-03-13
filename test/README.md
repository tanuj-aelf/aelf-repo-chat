# AELF Repository Chat Tests

This directory contains test scripts for the AELF Repository Chat application.

## Model Configuration Test

The `test_model_config.py` script tests the model configuration system, ensuring that the application can properly connect to different LLM providers.

### Running the Test

You can run the test using the provided shell script:

```bash
# Run with default provider from .env
./run_model_test.sh

# Test with a specific provider
./run_model_test.sh --provider azure_openai
./run_model_test.sh --provider google
./run_model_test.sh --provider openai
./run_model_test.sh --provider anthropic

# Run with verbose output
./run_model_test.sh --verbose
```

### Available Providers

The following model providers are supported:

1. `azure_openai` - Azure OpenAI Service
2. `openai` - OpenAI API
3. `google` - Google Gemini API
4. `anthropic` - Anthropic Claude API

Each provider requires specific environment variables to be set in the `.env` file. Refer to the main project documentation for details.

### Troubleshooting

If the test fails, check the following:

1. Make sure the required environment variables are set in the `.env` file
2. Verify that the API keys are valid
3. Check that the required Python packages are installed:
   - For Azure OpenAI/OpenAI: `pip install openai`
   - For Google: `pip install google-generativeai`
   - For Anthropic: `pip install anthropic`
4. If testing a specific provider, ensure you have the corresponding API key set

### Example Output

A successful test will output something like:

```
=== AELF Repository Chat - Model Configuration Test ===

Running test with options:  
2023-03-13 15:42:10,123 - test_model_config - INFO - Starting model configuration test
2023-03-13 15:42:10,345 - test_model_config - INFO - Successfully initialized client: AzureOpenAI
2023-03-13 15:42:10,456 - test_model_config - INFO - Sending test completion request...
2023-03-13 15:42:12,789 - test_model_config - INFO - Response received!

--------------------------------------------------------------------------------
TEST RESPONSE:
--------------------------------------------------------------------------------
AELF is a decentralized cloud computing blockchain network designed to enable the development, deployment, and operation of smart contracts and decentralized applications with high performance and flexibility.
--------------------------------------------------------------------------------

2023-03-13 15:42:12,901 - test_model_config - INFO - Model configuration test completed successfully
2023-03-13 15:42:12,902 - test_model_config - INFO - Test completed successfully
Test completed successfully! 
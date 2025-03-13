#!/bin/bash

# Script to run model configuration tests with different providers

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Set current directory to the script's directory
cd "$(dirname "$0")"

echo -e "${BLUE}=== AELF Repository Chat - Model Configuration Test ===${NC}"
echo

# Parse command line arguments
PROVIDER=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --provider)
      PROVIDER="--provider $2"
      echo -e "${YELLOW}Testing specific provider: $2${NC}"
      shift 2
      ;;
    --verbose|-v)
      VERBOSE="--verbose"
      echo -e "${YELLOW}Verbose mode enabled${NC}"
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}Running test with options: $PROVIDER $VERBOSE${NC}"
echo 

# Run the test
python3 test_model_config.py $PROVIDER $VERBOSE

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo -e "${GREEN}Test completed successfully!${NC}"
  exit 0
else
  echo -e "${RED}Test failed with exit code: $EXIT_CODE${NC}"
  exit $EXIT_CODE
fi 
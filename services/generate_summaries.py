#!/usr/bin/env python3
"""
Script to generate repository summaries for existing repositories.
Run this script to create or update repository summaries without re-indexing.
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.repo_summarizer import generate_all_repo_summaries, get_formatted_repo_summaries

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("generate_summaries")

# Load environment variables
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

def main():
    parser = argparse.ArgumentParser(description="Generate repository summaries for existing repositories.")
    parser.add_argument("--print", action="store_true", help="Print formatted summaries to console")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all summaries, even if they already exist")
    
    args = parser.parse_args()
    
    logger.info("Starting repository summary generation")
    
    if args.force:
        logger.info("Forcing regeneration of all summaries")
        summaries = generate_all_repo_summaries()
    else:
        # get_formatted_repo_summaries will generate summaries if they don't exist
        formatted_summaries = get_formatted_repo_summaries()
        logger.info(f"Retrieved/generated repository summaries ({len(formatted_summaries)} characters)")
    
    if args.print:
        print("\n" + "="*80)
        print("REPOSITORY SUMMARIES")
        print("="*80)
        print(get_formatted_repo_summaries())
    
    logger.info("Repository summary generation complete")

if __name__ == "__main__":
    main() 
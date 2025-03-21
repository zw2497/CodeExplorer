import os
import logging
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = "us-east-1"
# BEDROCK_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
BEDROCK_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
# CODEBASE_PATH = "/Users/wuzhiche/Workspace/ATE"  # Default path
CODEBASE_PATH = "/Users/wuzhiche/Workspace/BTS"
BATCH_SIZE = 8

# Use boto3 to load credentials from ~/.aws/credentials default profile
session = boto3.Session(profile_name="default")
credentials = session.get_credentials()
if credentials:
    logger.info(f"Loaded credentials from ~/.aws/credentials: Access Key ID = {credentials.access_key[:4]}...")
else:
    logger.warning("No credentials found in ~/.aws/credentials. Configure via AWS CLI ('aws configure').")

"""設定。"""

import os
import pathlib

import dotenv

BASE_DIR = pathlib.Path(__file__).resolve().parent

dotenv.load_dotenv(BASE_DIR / ".env", verbose=True, override=False)

AZURE_API_BASE = "https://privchat-eu.openai.azure.com"
AZURE_TENANT_ID = os.environ["AZURE_TENANT_ID"]
AZURE_CLIENT_ID = os.environ["AZURE_CLIENT_ID"]
AZURE_CLIENT_CERTIFICATE_PATH = BASE_DIR / "cert-azure" / "client.pem"

AWS_IAMRA_PROFILE_ARN = os.environ["AWS_IAMRA_PROFILE_ARN"]
AWS_IAMRA_ROLE_ARN = os.environ["AWS_IAMRA_ROLE_ARN"]
AWS_IAMRA_TRUST_ANCHOR_ARN = os.environ["AWS_IAMRA_TRUST_ANCHOR_ARN"]
AWS_IAMRA_REGION = os.environ.get("AWS_IAMRA_REGION", "ap-northeast-1")
AWS_IAMRA_CERTIFICATE_PATH = BASE_DIR / "cert-aws" / "client.pem"
AWS_IAMRA_PRIVATE_PATH = BASE_DIR / "cert-aws" / "client-key.pem"

GOOGLE_PROJECT_ID = os.environ["GOOGLE_PROJECT_ID"]
GOOGLE_REGION = os.environ.get("GOOGLE_REGION", "asia-northeast1")
GOOGLE_APPLICATION_CREDENTIALS = BASE_DIR / "cert-gc" / "service-account-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(GOOGLE_APPLICATION_CREDENTIALS)

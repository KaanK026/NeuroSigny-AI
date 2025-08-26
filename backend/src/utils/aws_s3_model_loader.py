import boto3
import os
from dotenv import load_dotenv

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

def download_model_if_needed(model_name: str):
    model_dir = "model_files"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_{model_name.lower()}_model.pth")

    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading {model_name} from S3...")
        s3 = get_s3_client()
        s3.download_file(os.getenv("AWS_BUCKET_NAME"), model_name, model_path)
        print(f"Downloaded {model_name} ")
    else:
        print(f"Model {model_name} found locally ")

    return model_path
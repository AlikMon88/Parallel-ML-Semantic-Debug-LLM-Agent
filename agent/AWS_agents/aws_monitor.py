import boto3
import json
import time
import os
from datetime import datetime, timezone
from openai import OpenAI
from botocore.exceptions import ClientError

# --- AWS Configuration ---
REGION = "us-east-1"
SECRET_NAME = "prod/ml/openai-key" # Create this in AWS Secrets Manager
BUCKET_NAME = "my-training-logs-bucket"
LOG_FILE_KEY = "model/logs/training_logs.json"
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:123456789012:ML-Alerts"

def get_openai_key():
    """Fetches the OpenAI key from AWS Secrets Manager."""
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=REGION)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=SECRET_NAME)
        return json.loads(get_secret_value_response['SecretString'])['OPENAI_API_KEY']
    except Exception as e:
        print(f"Failed to fetch secret: {e}")
        raise

# Initialize OpenAI with AWS-sourced key
client = OpenAI(api_key=get_openai_key())

def read_logs_from_s3():
    """Fetches the latest logs from an S3 bucket."""
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=LOG_FILE_KEY)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error reading S3: {e}")
        return None

def notify_user(reason, epoch, loss):
    """Sends an alert via AWS SNS."""
    sns = boto3.client('sns', region_name=REGION)
    message = f"🚨 ML Training Triggered!\nEpoch: {epoch}\nLoss: {loss}\nReason: {reason}"
    sns.publish(TopicArn=SNS_TOPIC_ARN, Message=message, Subject="ML Monitor Alert")

def get_gpt_decision(logs):
    """GPT-4o reasoning logic."""
    timestamp = datetime.now(timezone.utc).isoformat()
    system_prompt = """
    ROLE: ML Monitoring Agent. 
    OUTPUT: Valid JSON ONLY. 
    SCHEMA: {"epoch": int, "train_loss": float, "is_trigger": bool, "trigger_reason": str}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Time: {timestamp}\nLogs: {json.dumps(logs)}"}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return None

def main_loop():
    print("--- AWS ML MONITOR ACTIVE ---")
    while True:
        logs = read_logs_from_s3()
        if logs:
            decision = get_gpt_decision(logs)
            if decision:
                print(f"[{datetime.now()}] Epoch {decision['epoch']} | Loss: {decision['train_loss']}")
                
                if decision['is_trigger'] and decision['epoch'] > 3:
                    print(f"!!! TRIGGER: {decision['trigger_reason']}")
                    notify_user(decision['trigger_reason'], decision['epoch'], decision['train_loss'])
                    # Optional: Run a subprocess to stop the training instance via boto3
        
        time.sleep(300) # Check every 5 minutes

if __name__ == "__main__":
    main_loop()
import os
import json
import time
import subprocess
import re
from datetime import datetime, timezone
from pprint import pprint
from openai import OpenAI
from pathlib import Path
# from data.load import *
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

BASEDIR = Path(__file__).parents[2]

TRAIN_LOGS_PATH = BASEDIR /'model'/ 'logs' / 'training_logs.json'
DECISION_LOGS_PATH = BASEDIR / 'agent' / 'AWS_agents' / 'logs' / 'decision_logs.json'
MODEL_TRAIN_FILE_PATH = BASEDIR / 'model' / 'sample_train_2.py'
STREAM_FILE_PATH = BASEDIR / 'main_stream.py'


print(TRAIN_LOGS_PATH)
print(DECISION_LOGS_PATH)

load_dotenv()

def load_llm(model_name='other'):
    if model_name == 'openai':
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: Please set your OPENAI_API_KEY as an environment variable.")
            exit(1)

        print("Initializing LLM model...")
    
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # gpt-4o-mini is fast and cheap
    
    else:
        
        # model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        model_id = "distilgpt2"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
        )

        llm = HuggingFacePipeline(pipeline=pipe)

    return llm

def get_sys_instructions_e1():
    return """
    ROLE:
    You are an automated ML training monitoring agent.

    OBJECTIVE:
    Analyze the latest machine learning training logs and determine if a debugging workflow should be triggered.

    TRIGGER CONDITIONS:
    - Loss becomes NaN
    - Loss increases continuously (divergence)
    - Accuracy drops significantly
    - Unexpected runtime errors appear

    OUTPUT FORMAT:
    You must return ONLY valid JSON in this schema:
    {{
        "utc_timestamp": "ISO-8601",
        "epoch": int,
        "train_loss": float,
        "is_trigger": bool,
        "trigger_reason": "string"
    }}
    
    IMPORTANT:
    - Return ONLY JSON
    - No explanations
    - No markdown
    """

def run_adjoin_code_parallel(train_file_path=MODEL_TRAIN_FILE_PATH, stream_file_path=STREAM_FILE_PATH, is_train=False):
    if is_train:
        print('Training the model ...')
        subprocess.run(['python', train_file_path], check=True)
    else:
        print('loaded trained logs')
        
    subprocess.run(['streamlit', 'run', stream_file_path])

def utc_now():
    from datetime import datetime, timezone
    """provides the UTC time"""
    return {"utc_time": datetime.now(timezone.utc).isoformat()}

def clean_str_to_json(stdout):
    match = re.search(r'\{.*\}', stdout, re.DOTALL)
    if match:
        json_text = match.group()
        decision = json.loads(json_text)
        return decision
    else:
        print("No JSON found in output")
        return None
    
def read_train_logs(path=TRAIN_LOGS_PATH):
    """Reads the local ml-model training-logs"""
    try:
        with open(path, 'r') as f:
            train_logs = json.load(f)
        f.close()
        train_logs = train_logs['epochs']
        return {'ml_training_logs' : train_logs}
    except Exception as e:
        return {"error": f"Could not read logs: {str(e)}"}

def run_monitor_agent():
    """The core reasoning-decision step"""
    def log_str(logs):
        logs = logs['ml_training_logs']
        logs_str = ''
        for elements in logs:
            temp_str = ''
            for key, value in elements.items():
                temp_str += f"'{key}': {value}\n"
            logs_str += f"\n\n{temp_str}"
        return logs_str
        
    logs = read_train_logs()
    logs = log_str(logs)
    timestamp = str(utc_now()['utc_time'])
    
    sys_prompt = get_sys_instructions_e1()
    inj_prompt = (f"Current UTC Time: \n",
                  f"{timestamp}\n\n",
                  f"Latest Training Logs:\n"
                  f"{logs}\n\n")
    
    inj_prompt = str(inj_prompt)

    try:
        llm = load_llm(model_name='openai')
        prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{query}")
        ])
    
        ## LCEL Chain: Prompt -> LLM -> String Output -> json-parse
        chain = prompt | llm | StrOutputParser()  
        output = chain.invoke({'query': inj_prompt})
        return clean_str_to_json(output)
    
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return None

def monitoring_loop(cycles=10, interval_seconds=300):
    """The main 24/7 loop equivalent"""
    print()
    print(f"--- Starting <Monitoring> Agent ---")
    
    for i in range(cycles):
        print(f"\n[Cycle {i+1}] Checking logs...")
        
        decision = run_monitor_agent()
        
        if not decision:
            continue

        pprint(decision)

        # Save decision history
        with open(DECISION_LOGS_PATH, 'a') as f:
            f.write(json.dumps(decision) + "\n")

        # logic: Only trigger if past epoch 3 and AI flags an issue
        if decision.get('epoch', 0) > 3:
            if decision.get('is_trigger'):
                print(f"Triggered <Parallel>: {decision['trigger_reason']}")
                run_adjoin_code_parallel(is_train=False)
            else:
                print("Status: Normal.")
        else:
            print(f"Status: Warm-up phase (Epoch {decision.get('epoch')})")

        print(f"Waiting {interval_seconds}s for next check...")
        time.sleep(interval_seconds)

if __name__ == '__main__':
    monitoring_loop(cycles=2, interval_seconds=10) # Runs for ~4 hours
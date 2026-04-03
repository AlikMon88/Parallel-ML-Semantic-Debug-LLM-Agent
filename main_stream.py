import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

from data.load import *
from rag.embed import *
from rag.retrieve import *
from agent.logic import *

from pprint import pprint 
from dotenv import load_dotenv
import sys

import streamlit as st
from stream.template import *

load_dotenv()

def load_llm(model_name='other'):
    if model_name == 'openai':
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: Please set your OPENAI_API_KEY as an environment variable.")
            exit(1)

        print("Initializing ML Debugging Assistant...")
    
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

if __name__ == '__main__':
    
    cate_dict = {
        '[DATA]': 'data/incident_spaces/data_incidents.json',
        '[COMPUTE]': 'data/incident_spaces/compute_incidents.json',
        '[CODE]': 'data/incident_spaces/code_incidents.json'
    }
    vector_embed_cache = {}
    
    documents = load_documents(cate_dict['[DATA]'])
    
    ## Reference for RAG
    print(' --- DATA-RAG-DOC --- ')
    pprint(documents)
    
    for key in cate_dict.keys():
        documents = load_documents(cate_dict[key])
        vector_embed_cache[key] = create_vector_store(documents) ## BERT-based embedding-vector (encoder-architechture)
        print('vector-embed-shape: ', vector_embed_cache[key].index.ntotal, vector_embed_cache[key].index.d)
        
    
    ## Decoder LLM (TinyLLAMA/GPT-4o-mini)
    llm = load_llm(model_name='openai')   
    
    
    ### Streamlit / FrontEnd
    stream_frontend(load_vector_embed=vector_embed_cache, load_llm=llm)
    
    
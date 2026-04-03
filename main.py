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
        
    user_query = "My model's accuracy suddenly dropped after yesterday's deployment. What should I check?"
    ## out-of-distrib.
    # user_query = "My eyes are feverish yellow, not the model look one would have since yesterday. What should I check?"
    user_query = "My model is suddenly facing memory botteleneck RAM after yesterday's test when it was perfect. What should I check?"
    
    print(f"User Query: '{user_query}'\n")
    
    print('SIMILARITY-SEARCH')
    docs = vector_embed_cache['[CODE]'].similarity_search(user_query, k=2)
    for d in docs:
        print(d.page_content)
        print("---")
    
    # sys.exit()
    
    ## Decoder LLM (TinyLLAMA)
    llm = load_llm(model_name='openai')   
    
    pprint(llm)
    
    category = route_query(user_query, llm)
    print(f"Agent Router classified this as: {category}\n")
    
    ## better to use a vector_embedd cache
    rag_chain = build_rag_chain(vector_embed_cache[category], llm)
    response = rag_chain.invoke(user_query)
    
    
    print("=== AI Diagnostic Report ===")
    print(response)
    
    ### Streamlit
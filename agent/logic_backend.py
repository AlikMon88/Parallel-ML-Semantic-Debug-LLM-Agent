
import json
import os
from data.load import *
from rag.embed import *
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def route_query(query: str, llm) -> str:
    ## Categorizes the user's query
    
    system_prompt = """You are an intelligent routing agent for an ML system.
    Categorize the user's issue into exactly ONE of these categories:
    - [COMPUTE] (e.g., OOM, CUDA errors, hardware)
    - [DATA] (e.g., NaN loss, data drift, distribution shift)
    -[CODE] (e.g., syntax errors, tensor shape mismatches)
    
    Respond with ONLY the category tag (e.g., [DATA])."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    ## LCEL Chain: Prompt -> LLM -> String Output
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query}).strip()


### Tooling (JSON-model-Schema) / MCP-like
@tool
def read_training_logs(filepath: str = "training_logs.json"):
    """Reads the JSON log file from the recent training run to check metrics like loss and gradients."""
    if not os.path.exists(filepath):
        return "Error: Log file not found."
    with open(filepath, "r") as f:
        data = json.load(f)
    return json.dumps(data)

@tool
def run_shap_analysis(model_path: str = "model.pt"):
    """Runs SHAP feature importance analysis on a saved model checkpoint."""
    # Simulating a heavy SHAP computation
    return """SHAP Execution Complete: 
    - Feature 'pixel_cluster_center' importance: 88% (Abnormally high)
    - Feature 'edges' importance: 2%
    Conclusion: Model is ignoring spatial context and over-relying on center pixels (Memorization/Overfitting)."""

@tool
def search_framework_docs(query: str):
    """Searches PyTorch/TensorFlow documentation for fixes to ML issues."""
    # Simulating a FAISS vector DB retrieval
    if "overfit" in query.lower() or "val loss" in query.lower():
        return "DocSnippet [PyTorch]: To combat overfitting and memorization, apply `nn.Dropout(p=0.5)` to fully connected layers and utilize `torchvision.transforms` for data augmentation."
    return "No relevant documentation found."

@tool
def search_db_files(query): ##for categorical-routing
    """ Searches Local Database Files/JSONs/Data for causes and fixes to ML issues."""

    cate_dict = {
    '[DATA]': 'data/incident_spaces/data_incidents.json',
    '[COMPUTE]': 'data/incident_spaces/compute_incidents.json',
    '[CODE]': 'data/incident_spaces/code_incidents.json'
    }
    
    vector_embed_cache = {}
    
    documents = load_documents(cate_dict['[DATA]'])
            
    for key in cate_dict.keys():
        documents = load_documents(cate_dict[key])
        vector_embed_cache[key] = create_vector_store(documents) ## BERT-based embedding-vector (encoder-architechture)
        print('vector-embed-shape: ', vector_embed_cache[key].index.ntotal, vector_embed_cache[key].index.d)

    category = route_query(query, llm)
    selected_vs = vector_embed_cache[category]
    retriever = selected_vs.as_retriever(search_kwargs={"k": 2})
    ret_docs = retriever.invoke(query) ## LangChain-doc-datatype
    
    return "\n\n".join(doc.page_content for doc in ret_docs)
    
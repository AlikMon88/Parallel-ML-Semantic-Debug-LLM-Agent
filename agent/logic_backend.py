
import json
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

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

    category = route_query(user_query, llm)
    st.write(f"2. Routing to the **{category}** vector database...")
    selected_vs = vector_stores[category]
    retriever = selected_vs.as_retriever(search_kwargs={"k": 2})
    return retriever ## return retrieved vectors
    
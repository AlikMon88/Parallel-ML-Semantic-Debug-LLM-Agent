
# ML System Debugging Assistant (RAG + LLM)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system to diagnose common issues in machine learning pipelines. Given a user query about model failures, the system retrieves similar past incidents from a vector database and generates possible causes and fixes using an LLM.

---

## Motivation
Debugging ML systems in production is often time-consuming and relies heavily on past experience. This project simulates a system that leverages historical incidents and documentation to assist in identifying root causes efficiently.

---

## Architecture

1. Incident data is embedded and stored in a vector database (FAISS)  
2. User queries are converted into embeddings  
3. Relevant incidents are retrieved using similarity search  
4. An LLM generates diagnostic insights and recommendations  

---

## Example

### Input
```

Model accuracy dropped after deployment

```

### Output
- Possible causes:
  - Data distribution shift
  - Feature mismatch  

- Suggested fixes:
  - Retrain with updated dataset  
  - Validate preprocessing pipeline  

---

## Tech Stack

- Python  
- FAISS (vector database)  
- LLM API (OpenAI or open-source models)  

---

## Key Features

- Semantic retrieval of similar ML incidents  
- Context-aware response generation using LLMs  
- Simple agent-style routing for different issue types (e.g. accuracy, latency)  

---

## Project Structure

```

project/
│── data/
│   └── incidents.json
│── rag/
│   ├── embed.py
│   ├── retrieve.py
│── agent/
│   └── logic.py
│── main.py
│── README.md

```

---

## How It Works

1. Incident descriptions are converted into embeddings  
2. Stored in a FAISS index for fast similarity search  
3. User query is embedded and matched with similar incidents  
4. Retrieved context is passed to an LLM  
5. LLM generates root cause analysis and suggested fixes  

---

## Future Improvements

- Integrate real system logs and monitoring data  
- Add reranking for improved retrieval quality  
- Implement evaluation metrics for response quality  
- Extend agent logic with more tools  

---

## Why This Project Matters

- Demonstrates practical use of RAG in ML systems  
- Shows ability to build end-to-end LLM applications  
- Focuses on real-world problem solving (debugging ML pipelines)  

---

## Getting Started

1. Clone the repository  
2. Install dependencies  
3. Run embedding pipeline  
4. Start querying the system  

---

## Final Note

This project focuses on clarity and practicality over complexity.  
The goal is to demonstrate how LLMs + retrieval can be used to solve real engineering problems.

```

---

* add a **screenshot or sample output**
* keep repo clean (no random notebooks)

---

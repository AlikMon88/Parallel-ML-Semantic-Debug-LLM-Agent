
---

# Parallel: ML Semantic Debugging Assistant (RAG + LLM)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system to diagnose common issues in machine learning pipelines. Given a user query about model failures, the system retrieves similar past incidents from a vector database and generates possible causes and fixes using an LLM.

The system has been extended with intelligent routing, live agent decision-making, and an interactive frontend for real-time debugging workflows.

---

<p align="center">
  <img src="snaps/snap_1.png" width="45%" />
  <img src="snaps/snap_2.png" width="45%" />
</p>

---

## Motivation

Debugging ML systems in production is often time-consuming and relies heavily on past experience. This project simulates a system that leverages historical incidents, structured retrieval, and dynamic reasoning to assist in identifying root causes efficiently.

---

## Architecture

1. Incident data is embedded and stored in a vector database (FAISS)
2. User queries are converted into embeddings
3. A routing layer determines the appropriate data source or tool
4. Relevant incidents are retrieved using similarity search
5. A LangChain-powered agent dynamically decides:

   * Whether to query the database
   * Whether to fetch system/model state
6. Retrieved context is passed to an LLM
7. The LLM generates diagnostic insights and recommendations
8. Results are displayed via an interactive Streamlit frontend

---

## Key Enhancements

* **Routing-Based Database Access**

  * Intelligent selection of retrieval pathways based on query type

* **Agent-Based Decision Making**

  * Dynamic tool usage using LangChain agents
  * Enables context-aware reasoning beyond static RAG

* **Live System State Retrieval**

  * Simulates fetching runtime model/system signals

* **Interactive Frontend**

  * Built with Streamlit for real-time querying and visualization

---

## Example

### Input

```
Model accuracy dropped after deployment
```

### Output

* Possible causes:

  * Data distribution shift
  * Feature mismatch

* Suggested fixes:

  * Retrain with updated dataset
  * Validate preprocessing pipeline

---

## Tech Stack

* Python
* FAISS (vector database)
* LangChain (agent + tool orchestration)
* LLM API (OpenAI or open-source models)
* Streamlit (frontend UI)

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
│── snaps/
│   ├── screen1.png
│   ├── screen2.png
│── app.py              # Streamlit frontend
│── main.py
│── README.md
```

---

## How It Works

1. Incident descriptions are converted into embeddings
2. Stored in a FAISS index for fast similarity search
3. User query is embedded and routed intelligently
4. Agent determines whether to:

   * Retrieve past incidents
   * Query system/model state
5. Relevant context is aggregated
6. LLM generates root cause analysis and suggested fixes
7. Output is displayed in an interactive UI

---

## Future Improvements

* Integrate real system logs and monitoring pipelines
* Add reranking for improved retrieval quality
* Implement evaluation metrics for response quality
* Expand agent tooling for deeper diagnostics
* Add memory for session-based debugging workflows

---

## Why This Project Matters

* Demonstrates practical use of RAG in ML systems
* Showcases agent-based reasoning with tool usage
* Highlights end-to-end LLM application development
* Focuses on real-world engineering problem solving

---

## Getting Started

1. Clone the repository
2. Install dependencies
3. Run embedding pipeline
4. Launch the Streamlit app:

   ```
   streamlit run app.py
   ```
5. Start querying the system

---

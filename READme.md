# << Parallel >>

### Agentic Semantic Debugging for ML Pipelines

---

## Overview

**Parallel** is a continuous ML debugging system that monitors training in real time and automatically performs **Root Cause Analysis (RCA)** when failures occur.

Parallel runs alongside training, watches logs, detects abnormal behavior, and launches structured debugging workflows using coordinated agents.

It acts as an **autonomous debugging layer** for complex machine learning systems.

---

<p align="center">
  <img src="snaps/pic_1.png" width="32%" />
  <img src="snaps/pic_2.png" width="32%" />
  <img src="snaps/pic_3.png" width="32%" />
</p>

## Core Idea

Traditional ML debugging is manual and reactive.

Parallel converts debugging into an automated workflow:

```text
Training Runs
        ↓
Monitoring Agent Watches Logs
        ↓
Failure Detected
        ↓
Parallel Agents Triggered
        ↓
Root Cause Identified
        ↓
Fix Suggested
        ↓
Incident Stored
````

---

## System Architecture

Parallel uses **LangGraph** to orchestrate modular diagnostic agents.

The system consists of:

Monitoring Layer
• Continuous log monitoring
• Failure detection
• Automatic triggering

Debugging Layer
• Multi-tool diagnostics
• Structured reasoning
• Root cause generation

Memory Layer
• Historical incident storage
• Retrieval-based reasoning (FAISS)

---

## Monitoring

Parallel supports **continuous monitoring** using a persistent agent.

Run the monitoring agent:

```bash
python agent/AWS_Agents/local_monitor.py
```

This agent:

• Watches training logs continuously
• Detects abnormal behavior
• Triggers debugging workflows
• Stores monitoring decisions

It can run:

• Locally (development mode)
• On cloud instances (24/7 mode)

---

## Diagnostic Tools

Parallel includes modular tools for:

• Log analysis
• Data distribution inspection
• Feature importance analysis
• Model architecture validation
• Historical incident retrieval (FAISS)

Tools are dynamically orchestrated using **LangGraph**.

---

## Example Trigger

Monitoring detects instability:

```json
{
  "epoch": 17,
  "is_trigger": true,
  "trigger_reason": "Loss became NaN"
}
```

Parallel automatically runs RCA and produces:

• Root causes
• Supporting signals
• Suggested fixes

Results are displayed through the Streamlit debugging interface.

---

## Key Capabilities

• Continuous training monitoring
• Automated root cause analysis
• Multi-agent debugging workflows
• Experience-based failure memory
• Modular diagnostic architecture

---

## Running Parallel

### Step 1 — Start Training

```bash
python models/sample_train_2.py
```

---

### Step 2 — Start Monitoring

```bash
python agent/AWS_Agents/local_monitor.py
```

Parallel will automatically trigger debugging when failures are detected.

---

## Optional Deployment

Parallel can run:

Local Mode
Run monitoring on your local machine.

Cloud Mode
Run monitoring on a persistent server (e.g., AWS EC2) for 24/7 operation.

---

## Future Work

• Inference-time monitoring
• Concept drift detection
• Automated retraining
• RCA visualization dashboards
• Multi-model debugging support

---

## License

MIT License.

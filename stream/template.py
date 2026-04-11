import streamlit as st
from data.load import *
from rag.embed import *
from rag.retrieve import *
from agent.logic import *
from agent.logic_backend import *
from langchain_core.messages import HumanMessage, AIMessage

def stream_frontend(load_vector_embed, load_llm):
    st.set_page_config(page_title="ML-SYS-DBG", page_icon="⚙️")

    st.title(" < Autonomous ML Semantic-Debugging Assist. > ")
    st.markdown("Enter your ML pipeline issue below. The agent will route your query to the correct specialized database ([COMPUTE] / [DATA] / [CODE]) and retrieve historical fixes.")

    with st.spinner("Initializing AI Models and Vector Databases..."):
        vector_stores = load_vector_embed.copy()
        llm = load_llm.copy()
        llm_with_tools = llm.bind_tools([query_grafana_metrics, run_shap_explainer])

    # UI Form
    user_query = st.text_area("Describe your ML Issue:", placeholder="e.g., My training job keeps crashing with CUDA out of memory on step 500.")

    if st.button("Diagnose Issue"):
        if not user_query:
            st.warning("Please enter a query.")
        else:
            # Routing (Displaying the thought process)
            with st.status("Agent thinking...", expanded=True) as status:
                live_tool_results = 'Agent: No live tools data required.'
                
                st.write('Evaluating if live diagonistic tool are needed ...')
                agent_decision = llm_with_tools.invoke(user_query)
                
                if agent_decision.tool_calls:
                    live_tool_results = agent_tool_call(agent_decision)                    
                else:
                    st.write(live_tool_results)
                    
                st.write("1. Analyzing user intent...")
                category = route_query(user_query, llm)
                
                st.write(f"2. Routing to the **{category}** vector database...")
                selected_vs = vector_stores[category]
                  
                st.write("3. Retrieving historical context & generating diagnosis...")
                ## better to use a vector_embedd cache
                rag_chain = build_rag_chain(selected_vs, llm)
                response = rag_chain.invoke({"query": user_query, "live_metrics": live_tool_results})
                status.update(label="Diagnosis Complete!", state="complete", expanded=False)
            
            # Final Output
            st.subheader(" --- Diagnostic Report --- ")
            st.success(response)
            
            # Show a UI pill so recruiters see the exact tag it used
            st.caption(f"Routed via: `{category}` Database")
            
            
### Parallel-Stream-FrontEnd
def stream_frontend_parallel(load_vector_embed, load_llm):
    st.set_page_config(page_title="Parallel | ML-SYS-DBG", page_icon="⚙️")

    st.title(" < Parallel | Autonomous ML Semantic/Debugging Assist. > ")
    # st.markdown("Enter your ML pipeline issue below. The agent will route your query to the correct specialized database ([COMPUTE] / [DATA] / [CODE]) and retrieve historical fixes.")

    with st.spinner("Initializing LLM Model ..."):
        llm = load_llm.copy()

    if 'agent' not in st.session_state:
        ## RAG-agent-call / creates OOS states
        tools_pack = [read_training_logs, run_shap_analysis, search_db_files, search_framework_docs]
        st.session_state.agent = get_debugging_agent(llm, tools_pack=tools_pack)
        st.session_state.chat_history = []
    
    if 'agent_call' not in st.session_state:
        st.session_state.agent_call = True   
        
        intitial_instruction = """A training run just finished.
        1. Read the training logs.
        2. If you see an anamoly, run the required tools.
        3. Query the framework docs and internal db files for a solution.
        4. Generate a comphrehensive root cause analysis report.
        """
        
    with st.spinner("Parallel: AI Agent analyzing the recent training run ... "):
        response = st.session_state.agent.invoke({"messages": [HumanMessage(content=intitial_instruction)]})
        final_report = response["messages"][-1].content
        st.session_state.chat_history.apppend(AIMessage(content=final_report))
        
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # HUMAN-IN-THE-LOOP CHAT
    if user_input := st.chat_input("Ask follow-up debugging questions..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        with st.spinner("Agent thinking..."):
            response = st.session_state.agent.invoke({"messages": st.session_state.chat_history})
            agent_reply = response["messages"][-1]
            st.chat_message("assistant").markdown(agent_reply.content)
            st.session_state.chat_history.append(agent_reply)

    
import streamlit as st
from data.load import *
from rag.embed import *
from rag.retrieve import *
from agent.logic import *


def stream_frontend(load_vector_embed, load_llm):
    st.set_page_config(page_title="AI-SYS Debugger", page_icon="⚙️")

    st.title(" < ML System Debugging Assist. > ")
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
                live_tool_results = 'No live tools data required.'
                
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
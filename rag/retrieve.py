from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def build_rag_chain(vector_store, llm): #query, live_metrics, 
    # Retrieve top 2 most similar past incidents
    retriever = vector_store.as_retriever(search_kwargs={"k": 2}) ## similarity_match and doc-stringified retrieval
    
    system_prompt = """You are an AI ML Site Reliability Engineer.
    Based on the following historical incidents from our company:
    
    <context>
    {context}
    </context>
    
    Live Tool Output / Telemetry:
    {live_metrics}
    
    Diagnose the user's issue: {query}
    
    Structure your response as follows:
    **Possible Causes:** (List based on context)
    **Suggested Fixes:** (Actionable steps based on context)
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # LCEL Chain: strInput (from input-dict) -> Retrieve context -> Inject to Prompt -> LLM -> String Output
    rag_chain = (
        {"context": (lambda k: k['query']) | retriever | format_docs, "query": RunnablePassthrough(), "live_metrics": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
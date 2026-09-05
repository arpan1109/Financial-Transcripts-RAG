import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from vector_store import load_existing_vector_store

load_dotenv()

def get_financial_qa_chain():
    vectorstore = load_existing_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Initialize the OpenAI GPT-OSS-20B model on Groq
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.0 
    )
    
    financial_prompt_template = """
    You are an expert financial equity research analyst.
    Use the following pieces of retrieved context to answer the user's question.
    
    Strict Rules:
    - If the context does not explicitly state the answer or the exact numbers, you MUST say "I do not have enough information to answer this."
    - Do not invent, guess, or extrapolate any financial metrics.
    - Quote the exact numbers as they appear in the text.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=financial_prompt_template, 
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain
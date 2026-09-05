import streamlit as st
from search import get_financial_qa_chain

st.set_page_config(page_title="Financial Earnings RAG", layout="centered")

st.title("📈 Financial Earnings Call Assistant")
st.markdown("Ask specific questions about revenue, margins, or executive statements based on the ingested transcripts.")

# Initialize the chain only once using Streamlit's cache to save memory
@st.cache_resource
def load_chain():
    return get_financial_qa_chain()

qa_chain = load_chain()

# Create a text input for the user's question
user_query = st.text_input("Enter your financial question:")

if st.button("Search Transcripts"):
    if user_query:
        with st.spinner("Analyzing transcripts..."):
            
            # Pass the query to the LangChain QA Chain
            response = qa_chain.invoke({"query": user_query})
            
            # Display the final LLM Answer
            st.subheader("Answer")
            st.write(response["result"])
            
            # Create a dropdown to show the exact sources used
            with st.expander("View Source Documents"):
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content)
                    st.markdown("---")
    else:
        st.warning("Please enter a question first.")
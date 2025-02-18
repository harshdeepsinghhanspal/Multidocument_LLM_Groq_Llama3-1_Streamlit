import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Load the persisted Chroma vector database
persist_directory = "doc_db"
embedding = HuggingFaceEmbeddings()
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectorstore.as_retriever()

#Groq API
GROQ_API_KEY = "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"

# Initialize LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=500
)

# Custom Prompt Template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Use only the following document excerpts:\n{context}\n\n"
             "Question: {question}\n"
             "Answer strictly based on the excerpts. If not found, say 'Answer not found!'."
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Streamlit UI
st.title("📄 PID Q&A Chatbot")

question = st.text_input("Ask me the question:")

if st.button("Get Answer"):
    if question.strip():
        response = qa_chain.invoke({"query":question})
        st.subheader("Answer:")
        st.write(response["result"])
    else:
        st.warning("Please enter a question.")
import os

from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate

GROQ_API_KEY = "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"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

loader = DirectoryLoader("data/", glob="*.docx", loader_cls=Docx2txtLoader)
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500
)

text_chunks = text_splitter.split_documents(documents)

persist_directory = "doc_db"

embedding = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding,
    persist_directory=persist_directory
)

retriever = vectorstore.as_retriever()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=500
)


custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Use only the following document excerpts:\n{context}\n\n"
             "Question: {question}\n"
             "Answer strictly based on the excerpts. If not found, say 'Not available in the document'."
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)
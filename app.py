import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("📄 RAG Chatbot - Student Assistant")

# Load embeddings + FAISS index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# STRICT PROMPT (IMPORTANT FOR GRADING)
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that answers questions using ONLY the provided context.

Follow these rules:
- If the answer is clearly found in the context, give a clear and complete answer.
- If the answer is NOT found in the context, then respond exactly:
"I’m sorry, I am only authorized to talk about the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
)

# Input box
query = st.text_input("Ask a question from the PDF:")

if query:
    # Retrieve TOP 4 chunks
    docs = db.similarity_search(query, k=4)

    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(context=context, question=query)

    response = llm.invoke(final_prompt)

    # Answer
    st.subheader("Answer")
    st.write(response.content)

    # Sources (top 3)
    st.subheader("Sources")
    for i, doc in enumerate(docs[:3]):
        st.write(f"Source {i+1}")
        st.write(doc.page_content[:300])
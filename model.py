from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrivalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """ Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say you don't know the answer. Don't try to make up an answer. 

Context: {}
Question: {question}

Only return the helpful answer and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Propt template for QA retrival for each vector store
    """

    prompt  = PromptTemplate(template=custom_prompt_template, input_variabls = [context, question])

    return prompt

def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

    return llm

def retrival_qa_chain(llm, prompt, db):
    qa_chain = RetrivalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {'k': 2}), chain_type_kwargs = {'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name = "/all-MiniLM-Lsentence-transformers6-v2",
        model_kwargs = {"device" : "cpu"}
        )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrival_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


final_result("What methods do you know for dimensionsionality reduction?")
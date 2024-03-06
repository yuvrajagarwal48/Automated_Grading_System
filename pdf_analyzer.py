from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "your api key"



def analyze_pdf(pdf):
    pdfreader = PdfReader(pdf)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()

    document_search = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    query = "give score out of 10 to each answer based on how much information it contains accoriding to the question"
    docs = document_search.similarity_search(query)
    analysis=chain.run(input_documents=docs, question=query)
    return analysis

# print(analyze_pdf('pdf4.pdf'))

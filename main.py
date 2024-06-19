import logging
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
import os
load_dotenv()

google_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

def semanticSearch(text, query):
    # llm = ChatGoogleGenerativeAI(model = "gemini-pro",temperature =0.7)
    # result = llm.invoke("What is a LLM?")
    # property

    #splitting the text from the pdf document
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n", " "],
        chunk_size = 200,
        chunk_overlap = 100,
    )

    split_text = text_splitter.split_text(text)

    #mapping to vectors
    model_name  = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device":'cpu'}
    encode_kwargs = {'normalize_embeddings':True}

    hf_embedding_model = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    # embeddings = [hf.embed_query(chunk) for chunk in split_text]
    # print(embeddings)


    #vector store for PDF 
    db = FAISS.from_texts(
        split_text,
        hf_embedding_model
    )

    #similarity searching
    embedding_vector = hf_embedding_model.embed_query(query)
    matches = db.similarity_search_by_vector(embedding_vector)
    

    #retriever
    retriever = BM25Retriever.from_texts(split_text)
    matches = retriever.invoke(query)
    return matches

    

def extractPDF(path):
    reader = PdfReader(path)
    num_pages = len(reader.pages)
    text = ""
    for i in range(num_pages):
        text += reader.pages[i].extract_text()
    return text

def main():
    path = "/Users/abhinavpappu/Documents/PersonalProjects/PDFSearcher/Lancet_20240618/Japan--health-after-the-earthquake_lancet.pdf"
    text = extractPDF(path)
    query = "how many people died"
    matches = semanticSearch(text, query)
    results = [match for match in matches]
    print(results)

if __name__ =='__main__':
    main()

import logging
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from annotated_text import annotated_text
import os

load_dotenv()

google_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


#performs the Semantic Search
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
    matches = db.similarity_search_with_score_by_vector(embedding_vector, k= 25)
    

    #retriever
    # retriever = BM25Retriever.from_texts(split_text)
    # matches = retriever.invoke(query)
    return matches

    

#used to extract text from pdf in a formate that can be formatte to Vector Stores
def extractPDF(path):
    reader = PdfReader(path)
    num_pages = len(reader.pages)
    
    text = ""
    for i in range(num_pages):
        text += reader.pages[i].extract_text()
    return text


#running the web page in streamlit
def runPage():
    st.set_page_config(
        page_title = "Magic Cntrl+f"
    )
    st.title("Magic Ctrl+F")
    pdf_file = st.file_uploader("Upload a PDF file", type = "pdf")
    search_query = st.text_input("Search Query")


    pdf_content = "" ##the text from the pdf 
    matches = []
    input_file = ""
    if pdf_file is not None:
        st.write(displayPDF(pdf_file))

        

    if search_query and pdf_file is not None:
        matches = semanticSearch(extractPDF(pdf_file), search_query)
        results = [match for match in matches]
        format_results = formatFilterResults(results, threshold = 1.0)
        switch_page("redirect")
        return format_results 
    

#used to extract the text from the pdf to dispaly in streamlit
def displayPDF(file):
    pdf_reader = PdfReader(file)
    content = ""
    for page in range(pdf_reader.get_num_pages()):
        content+= pdf_reader.get_page(page).extract_text()
    return content


def highlightText(textBlock, matches):
    text = "Hello my name is Bob"
    annotated_text(
    (text,"", "#8cff66")
)
        

def formatFilterResults(matches,threshold ):
    format_matches = []
    for doc,score in matches:
        if score < threshold:
            format_matches.append(doc.page_content)
    return format_matches
    

def main():
    results = runPage()
    st.write(results)

    # path = "/Users/abhinavpappu/Documents/PersonalProjects/PDFSearcher/Lancet_20240618/Japan--health-after-the-earthquake_lancet.pdf"
    # text = extractPDF(path)
    # query = "how many people died"
    # matches = semanticSearch(text, query)
    # results = [match for match in matches]
    # format_results = formatFilterResults(results, threshold = 0.7)
    # highlightText(displayPDF(path),format_results )
    
    

if __name__ =='__main__':
    main()


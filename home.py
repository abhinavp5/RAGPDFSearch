from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
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
    st.set_page_config(page_title="Magic Ctrl+F")
    st.title("Magic Ctrl+F")

    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'pdf_file' not in st.session_state:
        st.session_state.pdf_file = None
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    search_query = st.text_input("Search Query")

    if uploaded_file:
        st.session_state.pdf_file = uploaded_file
        st.session_state.show_results = False  
        st.session_state.search_query = ""

    #when the pdf is entered and the query is entered
    if search_query and st.session_state.pdf_file:
        st.session_state.search_query = search_query
        st.session_state.show_results = True
        matches = semanticSearch(extractPDF(st.session_state.pdf_file), search_query)
        format_results = formatFilterResults(matches, threshold=1.0)
        
        #clearing page to dispaly matches
        placeholder = st.empty()
        placeholder.empty()
        with placeholder.container():
            displayResults(displayPDF(st.session_state.pdf_file),format_results)

    #when the pdf is uploaded but query is not entered yet
    elif st.session_state.pdf_file and not st.session_state.show_results:
        # Display PDF content
        text_content = displayPDF(st.session_state.pdf_file)
        placeholder = st.empty()
        placeholder.empty()
        with placeholder.container():
            for page_num, text in text_content.items():
                st.subheader(f"{page_num}")
                st.write(text)

#used to extract the text from the pdf to dispaly in streamlit
def displayPDF(file):
    pdf_reader = PdfReader(file)
    ###this version works###
    # text_content = ""
    # for page in range(pdf_reader.get_num_pages()):
    #     content+= pdf_reader.get_page(page).extract_text()

    # return content
    text_content = {}
    #page is just an index value and getPage(page) is the actual content in the page 
    for page in range(pdf_reader.get_num_pages()):
        text_content[pdf_reader.get_page(page).extract_text()] = page 

    return text_content
def mergePages(pages):
    text = ""
    for content in pages.keys():
        text+=content +'\n'
    return text
        

def displayResults(pages, matches):
    text = mergePages(pages)
    st.subheader("These were the results in response to your query:")
    st.write(formatSentences(matches))  
    st.subheader("Annotations in PDF")
    st.write(text)

    segments = []
    last_index = 0  
    # Process each match and find its occurrences in the text
    for match in matches:
        start_index = text.find(match, last_index)  # Find the first occurrence after the last_index
        while start_index != -1:  # While there are still occurrences of the match
            if start_index >= last_index:
                if start_index > last_index:
                    segments.append(text[last_index:start_index])  
                segments.append((match, "", "#8cff66"))  
                last_index = start_index + len(match)  
            start_index = text.find(match, last_index)  

    # Add any remaining text after the last match
    if last_index < len(text):
        segments.append(text[last_index:])

    # Display the annotated text with highlights
    annotated_text(*segments)

def formatSentences(matches):
    llm = ChatGoogleGenerativeAI(model = "gemini-pro")
    result = llm.invoke(f"reformat the content in {matches} into understadable content")
    return result.content

def formatFilterResults(matches,threshold):
    format_matches = []
    for doc,score in matches:
        if score < threshold:
            format_matches.append(doc.page_content)
    if not format_matches:
        format_matches.append("There were no matches to your query")
    return format_matches

if __name__ =='__main__':
    runPage()


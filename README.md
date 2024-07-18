# RAGPDF Search
Allows you to enter a pdf, and enter a query and performs a semantic search based on the content in the pdf to give the most relevant answers. 
Instructions to run
- clone the repo ``` git clone https://github.com/abhinavp5/RAGPDFSearch ```
- cd into the repo ``` cd RAGPDFSearch ```
- activate virtual envirnoment ```python3 -m venv venv ``` ```source venv/bin/activate```
- install needed dependencies ``` pip install -r requirements.txt```
- setup api keys for gemini as ```GOOGLE_API_KEY = ``` in a ```.env``` file
- run stremalit app with ```streamlit run home.py ```

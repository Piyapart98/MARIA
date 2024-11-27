import os
import asyncio
import pickle
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore", category=DeprecationWarning)
'''
This code loads PDFs from folder "documents", split into chunks and embed them into vectorstore.
Then, the vectorstore is saved into vectorstore.pkl in the folder "vectorstore", ready to load and transform into a retriever in a RAG pipeline.

Created by: Piyapart Buttamart
Created date: 19 November 2024
'''

async def extract_text_from_pdfs(folder_path):
    pages = []
    for file_name in os.listdir(folder_path):
        
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            try:
                loader = PyPDFLoader(file_path)
                print(f"Extracting {file_name}")
            except Exception as e:
                print(f"Failed to extract text from {file_name}: {e}")
                continue

            # Use async for inside the async function
            async for page in loader.alazy_load():
                pages.append(page)
    return pages

# Main logic to call the async function
folder_path = "documents/"
pages = asyncio.run(extract_text_from_pdfs(folder_path))

embedding_wrapper = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore_page = InMemoryVectorStore.from_documents(pages, embedding_wrapper)
print("All chuncks are embedded and stored in vectorstore")

# Save vectorstore object to a file
vector_file = 'vectorstore/vectorstore.pkl'
with open(vector_file, "wb") as f:
    pickle.dump(vectorstore_page, f)
print("Vectorstore_page saved using pickle.")

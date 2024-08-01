import os
import sys
import docx2txt
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
UPLOAD_PATH = 'chat/static/upload/'

class llmVectorDB():
    def parse_documents(self, path):
        document = []
        for dirname, _, filenames in os.walk(UPLOAD_PATH):
            print(filenames)
            for filename in filenames:
                if filename.endswith(".pdf"):
                    pdf_path=os.path.join(dirname, filename)
                    loader=PyPDFLoader(pdf_path)
                    document.extend(loader.load())
                elif filename.endswith('.docx') or filename.endswith('.doc'):
                    doc_path=os.path.join(dirname, filename)
                    loader=Docx2txtLoader(doc_path)
                    document.extend(loader.load())
        return document
    
    def split_docs_create_embeddings(self,document):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        texts = text_splitter.split_documents(document)   
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cuda'}, cache_folder="/home/najeed/NED_Chatbot/ai_chatbot/embeddings_path")
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
   

    def add_documents_to_vector_DB(self):
        documents = self.parse_documents(UPLOAD_PATH)
        self.split_docs_create_embeddings(documents)
        return True

    # Create vector database
    def create_vector_db(self):
        documents = self.parse_documents(DATA_PATH)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                    chunk_overlap=64)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cuda'}, cache_folder="/home/najeed/NED_Chatbot/ai_chatbot/embeddings_path")

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
